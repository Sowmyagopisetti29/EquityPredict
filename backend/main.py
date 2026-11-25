
from fastapi import FastAPI, Query, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os, base64, hashlib, hmac, json, re, time
import yfinance as yf
from threading import Lock
from copy import deepcopy

from prediction import predict_stock_price

from database import (
    initialize_database,
    save_prediction,
    get_prediction_history,
    get_stock_prediction_history,
    create_user,
    get_user_by_email
)


app = FastAPI(title="EquityPredict")

# Authentication uses a signed bearer token and PBKDF2 password hashing.
AUTH_SECRET = os.getenv("AUTH_SECRET")
if not AUTH_SECRET and os.getenv("RENDER"):
    raise RuntimeError("AUTH_SECRET must be configured in the Render environment.")
if not AUTH_SECRET:
    # Local development fallback only. Set a strong AUTH_SECRET in production.
    AUTH_SECRET = "local-development-only-change-this-secret"

bearer_scheme = HTTPBearer(auto_error=False)

class AuthPayload(BaseModel):
    email: str
    password: str

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310000)
    return base64.urlsafe_b64encode(salt).decode() + "." + base64.urlsafe_b64encode(digest).decode()

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_text, digest_text = stored_hash.split(".", 1)
        salt = base64.urlsafe_b64decode(salt_text.encode())
        expected = base64.urlsafe_b64decode(digest_text.encode())
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310000)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False

def create_access_token(user_id: int) -> str:
    payload = json.dumps({"sub": user_id, "exp": int(time.time()) + 60 * 60 * 24 * 7}, separators=(",", ":")).encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    signature = hmac.new(AUTH_SECRET.encode(), encoded.encode(), hashlib.sha256).digest()
    signed = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return encoded + "." + signed

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    try:
        encoded, signature = credentials.credentials.split(".", 1)
        expected = hmac.new(AUTH_SECRET.encode(), encoded.encode(), hashlib.sha256).digest()
        supplied = base64.urlsafe_b64decode((signature + "=" * (-len(signature) % 4)).encode())
        if not hmac.compare_digest(expected, supplied):
            raise ValueError("Invalid signature")
        payload = json.loads(base64.urlsafe_b64decode((encoded + "=" * (-len(encoded) % 4)).encode()))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("Expired token")
        return {"id": int(payload["sub"])}
    except Exception:
        raise HTTPException(status_code=401, detail="Your session is invalid or expired. Please log in again.")



# Initialize database when FastAPI starts
@app.on_event("startup")
def startup_event():
    initialize_database()


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://192.168.0.160:5173",
        "https://equitypredict.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Prediction cache configuration
CACHE_DURATION = 60

prediction_cache = {}
cache_lock = Lock()


# Function to get cached prediction
def get_cached_prediction(symbol: str):

    symbol = symbol.strip().upper()
    current_time = time.monotonic()

    with cache_lock:

        cached_data = prediction_cache.get(symbol)

        # Return cached prediction if it is still valid
        if cached_data:
            cached_result, cached_time = cached_data

            if current_time - cached_time < CACHE_DURATION:
                return deepcopy(cached_result)

        # Generate a fresh prediction
        result = predict_stock_price(symbol)

        # Store prediction in cache
        prediction_cache[symbol] = (
            deepcopy(result),
            current_time
        )

        return result


# Account registration and login
@app.post("/auth/register")
def register(payload: AuthPayload):
    email = payload.email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must contain at least 8 characters.")
    if len(payload.password) > 128:
        raise HTTPException(status_code=400, detail="Password is too long.")
    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    user_id = create_user(email, hash_password(payload.password))
    return {"access_token": create_access_token(user_id), "token_type": "bearer", "email": email}

@app.post("/auth/login")
def login(payload: AuthPayload):
    user = get_user_by_email(payload.email)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    return {"access_token": create_access_token(user["id"]), "token_type": "bearer", "email": user["email"]}

@app.get("/auth/me")
def current_account(user=Depends(get_current_user)):
    return {"user_id": user["id"]}

# Home endpoint
@app.get("/")
def home():

    return {
        "message": "Welcome to EquityPredict!",
        "status": "Backend is running successfully"
    }


# Historical stock data endpoint
@app.get("/stock/{symbol}")
def get_stock_data(symbol: str):

    try:
        normalized_symbol = symbol.strip().upper()

        stock = yf.Ticker(normalized_symbol)
        data = stock.history(period="1mo")

        if data.empty:
            return {"error": "No stock data found"}

        # Convert the DataFrame index into a normal column
        history_data = data.reset_index()

        # Identify the date column returned by yfinance
        date_column = next(
            (
                column
                for column in ["Date", "Datetime"]
                if column in history_data.columns
            ),
            None
        )

        if date_column is None:
            return {
                "error": "Historical stock data does not contain a date column."
            }

        # Preserve the original exchange calendar date.
        # Do not convert timezone-aware timestamps to UTC.
        history_data[date_column] = history_data[date_column].apply(
            lambda value: (
                value.strftime("%Y-%m-%d")
                if hasattr(value, "strftime")
                else str(value).split("T")[0].split(" ")[0]
            )
        )

        return {
            "stock": normalized_symbol,
            "history": history_data.to_dict(orient="records")
        }

    except Exception as error:
        return {
            "error": "Unable to fetch stock data: " + str(error)
        }


# Single stock price prediction endpoint
@app.get("/predict/{symbol}")
def predict_stock(symbol: str, user=Depends(get_current_user)):

    try:
        result = get_cached_prediction(symbol)

        # Save successful prediction in database
        save_prediction(result, user["id"])

        return result

    except ValueError as error:
        return {"error": str(error)}

    except Exception as error:
        return {
            "error": "Prediction failed: " + str(error)
        }


# Multiple stock prediction endpoint
@app.get("/predict-multiple")
def predict_multiple_stocks(symbols: str, user=Depends(get_current_user)):

    stock_symbols = symbols.split(",")

    results = []

    for symbol in stock_symbols:

        symbol = symbol.strip().upper()

        if not symbol:
            continue

        try:
            result = get_cached_prediction(symbol)

            # Save successful prediction in database
            save_prediction(result, user["id"])

            results.append(result)

        except Exception as error:
            results.append({
                "stock": symbol,
                "error": str(error)
            })

    return {
        "total_stocks": len(results),
        "results": results
    }


# Retrieve all prediction history
@app.get("/predictions/history")
def prediction_history(
    limit: int = Query(default=50, ge=1, le=500),
    user=Depends(get_current_user),
):

    predictions = get_prediction_history(user["id"], limit)

    return {
        "total_records": len(predictions),
        "predictions": predictions
    }


# Retrieve prediction history for one stock
@app.get("/predictions/history/{symbol}")
def stock_prediction_history(
    symbol: str,
    limit: int = Query(default=50, ge=1, le=500),
    user=Depends(get_current_user),
):

    predictions = get_stock_prediction_history(
        user["id"],
        symbol,
        limit
    )

    return {
        "stock": symbol.strip().upper(),
        "total_records": len(predictions),
        "predictions": predictions
    }
