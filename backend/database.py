
import os
import json
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
    Text,
    select,
    desc,
    insert,
)


# Database configuration
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "equitypredict.db"

DATABASE_URL = os.getenv("DATABASE_URL")

# Use local SQLite if DATABASE_URL is not configured
if DATABASE_URL:
    # Support older PostgreSQL URL formats
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgres://", "postgresql://", 1
        )

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )
else:
    from sqlalchemy.engine import URL

    local_url = URL.create(
        "sqlite",
        database=str(DATABASE_PATH),
    )

    engine = create_engine(
        local_url,
        connect_args={"check_same_thread": False},
    )


# Define database table
metadata = MetaData()

prediction_history = Table(
    "prediction_history",
    metadata,

    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("stock", String(20), nullable=False),
    Column("latest_close", Float, nullable=False),
    Column("predicted_return_percent", Float, nullable=False),
    Column("predicted_next_close", Float, nullable=False),
    Column("best_model", String(100)),
    Column("model_evaluation", Text),
    Column("created_at", String(100), nullable=False),
)


# Get database connection
def get_connection():
    return engine.connect()


# Create prediction history table
def initialize_database():
    metadata.create_all(engine)


# Save prediction results
def save_prediction(result):

    if not result or result.get("error"):
        return None

    model_evaluation = json.dumps(
        result.get("model_evaluation", {})
    )

    created_at = datetime.now(timezone.utc).isoformat()

    statement = insert(prediction_history).values(
        stock=result["stock"],
        latest_close=result["latest_close"],
        predicted_return_percent=result["predicted_return_percent"],
        predicted_next_close=result["predicted_next_close"],
        best_model=result.get("model_evaluation", {}).get("best_model"),
        model_evaluation=model_evaluation,
        created_at=created_at,
    )

    with engine.begin() as connection:
        cursor = connection.execute(statement)
        return cursor.inserted_primary_key[0]


# Retrieve recent prediction history
def get_prediction_history(limit=50):

    statement = (
        select(prediction_history)
        .order_by(desc(prediction_history.c.id))
        .limit(limit)
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

        results = []

        for row in rows:
            item = dict(row)

            item["model_evaluation"] = json.loads(
                item["model_evaluation"] or "{}"
            )

            results.append(item)

        return results


# Retrieve prediction history for one stock
def get_stock_prediction_history(symbol, limit=50):

    statement = (
        select(prediction_history)
        .where(
            prediction_history.c.stock == symbol.strip().upper()
        )
        .order_by(desc(prediction_history.c.id))
        .limit(limit)
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

        results = []

        for row in rows:
            item = dict(row)

            item["model_evaluation"] = json.loads(
                item["model_evaluation"] or "{}"
            )

            results.append(item)

        return results
