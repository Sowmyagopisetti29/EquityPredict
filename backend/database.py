
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone


# Database file location
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "equitypredict.db"


# Connect to SQLite database
def get_connection():

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    return connection


# Create prediction history table
def initialize_database():

    connection = get_connection()

    try:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                stock TEXT NOT NULL,

                latest_close REAL NOT NULL,

                predicted_return_percent REAL NOT NULL,

                predicted_next_close REAL NOT NULL,

                best_model TEXT,

                model_evaluation TEXT,

                created_at TEXT NOT NULL

            )
        """)

        connection.commit()

    finally:

        connection.close()


# Save prediction results
def save_prediction(result):

    if not result or result.get("error"):
        return None

    connection = get_connection()

    try:

        created_at = datetime.now(timezone.utc).isoformat()

        model_evaluation = json.dumps(
            result.get("model_evaluation", {})
        )

        cursor = connection.execute("""

            INSERT INTO prediction_history (

                stock,
                latest_close,
                predicted_return_percent,
                predicted_next_close,
                best_model,
                model_evaluation,
                created_at

            )

            VALUES (?, ?, ?, ?, ?, ?, ?)

        """, (

            result["stock"],
            result["latest_close"],
            result["predicted_return_percent"],
            result["predicted_next_close"],
            result.get("model_evaluation", {}).get("best_model"),
            model_evaluation,
            created_at

        ))

        connection.commit()

        return cursor.lastrowid

    finally:

        connection.close()


# Retrieve recent prediction history
def get_prediction_history(limit=50):

    connection = get_connection()

    try:

        rows = connection.execute("""

            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?

        """, (limit,)).fetchall()

        results = []

        for row in rows:

            item = dict(row)

            item["model_evaluation"] = json.loads(
                item["model_evaluation"] or "{}"
            )

            results.append(item)

        return results

    finally:

        connection.close()


# Retrieve prediction history for one stock
def get_stock_prediction_history(symbol, limit=50):

    connection = get_connection()

    try:

        rows = connection.execute("""

            SELECT *
            FROM prediction_history
            WHERE stock = ?
            ORDER BY id DESC
            LIMIT ?

        """, (symbol.strip().upper(), limit)).fetchall()

        results = []

        for row in rows:

            item = dict(row)

            item["model_evaluation"] = json.loads(
                item["model_evaluation"] or "{}"
            )

            results.append(item)

        return results

    finally:

        connection.close()
