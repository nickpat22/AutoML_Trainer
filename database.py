"""Small MongoDB helper for the AutoML training history."""


import os
import streamlit as st
from datetime import datetime, timezone
from pymongo import MongoClient


def get_setting(name, default):
    # Environment variables take priority.
    value = os.getenv(name)
    if value:
        return value

    # Read secrets configured locally or on Streamlit Cloud.
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    return default


MONGO_URI = get_setting(
    "MONGODB_URI",
    "mongodb://localhost:27017/"
)

DB_NAME = get_setting(
    "MONGODB_DATABASE",
    "automl_database"
)

_client = None


def _collection():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2500)
    _client.admin.command("ping")
    return _client[DB_NAME]["training_history"]


def save_training(data):
    """Save one training record; let the UI handle connection errors."""
    document = dict(data)
    document["created_at"] = datetime.now(timezone.utc)
    result = _collection().insert_one(document)
    return str(result.inserted_id)


def get_training_history(limit=100):
    """Return newest history records as normal Python dictionaries."""
    records = list(_collection().find().sort("created_at", -1).limit(limit))
    for record in records:
        record["_id"] = str(record["_id"])
    return records
