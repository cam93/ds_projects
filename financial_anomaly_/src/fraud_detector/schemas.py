from datetime import datetime

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(gt=0, le=1_000_000)
    account_age_days: int = Field(ge=0, le=100_000)
    transactions_last_hour: int = Field(ge=0, le=10_000)
    is_international: bool
    timestamp: datetime


class Prediction(BaseModel):
    transaction_id: str
    timestamp: datetime
    fraud_probability: float = Field(ge=0, le=1)
    is_fraud: bool
    model_version: str
