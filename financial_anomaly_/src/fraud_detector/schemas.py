from datetime import datetime, timezone
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]


class BehavioralFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    transactions_last_5m: int = Field(ge=0, le=10_000_000, strict=True)
    transactions_last_day: int = Field(ge=0, le=10_000_000, strict=True)
    customer_prior_transactions: int = Field(ge=0, le=100_000_000, strict=True)
    customer_mean_amount: float = Field(ge=0, le=1_000_000)
    customer_amount_std: float = Field(ge=0, le=1_000_000)
    amount_to_customer_mean: float = Field(ge=0, le=100_000_000)
    amount_zscore: float = Field(ge=-1000, le=1000)
    is_new_terminal: int = Field(ge=0, le=1, strict=True)
    customer_terminal_transactions: int = Field(ge=0, le=100_000_000, strict=True)
    terminal_transactions_last_hour: int = Field(ge=0, le=10_000_000, strict=True)
    terminal_mean_amount: float = Field(ge=0, le=1_000_000)
    amount_to_terminal_mean: float = Field(ge=0, le=100_000_000)
    hour_sin: float = Field(ge=-1, le=1)
    hour_cos: float = Field(ge=-1, le=1)

    @classmethod
    def from_features(cls, values):
        integer_fields = {
            "transactions_last_5m",
            "transactions_last_day",
            "customer_prior_transactions",
            "is_new_terminal",
            "customer_terminal_transactions",
            "terminal_transactions_last_hour",
        }
        return cls(
            **{
                name: int(values[name]) if name in integer_fields else values[name]
                for name in cls.model_fields
            }
        )


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    transaction_id: Identifier
    source_transaction_id: Identifier | None = None
    customer_id: Identifier
    terminal_id: Identifier
    amount: float = Field(gt=0, le=1_000_000)
    transactions_last_hour: int = Field(ge=0, le=10_000, strict=True)
    customer_history_days: float = Field(ge=0, le=100_000)
    hour_of_day: int = Field(ge=0, le=23, strict=True)
    timestamp: AwareDatetime
    behavioral_features: BehavioralFeatures | None = None

    @model_validator(mode="after")
    def consistent_hour(self):
        self.timestamp = self.timestamp.astimezone(timezone.utc)
        if self.hour_of_day != self.timestamp.hour:
            raise ValueError("hour_of_day must match timestamp in UTC")
        if self.behavioral_features is not None:
            b = self.behavioral_features
            if (
                not b.transactions_last_5m
                <= self.transactions_last_hour
                <= b.transactions_last_day
                <= b.customer_prior_transactions
            ):
                raise ValueError("Behavioral window counts are inconsistent")
            if b.customer_terminal_transactions > b.customer_prior_transactions:
                raise ValueError("Terminal visits exceed customer history")
            if b.is_new_terminal != int(b.customer_terminal_transactions == 0):
                raise ValueError("Terminal novelty conflicts with visit count")
        return self


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    transaction_id: Identifier
    source_transaction_id: Identifier | None = None
    timestamp: datetime
    fraud_probability: float = Field(ge=0, le=1)
    is_fraud: bool
    model_version: Identifier


class AuditRecord(Prediction):
    timestamp: AwareDatetime
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
