from datetime import datetime, timezone
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]


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

    @model_validator(mode="after")
    def consistent_hour(self):
        self.timestamp = self.timestamp.astimezone(timezone.utc)
        if self.hour_of_day != self.timestamp.hour:
            raise ValueError("hour_of_day must match timestamp in UTC")
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
