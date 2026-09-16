"""Seeded synthetic transactions in the Handbook source schema (not its exact simulator)."""

import math
import random
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, ConfigDict, Field

COLUMNS = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
)
GENERATOR_VERSION = 1


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    seed: int = Field(default=7, ge=0, le=2**31 - 1, strict=True)
    customers: int = Field(default=1000, ge=1, le=10000, strict=True)
    terminals: int = Field(default=100, ge=1, le=10000, strict=True)
    transactions_per_customer_day: float = Field(default=3.0, gt=0, le=20)
    terminal_compromise_rate: float = Field(default=0.02, ge=0, le=1)
    customer_compromise_rate: float = Field(default=0.005, ge=0, le=1)
    unusual_purchase_fraud_rate: float = Field(default=0.002, ge=0, le=1)
    drift_day: int | None = Field(default=None, ge=1, strict=True)
    drift_multiplier: float = Field(default=1.35, ge=0.1, le=5)


def parse_start(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if len(value) == 10:  # A calendar date explicitly means midnight UTC.
        stamp = stamp.replace(tzinfo=timezone.utc)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Start timestamp needs a timezone, or use YYYY-MM-DD for midnight UTC")
    stamp = stamp.astimezone(timezone.utc)
    if stamp.time().isoformat() != "00:00:00":
        raise ValueError(
            "Start must be midnight UTC so daily spending patterns retain their meaning"
        )
    return stamp


def poisson(rng, mean):
    # Knuth's sampler: daily means are small and bounded by configuration.
    limit, product, count = math.exp(-mean), 1.0, 0
    while product > limit:
        product *= rng.random()
        count += 1
    return max(0, count - 1)


class TransactionWorld:
    def __init__(self, config: SimulationConfig, start: datetime):
        self.config = config
        self.start = parse_start(start.isoformat())
        rng = random.Random(f"fraud-simulator-v{GENERATOR_VERSION}:{config.seed}:profiles")
        self.profiles = [
            {
                "mean_amount": math.exp(rng.uniform(math.log(8), math.log(150))),
                "activity": rng.uniform(0.4, 1.6),
                "hour": rng.uniform(9, 20),
                "favorites": rng.sample(range(config.terminals), min(5, config.terminals)),
            }
            for _ in range(config.customers)
        ]

    def day(self, day):
        if not 0 <= day < 36500:
            raise ValueError("Simulation supports day indices 0..36499")
        cfg = self.config
        rng = random.Random(f"fraud-simulator-v{GENERATOR_VERSION}:{cfg.seed}:day:{day}")
        attacks = random.Random(f"fraud-simulator-v{GENERATOR_VERSION}:{cfg.seed}:week:{day // 7}")
        # Persistent compromise groups rotate weekly; labels are never generated from model scores.
        bad_terminals = {
            i for i in range(cfg.terminals) if attacks.random() < cfg.terminal_compromise_rate
        }
        bad_customers = {
            i for i in range(cfg.customers) if attacks.random() < cfg.customer_compromise_rate
        }
        stamp = self.start + timedelta(days=day)
        weekend = 0.8 if stamp.weekday() >= 5 else 1.0
        drift = cfg.drift_multiplier if cfg.drift_day is not None and day >= cfg.drift_day else 1.0
        rows = []

        def append(customer, terminal, second, amount, scenario):
            rows.append(
                {
                    "TRANSACTION_ID": 0,
                    "TX_DATETIME": (stamp + timedelta(seconds=second))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "CUSTOMER_ID": customer,
                    "TERMINAL_ID": terminal,
                    "TX_AMOUNT": round(min(999999.99, max(0.01, amount)), 2),
                    "TX_TIME_SECONDS": day * 86400 + second,
                    "TX_TIME_DAYS": day,
                    "TX_FRAUD": int(scenario != 0),
                    "TX_FRAUD_SCENARIO": scenario,
                }
            )

        for customer, profile in enumerate(self.profiles):
            count = poisson(rng, cfg.transactions_per_customer_day * profile["activity"] * weekend)
            for _ in range(count):
                hour = (
                    rng.gauss(profile["hour"], 3) % 24 if rng.random() < 0.9 else rng.uniform(0, 24)
                )
                second = min(86399, int(hour * 3600))
                terminal = (
                    rng.choice(profile["favorites"])
                    if rng.random() < 0.85
                    else rng.randrange(cfg.terminals)
                )
                amount = rng.lognormvariate(math.log(profile["mean_amount"]), 0.6) * drift
                scenario = 0
                if rng.random() < cfg.unusual_purchase_fraud_rate:
                    amount *= rng.uniform(3, 9)
                    scenario = 1
                elif terminal in bad_terminals and rng.random() < 0.65:
                    scenario = 2  # Deliberately hard to detect using the current four features.
                elif rng.random() < 0.015:
                    amount *= rng.uniform(
                        3, 7
                    )  # Legitimate large purchases prevent a trivial rule.
                append(customer, terminal, second, amount, scenario)
            if customer in bad_customers:
                burst_start = rng.randrange(0, 85500)
                terminal = rng.randrange(cfg.terminals)
                for offset in range(rng.randint(3, 8)):
                    amount = rng.lognormvariate(math.log(profile["mean_amount"] * 2), 0.8) * drift
                    append(customer, terminal, burst_start + offset * 45, amount, 3)
        rows.sort(key=lambda row: (row["TX_TIME_SECONDS"], row["CUSTOMER_ID"]))
        if len(rows) >= 1_000_000:
            raise ValueError("Too many daily events for the transaction ID namespace")
        for index, row in enumerate(rows):
            row["TRANSACTION_ID"] = day * 1_000_000 + index
        # Match the ingestion pipeline's lexical-ID tie break at equal timestamps.
        rows.sort(key=lambda row: (row["TX_TIME_SECONDS"], str(row["TRANSACTION_ID"])))
        return rows
