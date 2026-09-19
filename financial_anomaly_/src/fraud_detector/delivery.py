"""Bounded HTTP delivery shared by fixed-file replay and synthetic streaming."""

import random
import time

import httpx


def post_transaction(client, transaction, attempts=6, sleep=time.sleep):
    for attempt in range(attempts):
        try:
            response = client.post("/predict", json=transaction)
            if response.status_code < 400:
                return response.json()
            if response.status_code not in (429, 500, 502, 503, 504):
                raise ValueError(f"Permanent replay error HTTP {response.status_code}")
        except httpx.TransportError:
            pass
        if attempt + 1 < attempts:
            sleep(min(30, 2**attempt) + random.random())
    raise RuntimeError("Replay retry budget exhausted; checkpoint retained")
