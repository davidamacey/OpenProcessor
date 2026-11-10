"""Wire models for the non-stream event routes. Leaf module."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EventPublishAck(BaseModel):
    ok: Literal[True]


class EventStreamStats(BaseModel):
    """Ops counters for the bound project's event stream."""

    subscribers: int = Field(description="This process's local subscribers of the project stream.")
    clients_dropped: int = Field(description='Clients this process dropped for falling behind.')
    listener_running: bool
    pubsub_connected: bool = Field(description='The Valkey subscription is up.')
