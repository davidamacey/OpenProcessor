"""Sync publish refuses to run where it would block the loop or lose the write."""

from __future__ import annotations

import pytest

from src.services.events.publish import PublishInEventLoopError, publish


pytestmark = pytest.mark.db


@pytest.mark.asyncio
async def test_publish_inside_a_running_loop_raises_and_writes_nothing() -> None:
    with pytest.raises(PublishInEventLoopError, match='apublish'):
        publish('crop.created', {'crop_id': 'x'}, project='anything')
