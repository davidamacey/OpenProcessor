"""A combine copies `capture.*` verbatim, except a position the target's location policy does not keep."""

from __future__ import annotations

from typing import Any

import pytest

from .test_combine_execute import MAPPING
from .world import World, run_job


CAPTURE: dict[str, Any] = {
    'exif_status': 'ok',
    'parser': 'capture@1',
    'taken_local': '2024-03-05T14:22:11',
    'taken_utc': '2024-03-05T13:22:11',
    'camera_make': 'Acme',
    'has_location': True,
    'location': {'lat': 47.123456, 'lon': 8.987654},
    'altitude_m': 410.0,
    'location_precision': 'exact',
}


def _world(world: World) -> str:
    world.project('cars-a', ['car', 'truck'])
    world.project('cars-b', ['sedan', 'lorry'])
    image_id, _ = world.add_image('cars-a', items=[{'cls': 'car', 'bbox': [0.1, 0.1, 0.5, 0.5]}])
    world.images('cars-a')[image_id]['capture'] = dict(CAPTURE)
    world.items('cars-a')[next(iter(world.items('cars-a')))].update(
        capture_taken_local=CAPTURE['taken_local'], capture_taken_utc=CAPTURE['taken_utc']
    )
    return image_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('policy', 'location', 'precision'),
    [
        ('store', {'lat': 47.123456, 'lon': 8.987654}, 'exact'),
        ('store_coarse', {'lat': 47.12, 'lon': 8.99}, 'coarse'),
        ('drop', None, None),
    ],
)
async def test_capture_is_copied_and_the_position_follows_the_target_policy(
    world: World, policy: str, location: dict[str, float] | None, precision: str | None
) -> None:
    _world(world)
    run, _ = await run_job(
        world, world.request(['cars-a', 'cars-b'], MAPPING), location_policy=policy
    )
    assert run.status == 'completed', run.error

    [image] = world.images('combined').values()
    capture = image['capture']
    assert capture['taken_utc'] == CAPTURE['taken_utc']
    assert capture['camera_make'] == 'Acme'
    assert capture['has_location'] is True
    assert capture.get('location') == location
    assert capture.get('location_precision') == precision
    if policy != 'store':
        assert 'altitude_m' not in capture
    [item] = world.items('combined').values()
    assert item['capture_taken_utc'] == CAPTURE['taken_utc']
    assert '47.1' not in str(item)
    # the source project is untouched
    [source] = world.images('cars-a').values()
    assert source['capture']['location'] == {'lat': 47.123456, 'lon': 8.987654}
