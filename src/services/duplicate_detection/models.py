"""Result record of the core duplicate group queries."""

from dataclasses import dataclass
from typing import Any


@dataclass
class DuplicateGroup:
    """A group of near-duplicate images."""

    group_id: str
    primary_image_id: str
    primary_image_path: str
    member_count: int
    members: list[dict[str, Any]]
