"""The names of the configurations that exist without a stored document.

A built-in or file prompt pack, an env or registered region profile and the ``env`` VLM endpoint
are activated by name alone. Activating one stores its body as a system revision (so the
activation's foreign key points at a real document), but the configuration stays what it was: it
is not listed as a stored config, its activation shows no revision, and the live body (the file
on disk, the environment) is what runs. Anything stored under one of these names is such a
system revision, because a saved config can never take a registered name.
"""

from __future__ import annotations

from typing import Literal


def registered_names(axis: str) -> frozenset[str]:
    """The registered names of ``axis`` (empty for an axis with none)."""
    if axis == 'prompt_pack':
        from src.services.labeling.vlm_prompt_resolution import registered_prompt_packs

        return frozenset(registered_prompt_packs())
    if axis == 'detection_profile':
        from src.services.detection.profile_registry import registered_profile_names

        return registered_profile_names()
    if axis == 'vlm':
        from src.services.labeling.vlm_endpoints import ENV_ENDPOINT_NAME

        return frozenset({ENV_ENDPOINT_NAME})
    return frozenset()


def is_registered(
    axis: Literal['prompt_pack', 'detection_profile', 'vlm'] | str, name: str | None
) -> bool:
    return name is not None and name in registered_names(axis)


__all__ = ['is_registered', 'registered_names']
