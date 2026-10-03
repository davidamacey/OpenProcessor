"""Where an exporter's generated Triton ``config.pbtxt`` goes.

``models/*/config.pbtxt`` is tracked and often hand-tuned (instance counts,
batching), so a regenerated one never replaces an existing file that differs
unless the caller says so; it is written next to it as ``config.pbtxt.generated``
(git-ignored) for a diff instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)

GENERATED_SUFFIX = '.generated'


def write_generated_config(config_path: Path, content: str, *, overwrite: bool = False) -> Path:
    """Write ``content`` for ``config_path`` and return the file written.

    A missing file is created; a byte-identical one is left alone; a differing
    one is replaced only with ``overwrite``, otherwise the content goes to
    ``<name>.generated`` beside it."""
    if config_path.exists():
        if config_path.read_text() == content:
            logger.info(f'Triton config unchanged, not rewriting: {config_path}')
            return config_path
        if not overwrite:
            side = config_path.with_name(config_path.name + GENERATED_SUFFIX)
            side.write_text(content)
            logger.warning(
                f'{config_path} differs from the generated config; wrote {side} instead '
                '(pass --overwrite-config to replace the tracked file)'
            )
            return side
    config_path.write_text(content)
    logger.info(f'Generated Triton config: {config_path}')
    return config_path
