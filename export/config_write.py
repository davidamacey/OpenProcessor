"""Where an exporter's generated Triton ``config.pbtxt`` goes.

``models/*/config.pbtxt`` is tracked and often hand-tuned (instance counts,
batching), so a regenerated one never replaces an existing file that differs
unless the caller says so; it is written next to it as ``config.pbtxt.generated``
(git-ignored) for a diff instead.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)

GENERATED_SUFFIX = '.generated'


def sync_output_dtypes(config_path: Path, dtypes: dict[str, str]) -> bool:
    """Set the ``data_type`` of the named output tensors in an existing config.

    TensorRT decides the output precision of the built engine (FP16 boxes and
    scores under 11.0), and Triton refuses a model whose config disagrees, so
    this is the one edit a kept config cannot do without. Returns whether the
    file changed."""
    text = original = config_path.read_text()
    for name, dtype in dtypes.items():
        text = re.sub(
            rf'(name:\s*"{re.escape(name)}"\s*\n\s*data_type:\s*)TYPE_\w+',
            rf'\g<1>{dtype}',
            text,
        )
    if text == original:
        return False
    config_path.write_text(text)
    logger.info(f'Synced engine output dtypes into {config_path}: {dtypes}')
    return True


def write_generated_config(
    config_path: Path,
    content: str,
    *,
    overwrite: bool = False,
    output_dtypes: dict[str, str] | None = None,
) -> Path:
    """Write ``content`` for ``config_path`` and return the file written.

    A missing file is created; a byte-identical one is left alone; a differing
    one is replaced only with ``overwrite``, otherwise the content goes to
    ``<name>.generated`` beside it. A kept config still takes ``output_dtypes``
    (see ``sync_output_dtypes``)."""
    if config_path.exists():
        if config_path.read_text() == content:
            logger.info(f'Triton config unchanged, not rewriting: {config_path}')
            return config_path
        if not overwrite:
            side = config_path.with_name(config_path.name + GENERATED_SUFFIX)
            side.write_text(content)
            if output_dtypes:
                sync_output_dtypes(config_path, output_dtypes)
            logger.warning(
                f'{config_path} differs from the generated config; wrote {side} instead '
                '(pass --overwrite-config to replace the tracked file)'
            )
            return side
    config_path.write_text(content)
    logger.info(f'Generated Triton config: {config_path}')
    return config_path
