"""The single-key set the default keymap reserves from class hotkeys.

Not a test module itself.
"""

from __future__ import annotations


# The labeling keys that predate the configurable keymap.
LEGACY_RESERVED_HOTKEY_LETTERS = frozenset('gndzxuam/feb')

# Per-box accept and reject, the overlay's backtick, the Workspace's view, scope and details
# keys, and the image editor's tool, zoom and image-walking keys.
DEFAULT_RESERVED_HOTKEYS = LEGACY_RESERVED_HOTKEY_LETTERS | frozenset(
    {'y', 'r', '`', '[', ']', ',', '.', '\\', 'v', 'h', 'l', '=', '-', '0', '1'}
)
