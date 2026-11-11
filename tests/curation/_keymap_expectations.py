"""The single-key set the default keymap reserves from class hotkeys.

Not a test module itself.
"""

from __future__ import annotations


# The labeling keys that predate the configurable keymap.
LEGACY_RESERVED_HOTKEY_LETTERS = frozenset('gndzxuam/feb')

# Per-box accept and reject, the overlay's backtick, and the Workspace's
# view, scope and details keys.
DEFAULT_RESERVED_HOTKEYS = LEGACY_RESERVED_HOTKEY_LETTERS | frozenset(
    {'y', 'r', '`', '[', ']', ',', '.', '\\'}
)
