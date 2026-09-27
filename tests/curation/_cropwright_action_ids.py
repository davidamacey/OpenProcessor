"""Vendored action-id list from Cropwright's ``FALLBACK_KEYMAP``
(the frontend repo's ``src/lib/keymapFallback.ts``, commit
88692d72204587daf38d91696bfa96196ee191bf) -- every keyboard action the
real frontend registers or displays today. ``test_keymap.py`` asserts
the backend's action registry is a superset of this list, so the wire
contract never drops an id the client actually uses.

Not a test module itself.
"""

from __future__ import annotations


CROPWRIGHT_ACTION_IDS: tuple[str, ...] = (
    'global.shortcuts_overlay',
    'global.close_overlay',
    'review.skip',
    'review.undo',
    'review.queue.confirm',
    'review.queue.discard',
    'review.queue.class_picker',
    'review.queue.prev',
    'review.queue.next',
    'review.region.confirm',
    'review.region.reject',
    'review.region.false_positive',
    'review.region.edit_box',
    'review.region.back',
    'review.region.next',
    'review.region.accept_box',
    'review.region.reject_box',
    'box_edit.save',
    'box_edit.cancel',
    'box_edit.nudge_up',
    'box_edit.nudge_down',
    'box_edit.nudge_left',
    'box_edit.nudge_right',
    'box_edit.shrink_right',
    'box_edit.grow_right',
    'box_edit.delete_box',
    'box_edit.next_box',
    'cluster.confirm',
    'cluster.accept_all_vlm',
    'cluster.accept_vlm',
    'cluster.skip',
    'cluster.flag_new_class',
    'cluster.discard',
    'cluster.undo',
    'cluster.ignore',
    'cluster.unignore',
    'cluster.select_all',
    'cluster.prev',
    'cluster.next',
    'cluster.move',
    'cluster.cancel',
    'clusters_search.select_all',
    'clusters_search.cancel',
    'clusters_search.ignore',
    'clusters_search.undo',
    'region_gallery.undo',
)
