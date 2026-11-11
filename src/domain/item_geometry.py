"""What an item's stored box means: who set it and what a person may change.

Pure rules shared by the lock predicate (:mod:`src.clients.occ_locks`), the drawn-item
create and the geometry edit. ``bbox_norm`` is the box in image coordinates
(``x1, y1, x2, y2``, each in ``[0, 1]``); only the item geometry writers change it for a
person, and an automated writer never changes a box a person set.
"""

from __future__ import annotations


GEOMETRY_SOURCE_HUMAN = 'human'

NOT_AN_OBJECT = 'not_an_object'
"""The exclude reason of a person's soft delete: the box was never an object. The item stays a
document so its box keeps blocking a re-detect from creating it again."""
