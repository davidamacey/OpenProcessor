"""``class_source`` values the curation pipeline writes and filters on.

Ingest stamps an item with ``f'{profile.name}_model'`` (confident) or
``f'{profile.name}_low_conf'`` from its item-detector ``DetectionProfile``
(``src/routers/curation/ingest.py`` builds it with ``name='item'``). Every
query that asks "did the classifier label this?" filters on these values,
so they live here instead of as scattered literals.
"""

from __future__ import annotations


INGEST_ITEM_PROFILE_NAME = 'item'

CLASSIFIER_CLASS_SOURCE = f'{INGEST_ITEM_PROFILE_NAME}_model'
CLASSIFIER_LOW_CONF_CLASS_SOURCE = f'{INGEST_ITEM_PROFILE_NAME}_low_conf'
CLASSIFIER_CLASS_SOURCE_PREFIX = f'{INGEST_ITEM_PROFILE_NAME}_'

# Written by the (opt-in) auto-promote stage.
CLUSTER_MAJORITY_CLASS_SOURCE = 'cluster_majority_agreement'

VLM_CLASS_SOURCE = 'vlm'
VLM_UNMATCHED_CLASS_SOURCE = 'vlm_unmatched'
VLM_NEW_CLASS_PENDING_CLASS_SOURCE = 'vlm_new_class_pending'
CLASSIFIER_VLM_AGREEMENT_CLASS_SOURCE = 'classifier_vlm_agreement'


__all__ = [
    'CLASSIFIER_CLASS_SOURCE',
    'CLASSIFIER_CLASS_SOURCE_PREFIX',
    'CLASSIFIER_LOW_CONF_CLASS_SOURCE',
    'CLASSIFIER_VLM_AGREEMENT_CLASS_SOURCE',
    'CLUSTER_MAJORITY_CLASS_SOURCE',
    'INGEST_ITEM_PROFILE_NAME',
    'VLM_CLASS_SOURCE',
    'VLM_NEW_CLASS_PENDING_CLASS_SOURCE',
    'VLM_UNMATCHED_CLASS_SOURCE',
]
