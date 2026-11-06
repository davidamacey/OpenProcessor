"""The analysis settings of the ``items`` index: the compact text normalizer.

``op_text_compact`` strips every character that is not a letter or a digit, folds accents and
upper-cases, so a lower-case or spaced query matches a stored registration plate. Analysis
settings are fixed per index: a new index gets them in its creation body, an upgraded one through
``src/services/upgrade/from_060/analysis_settings.py`` (close, settings, open).

OpenSearch normalizers: https://docs.opensearch.org/latest/analyzers/normalizers/
OpenSearch pattern_replace char filter:
https://docs.opensearch.org/latest/analyzers/character-filters/pattern-replace-character-filter/
"""

from __future__ import annotations

from typing import Any


NORMALIZER_NAME = 'op_text_compact'
CHAR_FILTER_NAME = 'op_alnum_only'

ITEMS_ANALYSIS: dict[str, Any] = {
    'char_filter': {
        CHAR_FILTER_NAME: {
            'type': 'pattern_replace',
            'pattern': '[^\\p{L}\\p{Nd}]',
            'replacement': '',
        }
    },
    'normalizer': {
        NORMALIZER_NAME: {
            'type': 'custom',
            'char_filter': [CHAR_FILTER_NAME],
            'filter': ['asciifolding', 'uppercase'],
        }
    },
}
