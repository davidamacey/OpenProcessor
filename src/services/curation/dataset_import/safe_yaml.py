"""Parse an untrusted ``data.yaml`` without letting it expand.

``yaml.safe_load`` shares an aliased node by reference, so a few hundred
bytes of nested aliases ("billion laughs") are a tiny parse and an
astronomically large structure the moment anything walks it. This loader
refuses every alias outright (a dataset description has no use for one),
and bounds the document's node count and nesting depth, so what comes out
is never larger than the file that went in.
"""

from __future__ import annotations

from typing import Any

import yaml

from src.services.curation.dataset_import.limits import MAX_YAML_DEPTH, MAX_YAML_NODES


class YamlTooComplexError(ValueError):
    """The document uses an alias, or is larger or deeper than the caps."""


class _BoundedLoader(yaml.SafeLoader):
    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._nodes = 0
        self._depth = 0

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.events.AliasEvent):
            raise YamlTooComplexError('aliases are not allowed')
        self._nodes += 1
        if self._nodes > MAX_YAML_NODES:
            raise YamlTooComplexError(f'more than {MAX_YAML_NODES} nodes')
        self._depth += 1
        try:
            if self._depth > MAX_YAML_DEPTH:
                raise YamlTooComplexError(f'nested deeper than {MAX_YAML_DEPTH}')
            return super().compose_node(parent, index)
        finally:
            self._depth -= 1


def load_bounded_yaml(text: str) -> Any:
    """``yaml.safe_load`` of ``text``; raises :class:`yaml.YAMLError` for a
    malformed document and :class:`YamlTooComplexError` for an alias or a
    document over the node or depth cap."""
    loader = _BoundedLoader(text)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


__all__ = ['YamlTooComplexError', 'load_bounded_yaml']
