"""Fixtures providers by name; ``families.toml`` lists the ones a family uses."""

from __future__ import annotations

from .base import Fixtures, Provider, ProviderError, present_in_both
from .projects import ProjectsProvider


PROVIDERS: dict[str, Provider] = {provider.name: provider for provider in (ProjectsProvider(),)}

__all__ = ['PROVIDERS', 'Fixtures', 'Provider', 'ProviderError', 'present_in_both']
