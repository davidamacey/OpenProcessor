"""An in-memory stand-in for the class registry, for offline tests that hand a registry to code.

The real registry is PostgreSQL (``src/db/repos/classes.py``, proven by ``tests/db/test_class_registry.py``);
the offline tier has no database, so the autouse fixture in ``tests/conftest.py`` makes every
project's ``get_class_registry()`` answer one of these. It keeps the name rules of the real thing
(``name_key``: one class per normalized name among the active ones, an active class wins over a
deprecated one, ids never reused) so the code under test sees the same answers.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from src.clients.curation_opensearch import RegistryClassEntry
from src.db.repos.classes import (
    ClassNameInvalidError,
    ClassNameTakenError,
    ClassNotFoundError,
    ClassRegistryError,
    HotkeyTakenError,
)
from src.services.classes.registry import RegistryDoc
from src.utils.class_names import normalize_class_name, resolve_class_by_name


def _run_sync(coro: Any) -> Any:
    """Run ``coro`` to completion from sync code, also when an event loop is already running."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class InMemoryRegistry:
    def __init__(
        self, base_dir: Any = None, names: list[str] | None = None, **entry_kw: Any
    ) -> None:
        del base_dir  # kept so call sites need not change; the registry has no file
        self._doc = RegistryDoc()
        self.projection_tasks: list[tuple[str, dict[str, Any]]] = []
        if names:
            self.seed(names, **entry_kw)

    # ---- sync setup / inspection

    def _next_id(self) -> int:
        d = self._doc
        return max(d.next_class_id, max((c.class_id for c in d.classes), default=-1) + 1)

    def seed(self, names: list[str], **entry_kw: Any) -> None:
        """Append classes straight into the registry, ids from the id counter."""
        for name in names:
            new_id = self._next_id()
            self._doc.classes.append(
                RegistryClassEntry(class_id=new_id, class_name=name, **entry_kw)
            )
            self._doc.next_class_id = new_id + 1
        self._doc.registry_revision += 1

    def seed_rows(self, rows: list[dict[str, Any]]) -> None:
        """Replace the classes with ``rows`` (dicts with ``class_id`` and ``class_name``)."""
        self._doc = RegistryDoc(classes=[RegistryClassEntry(**row) for row in rows])
        self._doc.next_class_id = max((c.class_id for c in self._doc.classes), default=-1) + 1

    def store(self, doc: Any) -> None:
        """Replace the registry with ``doc`` (states the API cannot reach)."""
        self._doc = RegistryDoc.model_validate(doc.model_dump())

    def load(self) -> Any:
        return self._doc.model_copy(deep=True)

    @property
    def classes(self) -> Any:
        """Live view, so a service handed this object as its registry snapshot sees later edits."""
        return self._doc.classes

    def get(self, class_id: int) -> Any:
        return self._doc.get(class_id)

    def validate_id(self, class_id: int) -> bool:
        return self._doc.validate_id(class_id)

    # ---- sync facades (test setup)

    def add(self, name: str, **kw: Any) -> int:
        return _run_sync(self.add_class(name, **kw))

    def deprecate(self, class_id: int, deprecated: bool = True) -> Any:
        return _run_sync(self.set_deprecated(class_id, deprecated))

    def merge(self, source_id: int, target_id: int) -> Any:
        return _run_sync(self.merge_class(source_id, target_id))

    def rename_class(self, class_id: int, new_name: str) -> Any:
        return _run_sync(self.update_class(class_id, name=new_name))

    # ---- the registry API

    async def snapshot(self, **_kw: Any) -> Any:
        await asyncio.sleep(0)
        return self._doc.model_copy(deep=True)

    def _entry(self, class_id: int) -> RegistryClassEntry:
        found = self._doc.get(class_id)
        if found is None:
            raise ClassNotFoundError(f'class_id {class_id} not found')
        return found

    def _touch(self) -> None:
        self._doc.registry_revision += 1

    async def copy_from(self, source_slug: str) -> int:
        from src.services.classes.registry import registry_for

        self._doc = registry_for(source_slug).load().model_copy(deep=True)
        return len(self._doc.classes)

    async def add_class(
        self,
        name: str,
        group: str = 'unknown',
        notes: str = '',
        hotkey_letter: str | None = None,
    ) -> int:
        entry, _ = self._add(name, group, notes, hotkey_letter, ensure=False)
        return entry.class_id

    async def ensure_class(self, name: str, group: str = 'unknown', notes: str = '') -> Any:
        entry, created = self._add(name, group, notes, None, ensure=True)
        return entry.class_id, created

    async def ensure_entry(self, name: str, group: str = 'unknown', notes: str = '') -> Any:
        return self._add(name, group, notes, None, ensure=True)

    def _add(
        self, name: str, group: str, notes: str, hotkey: str | None, *, ensure: bool
    ) -> tuple[RegistryClassEntry, bool]:
        stripped = name.strip()
        if not normalize_class_name(stripped):
            raise ClassNameInvalidError('class_name must contain a letter or digit')
        taken = resolve_class_by_name(self._doc.classes, stripped)
        if taken.active is not None:
            if ensure:
                return taken.active.model_copy(), False
            raise ClassNameTakenError(
                f'duplicate class_name {stripped!r} (id={taken.active.class_id})'
            )
        if hotkey is not None and any(c.hotkey_letter == hotkey for c in self._doc.classes):
            raise HotkeyTakenError('hotkey_letter is already used by another class')
        new_id = self._next_id()
        self._doc.next_class_id = new_id + 1
        entry = RegistryClassEntry(
            class_id=new_id, class_name=stripped, group=group, notes=notes, hotkey_letter=hotkey
        )
        self._doc.classes.append(entry)
        self._touch()
        return entry.model_copy(), True

    async def update_class(
        self,
        class_id: int,
        *,
        name: str | None = None,
        group: str | None = None,
        hotkey_letter: str | None = None,
        set_hotkey: bool = False,
    ) -> Any:
        target = self._entry(class_id)
        if name is not None:
            stripped = name.strip()
            if not stripped:
                raise ClassRegistryError('new_name must be non-empty')
            if stripped != target.class_name:
                taken = resolve_class_by_name(
                    (c for c in self._doc.classes if c.class_id != class_id), stripped
                )
                if taken.active is not None:
                    raise ClassNameTakenError(
                        f'rename target {stripped!r} already in use (id={taken.active.class_id})'
                    )
                target.class_name = stripped
                self.projection_tasks.append(
                    ('class_rename', {'class_id': class_id, 'new_name': stripped})
                )
        if group is not None:
            target.group = group
        if set_hotkey:
            target.hotkey_letter = hotkey_letter
        self._touch()
        return target.model_copy()

    async def merge_class(self, source_id: int, target_id: int) -> dict[str, Any]:
        if source_id == target_id:
            raise ClassRegistryError('cannot merge a class into itself')
        source, target = self._doc.get(source_id), self._doc.get(target_id)
        if source is None:
            raise ClassNotFoundError(f'source class_id {source_id} not found')
        if target is None:
            raise ClassNotFoundError(f'target class_id {target_id} not found')
        if target.deprecated:
            raise ClassRegistryError(
                f'target class_id {target_id} is deprecated; cannot merge into it'
            )
        source.deprecated = True
        source.merged_into = target_id
        source.hotkey_letter = None
        self._touch()
        self.projection_tasks.append(
            ('class_merge', {'source_ids': [source_id], 'target_id': target_id})
        )
        return {
            'source_id': source_id,
            'target_id': target_id,
            'deprecated': True,
            'source_name': source.class_name,
            'target_name': target.class_name,
        }

    async def set_deprecated(self, class_id: int, deprecated: bool) -> Any:
        target = self._entry(class_id)
        if not deprecated and target.deprecated:
            taken = resolve_class_by_name(
                (c for c in self._doc.classes if c.class_id != class_id), target.class_name
            )
            if taken.active is not None:
                raise ClassNameTakenError(
                    f'cannot restore class_id {class_id}: name {target.class_name!r} '
                    f'already in use by class_id {taken.active.class_id}'
                )
        target.deprecated = deprecated
        if deprecated:
            target.hotkey_letter = None
        else:
            target.merged_into = None
        self._touch()
        return target.model_copy()

    async def set_hotkeys(
        self, letters: dict[int, str | None], *, only_if_unset: bool = False
    ) -> dict[int, str | None]:
        applied: dict[int, str | None] = {}
        for c in self._doc.classes:
            if c.class_id not in letters:
                continue
            if only_if_unset and c.hotkey_letter is not None:
                continue
            if c.hotkey_letter != letters[c.class_id]:
                c.hotkey_letter = letters[c.class_id]
                applied[c.class_id] = c.hotkey_letter
        self._touch()
        return applied


def make_registry(base_dir: Any, names: list[str], **entry_kw: Any) -> InMemoryRegistry:
    return InMemoryRegistry(base_dir, names, **entry_kw)


class InMemoryRegistries(dict):  # type: ignore[type-arg]
    """Stands in for ``registry._registries``: every project gets its own empty in-memory
    registry the first time the code under test asks for it."""

    override: Any = None

    def get(self, key: Any, default: Any = None) -> Any:  # noqa: ARG002
        if self.override is not None:
            return self.override
        if key not in self:
            self[key] = InMemoryRegistry()
        return self[key]


def use_registry(registry: InMemoryRegistry) -> InMemoryRegistry:
    """Make ``registry`` the one every project's ``get_class_registry()`` answers in this test."""
    from src.services.classes import registry as registry_module

    registry_module._registries.override = registry
    return registry


def registry_of(record: Any) -> InMemoryRegistry:
    """The in-memory registry the autouse fixture gave ``record``'s project."""
    from src.services.classes import registry as registry_module

    reg = registry_module._registries.get(record.slug)
    assert isinstance(reg, InMemoryRegistry)
    return reg
