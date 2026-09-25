"""Seed/validate class_registry.json from a detector's embedded class names."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import onnx
import pytest
from onnx import TensorProto, helper

from src.clients.curation_opensearch import ClassRegistry, ClassRegistryFile, RegistryClassEntry


if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


@pytest.fixture(scope='module')
def mod() -> ModuleType:
    from scripts.curation import seed_class_registry

    return seed_class_registry


def _onnx_with_names(path: Path, names: str | None) -> Path:
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node('Identity', ['x'], ['y'])], 'g', [x], [y])
    model = helper.make_model(graph)
    if names is not None:
        entry = model.metadata_props.add()
        entry.key = 'names'
        entry.value = names
    onnx.save(model, str(path))
    return path


def _registry(path: Path, entries: list[tuple[int, str, bool]]) -> Path:
    reg = ClassRegistryFile(
        classes=[RegistryClassEntry(class_id=i, class_name=n, deprecated=d) for i, n, d in entries]
    )
    path.write_text(reg.model_dump_json(indent=2), encoding='utf-8')
    return path


def _ids_names(path: Path) -> list[tuple[int, str]]:
    reg = ClassRegistry(path).load()
    return [(c.class_id, c.class_name) for c in reg.classes]


class TestReadNames:
    def test_reads_ultralytics_repr_metadata(self, mod: ModuleType, tmp_path: Path) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', "{0: 'cat', 1: 'dog', 2: 'bird'}")
        assert mod.read_class_names(model) == ['cat', 'dog', 'bird']

    def test_orders_by_id_not_insertion(self, mod: ModuleType, tmp_path: Path) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', "{1: 'dog', 0: 'cat'}")
        assert mod.read_class_names(model) == ['cat', 'dog']

    def test_missing_names_metadata_is_an_error(self, mod: ModuleType, tmp_path: Path) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', None)
        with pytest.raises(mod.ClassNamesError, match='names'):
            mod.read_class_names(model)

    def test_non_contiguous_ids_rejected(self, mod: ModuleType, tmp_path: Path) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', "{0: 'cat', 2: 'bird'}")
        with pytest.raises(mod.ClassNamesError, match='contiguous'):
            mod.read_class_names(model)

    def test_reads_data_yaml_dict_and_list(self, mod: ModuleType, tmp_path: Path) -> None:
        d = tmp_path / 'data.yaml'
        d.write_text('nc: 1\nnames:\n  0: widget\n')
        assert mod.read_class_names(d) == ['widget']
        d.write_text('names: [a, b]\n')
        assert mod.read_class_names(d) == ['a', 'b']

    def test_data_yaml_nc_mismatch_rejected(self, mod: ModuleType, tmp_path: Path) -> None:
        d = tmp_path / 'data.yaml'
        d.write_text('nc: 2\nnames: [a]\n')
        with pytest.raises(mod.ClassNamesError, match='nc=2'):
            mod.read_class_names(d)


class TestSeedAndExtend:
    def test_seeds_fresh_registry_in_model_order(self, mod: ModuleType, tmp_path: Path) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', "{0: 'cat', 1: 'dog'}")
        reg = tmp_path / 'reg' / 'class_registry.json'
        assert mod.main(['--model', str(model), '--registry', str(reg)]) == 0
        assert _ids_names(reg) == [(0, 'cat'), (1, 'dog')]
        assert mod.main(['--model', str(model), '--registry', str(reg), '--check']) == 0

    def test_appends_new_model_classes_only(self, mod: ModuleType, tmp_path: Path) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', "{0: 'cat', 1: 'dog', 2: 'bird'}")
        reg = _registry(tmp_path / 'r.json', [(0, 'cat', False), (1, 'dog', False)])
        assert mod.main(['--model', str(model), '--registry', str(reg), '--check']) == 1
        assert mod.main(['--model', str(model), '--registry', str(reg)]) == 0
        assert _ids_names(reg) == [(0, 'cat'), (1, 'dog'), (2, 'bird')]
        # The pre-write file was snapshotted next to the canonical one.
        assert list(tmp_path.glob('r.*.json'))

    def test_dry_run_writes_nothing(self, mod: ModuleType, tmp_path: Path) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', "{0: 'cat', 1: 'dog'}")
        reg = _registry(tmp_path / 'r.json', [(0, 'cat', False)])
        before = reg.read_text()
        assert mod.main(['--model', str(model), '--registry', str(reg), '--dry-run']) == 0
        assert reg.read_text() == before

    def test_extra_class_appended_after_model_and_never_reuses_ids(
        self, mod: ModuleType, tmp_path: Path
    ) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', "{0: 'cat', 1: 'dog'}")
        # id 2 is a deprecated post-model class; the new extra must get id 3.
        reg = _registry(
            tmp_path / 'r.json', [(0, 'cat', False), (1, 'dog', False), (2, 'old', True)]
        )
        rc = mod.main(
            ['--model', str(model), '--registry', str(reg), '--extra-class', 'misc:special']
        )
        assert rc == 0
        loaded = ClassRegistry(reg).load()
        assert [(c.class_id, c.class_name, c.group) for c in loaded.classes][-1] == (
            3,
            'misc',
            'special',
        )
        # Idempotent: re-running with the same extra adds nothing.
        n = len(loaded.classes)
        assert (
            mod.main(
                ['--model', str(model), '--registry', str(reg), '--extra-class', 'misc:special']
            )
            == 0
        )
        assert len(ClassRegistry(reg).load().classes) == n

    def test_group_map_applies_to_appended(self, mod: ModuleType, tmp_path: Path) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', "{0: 'cat', 1: 'dog'}")
        groups = tmp_path / 'groups.json'
        groups.write_text(json.dumps({'pets': ['cat', 'dog']}))
        reg = tmp_path / 'r.json'
        assert (
            mod.main(['--model', str(model), '--registry', str(reg), '--group-map', str(groups)])
            == 0
        )
        assert {c.group for c in ClassRegistry(reg).load().classes} == {'pets'}


class TestDrift:
    def test_name_mismatch_fails_check_and_refuses_write(
        self, mod: ModuleType, tmp_path: Path
    ) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', "{0: 'dog', 1: 'cat'}")
        reg = _registry(tmp_path / 'r.json', [(0, 'cat', False), (1, 'dog', False)])
        before = reg.read_text()
        assert mod.main(['--model', str(model), '--registry', str(reg), '--check']) == 1
        assert mod.main(['--model', str(model), '--registry', str(reg)]) == 1
        assert reg.read_text() == before

    def test_model_emitting_deprecated_id_is_drift(self, mod: ModuleType, tmp_path: Path) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', "{0: 'cat', 1: 'dog'}")
        reg = _registry(tmp_path / 'r.json', [(0, 'cat', False), (1, 'dog', True)])
        assert mod.main(['--model', str(model), '--registry', str(reg), '--check']) == 1

    def test_registry_only_classes_ok_unless_strict(self, mod: ModuleType, tmp_path: Path) -> None:
        model = _onnx_with_names(tmp_path / 'm.onnx', "{0: 'cat', 1: 'dog'}")
        reg = _registry(
            tmp_path / 'r.json', [(0, 'cat', False), (1, 'dog', False), (2, 'extra', False)]
        )
        assert mod.main(['--model', str(model), '--registry', str(reg), '--check']) == 0
        assert mod.main(['--model', str(model), '--registry', str(reg), '--check', '--strict']) == 1

    def test_model_growth_blocked_by_registry_only_ids(
        self, mod: ModuleType, tmp_path: Path
    ) -> None:
        """Registry added a post-model class at id 2; a retrained model that now
        also has 3 classes must not be silently merged at mismatching ids."""
        model = _onnx_with_names(tmp_path / 'm.onnx', "{0: 'cat', 1: 'dog', 2: 'bird'}")
        reg = _registry(
            tmp_path / 'r.json', [(0, 'cat', False), (1, 'dog', False), (2, 'extra', False)]
        )
        before = reg.read_text()
        assert mod.main(['--model', str(model), '--registry', str(reg)]) == 1
        assert reg.read_text() == before

    def test_reconcile_reports_append_blocked(self, mod: ModuleType) -> None:
        existing = ClassRegistryFile(
            classes=[
                RegistryClassEntry(class_id=0, class_name='cat'),
                RegistryClassEntry(class_id=1, class_name='extra'),
                RegistryClassEntry(class_id=2, class_name='more'),
            ]
        )
        # Model has 'cat' plus 3 more; id 1 mismatches, id 3 is next free so
        # it would append — but the fatal mismatch blocks any write.
        result = mod.reconcile(['cat', 'dog', 'more', 'bird'], existing)
        kinds = {f.kind for f in result.findings}
        assert 'name_mismatch' in kinds
        assert result.registry is None

    def test_duplicate_model_names_rejected(self, mod: ModuleType) -> None:
        result = mod.reconcile(['cat', 'cat'], None)
        assert result.has_fatal
        assert result.registry is None
