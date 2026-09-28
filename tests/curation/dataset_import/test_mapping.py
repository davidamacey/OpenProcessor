"""W10.5/W10.17: class-name mapping — never by index."""

from __future__ import annotations

from src.services.curation.dataset_import.mapping import (
    ClassMappingEntry,
    RegistryClassView,
    norm_class_name,
    resolve_mapping,
    suggest_mapping,
)


REGISTRY = [
    RegistryClassView(class_id=0, class_name='car'),
    RegistryClassView(class_id=1, class_name='truck'),
    RegistryClassView(class_id=2, class_name='bus'),
    RegistryClassView(class_id=3, class_name='sedan', deprecated=True, merged_into=0),
]


class TestSuggestionLadder:
    def test_exact(self) -> None:
        assert suggest_mapping('car', registry_classes=REGISTRY) == suggest_mapping(
            'car', registry_classes=REGISTRY
        )
        s = suggest_mapping('car', registry_classes=REGISTRY)
        assert s.match == 'exact'
        assert s.action == 'map'
        assert s.class_id == 0

    def test_case_insensitive(self) -> None:
        s = suggest_mapping('Car', registry_classes=REGISTRY)
        assert s.match == 'case_insensitive'
        assert s.class_id == 0

    def test_merged(self) -> None:
        s = suggest_mapping('sedan', registry_classes=REGISTRY)
        assert s.match == 'merged'
        assert s.class_id == 0

    def test_synonym(self) -> None:
        s = suggest_mapping('automobile', registry_classes=REGISTRY, synonyms={'automobile': 'car'})
        assert s.match == 'synonym'
        assert s.class_id == 0

    def test_region(self) -> None:
        s = suggest_mapping('wheel', registry_classes=REGISTRY, region_class_name='wheel')
        assert s.match == 'region'
        assert s.action == 'region'

    def test_single_class_region_export(self) -> None:
        s = suggest_mapping('plate', registry_classes=REGISTRY, is_single_class_region_export=True)
        assert s.action == 'region'

    def test_none(self) -> None:
        s = suggest_mapping('automobile', registry_classes=REGISTRY)
        assert s.match == 'none'
        assert s.action == 'create'

    def test_same_registry_op_export(self) -> None:
        s = suggest_mapping('car', registry_classes=REGISTRY, source_class_id=0)
        assert s.match == 'same_registry'
        assert s.class_id == 0


class TestResolveMapping:
    def test_never_by_index(self) -> None:
        """Dataset ``{0: truck, 1: car}`` against a registry
        ``[car(0), truck(1)]`` imports trucks as truck, not car — the
        bug the pre-W10 index-based importer had."""
        entries = [
            ClassMappingEntry(dataset_class='truck', action='map', class_id=1),
            ClassMappingEntry(dataset_class='car', action='map', class_id=0),
        ]
        resolved = resolve_mapping(['truck', 'car'], entries, registry_classes=REGISTRY)
        assert resolved.ok
        assert resolved.targets['truck'].class_name == 'truck'
        assert resolved.targets['car'].class_name == 'car'

    def test_merge_synonyms_into_one_target(self) -> None:
        entries = [
            ClassMappingEntry(dataset_class='Car', action='map', class_id=0),
            ClassMappingEntry(dataset_class='automobile', action='map', class_id=0),
        ]
        resolved = resolve_mapping(['Car', 'automobile'], entries, registry_classes=REGISTRY)
        assert resolved.ok
        assert sorted(resolved.merged_from[0]) == ['Car', 'automobile']

    def test_create_name_collision_blocking(self) -> None:
        entries = [ClassMappingEntry(dataset_class='car2', action='create', new_class_name='car')]
        resolved = resolve_mapping(['car2'], entries, registry_classes=REGISTRY)
        assert not resolved.ok
        assert resolved.errors[0].code == 'class_name_exists'

    def test_two_creates_same_normalized_name_conflict(self) -> None:
        entries = [
            ClassMappingEntry(dataset_class='foo', action='create', new_class_name='widget'),
            ClassMappingEntry(dataset_class='bar', action='create', new_class_name='Widget'),
        ]
        resolved = resolve_mapping(['foo', 'bar'], entries, registry_classes=REGISTRY)
        assert not resolved.ok
        assert resolved.errors[0].code == 'class_mapping_conflict'

    def test_unmapped_class_incomplete(self) -> None:
        resolved = resolve_mapping(['car'], [], registry_classes=REGISTRY)
        assert not resolved.ok
        assert resolved.errors[0].code == 'class_mapping_incomplete'

    def test_map_to_deprecated_blocking(self) -> None:
        entries = [ClassMappingEntry(dataset_class='car', action='map', class_id=3)]
        resolved = resolve_mapping(['car'], entries, registry_classes=REGISTRY)
        assert not resolved.ok
        assert resolved.errors[0].code == 'class_mapped_to_deprecated'

    def test_accept_suggestions_fills_exact_case_insensitive_region_same_registry(self) -> None:
        suggestions = {
            'car': suggest_mapping('car', registry_classes=REGISTRY),
            'Truck': suggest_mapping('Truck', registry_classes=REGISTRY),
        }
        resolved = resolve_mapping(
            ['car', 'Truck'],
            [],
            registry_classes=REGISTRY,
            accept_suggestions=True,
            suggestions=suggestions,
        )
        assert resolved.ok
        assert resolved.targets['car'].class_id == 0
        assert resolved.targets['Truck'].class_id == 1

    def test_accept_suggestions_never_takes_synonym_merged_or_create(self) -> None:
        suggestions = {
            'sedan': suggest_mapping('sedan', registry_classes=REGISTRY),
            'automobile': suggest_mapping(
                'automobile', registry_classes=REGISTRY, synonyms={'automobile': 'car'}
            ),
            'newthing': suggest_mapping('newthing', registry_classes=REGISTRY),
        }
        resolved = resolve_mapping(
            ['sedan', 'automobile', 'newthing'],
            [],
            registry_classes=REGISTRY,
            accept_suggestions=True,
            suggestions=suggestions,
        )
        assert not resolved.ok
        assert {e.dataset_class for e in resolved.errors} == {'sedan', 'automobile', 'newthing'}


def test_norm_class_name() -> None:
    assert norm_class_name('  Car_Van-Type  ') == 'car van type'
