"""Which live modules may share the stack with others (see tests/live/README.md for the table).

A module in ``ISOLATED`` creates its own project, reads only that project's data (or a record it
names by a unique id) and never acts on a shared service; it may run beside other isolated
modules in separate xdist workers. Every other module is ``live_serial`` (the marker is added in
``conftest.pytest_collection_modifyitems``, so a module nobody classified is serial too).
``tests/test_live_classification.py`` fails when a module is in no set, or when an isolated one
uses a stack-acting helper.
"""

from __future__ import annotations


ISOLATED = frozenset(
    {
        'test_live_combine',
        'test_live_core_ingest',
        'test_live_crop_space',
        'test_live_detector_class',
        'test_live_ingest',
        'test_live_preset_documents',
        'test_live_region_drain_narrowed',
        'test_live_search',
        'test_queue_delivery',
    }
)

# Stops, restarts or kills a container, changes replicas or worker counts, makes the fake engine
# refuse, drives the fake VLM's mode, or asserts on stack-wide state (all exports, all containers).
STACK_ACTING = frozenset(
    {
        'test_control_plane_smoke',
        'test_job_load',
        'test_job_recovery_real_handlers',
        'test_live_assist_loop',
        'test_live_bulk_restart',
        'test_live_engine_unavailable',
        'test_live_models_promote',
        'test_live_train',
        'test_live_train_promote',
        'test_live_vlm',
        'test_one_vs_n',
        'test_sse_across_replicas',
    }
)

# Works on the seeded `default` project (shared cohorts, class registry, settings) or starts a
# container of its own; nothing here proves it safe beside other tests, so it stays serial.
UNSURE = frozenset(
    {
        'test_class_registry_live',
        'test_live_annotation',
        'test_live_bulk_occ',
        'test_live_capture_map',
        'test_live_classes',
        'test_live_clusters',
        'test_live_embedding_state',
        'test_live_export',
        'test_live_generic_detector',
        'test_live_image_counts',
        'test_live_images',
        'test_live_labels',
        'test_live_ocr',
        'test_live_open_vocab',
        'test_live_paging',
        'test_live_region_text',
        'test_live_regions',
        'test_live_seed_from_detector',
        'test_live_served_facts',
        'test_live_whole_photo',
    }
)

# Names whose use makes a module stack-acting: they stop, restart or kill a service, switch the
# fake engine's mode, change replica or worker counts, or switch the fake VLM's mode.
STACK_ACTING_NAMES = frozenset(
    {
        'engine_refuses',
        'engine_normal',
        'reconfigure',
        'restore_base_stack',
        'require_replicas',
        'kill',
        'stop',
        'start',
        'up',
        'remove_with_volumes',
        'docker_ps_snapshot',
        'fake_vlm',
        'start_burst',
    }
)
# `compose(...)` first arguments that change a container rather than read it.
STACK_ACTING_COMPOSE_VERBS = frozenset(
    {'kill', 'stop', 'start', 'restart', 'up', 'rm', 'run', 'down', 'pause', 'unpause', 'scale'}
)
