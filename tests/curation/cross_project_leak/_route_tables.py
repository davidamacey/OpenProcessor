"""Per-route exemptions and policy tables the isolation sweep checks against."""

from __future__ import annotations

from curation.cross_project_leak._constants import SCOPED


# Mutating routes that write nothing by design (read-only work under POST,
# or an event with no data write), and why. Every other mutating route must
# really write in the sweep.
NO_WRITE: dict[tuple[str, str], str] = {
    ('POST', '/datasets/preview'): 'scans a dataset read-only; writes nothing',
    ('POST', '/datasets/imports'): (
        'the sweep sends no mapping, so it is refused before anything is claimed; the '
        'import writes are covered in tests/integration/test_dataset_import_routes.py'
    ),
    ('POST', '/datasets/imports/{import_id}/cancel'): 'the sweep has no live import to cancel',
    ('POST', '/datasets/imports/{import_id}/resume'): 'the sweep has no interrupted import',
    ('POST', '/datasets/imports/{import_id}/undo'): 'a dry run reads the ledger; writes nothing',
    ('POST', '/ingest/bulk'): (
        'the sweep names no folder, so it is refused before a run is claimed; the run '
        'itself is covered in tests/integration/test_bulk_ingest_routes.py'
    ),
    ('POST', '/ingest/bulk/preview'): 'read-only walk of a source; writes nothing',
    ('POST', '/ingest/bulk/{run_id}/cancel'): 'the sweep has no live run to cancel',
    ('POST', '/ingest/sources'): 'the sweep names no existing folder, so the source is refused',
    ('PATCH', '/ingest/sources/{source_id}'): 'the sweep has no such source (404)',
    ('DELETE', '/ingest/sources/{source_id}'): 'the sweep has no such source (404)',
    ('POST', '/ingest/sources/{source_id}/run'): 'the sweep has no such source (404)',
    ('POST', '/ingest/sources/{source_id}/retry-dead'): 'the sweep has no such source (404)',
    ('POST', '/ingest/sources/{source_id}/events'): 'the sweep has no such source (404)',
    (
        'POST',
        '/ingest/objects',
    ): 'the sweep names a bucket outside OP_S3_ALLOWED, so it is refused first (422)',
    (
        'POST',
        '/ingest/objects/presign',
    ): 'the sweep names a bucket outside OP_S3_ALLOWED (422); it writes nothing',
    ('POST', '/events/publish'): 'publishes an event (checked separately), writes no data',
    ('POST', '/ingest/path_lookup'): 'read-only lookup under POST',
    (
        'POST',
        '/ingest/policy/preview',
    ): 'counts what a candidate policy would embed; writes nothing',
    ('POST', '/train/preflight'): 'read-only validation under POST',
    ('POST', '/keymap/validate'): 'dry-run report; writes nothing',
    ('POST', '/reprocess/jobs/{job_id}/cancel'): (
        'touches a cancel flag only for a live job; the sweep has none (the live path is '
        'covered in test_reprocess_entrypoints)'
    ),
    ('POST', '/assist/loops'): 'a dry run plans and counts; the sweep starts no loop',
    ('POST', '/assist/loops/{loop_id}/cancel'): (
        'touches a cancel flag only for a live loop; the sweep has none (the live path is '
        'covered in test_assist_loop_job)'
    ),
    ('POST', '/assist/loops/{loop_id}/resume'): (
        'names a loop of the bound project; the sweep has none (covered in test_assist_loop_job)'
    ),
    ('POST', '/assist/loops/{loop_id}/rollback'): (
        'names a loop of the bound project; the sweep has none (the write path is covered in '
        'test_assist_loop_routes and test_assist_loop_rollback)'
    ),
    ('POST', '/assist/runs'): 'a dry run plans and counts; the sweep starts no run',
    ('POST', '/assist/runs/{run_id}/cancel'): (
        'touches a cancel flag only for a live run; the sweep has none (the live path is '
        'covered in test_assist_run_classify)'
    ),
    ('POST', '/assist/runs/{run_id}/apply'): (
        'names a run of the bound project; the sweep has none (the write path is covered in '
        'test_assist_proposal)'
    ),
    ('POST', '/train/reload_promoted'): 'asks Triton to load promoted models; stores nothing',
    (
        'POST',
        '/vlm/verify_region_batch',
    ): "returns the VLM's verdicts to the caller; stores nothing",
    (
        'POST',
        '/vlm/region_visible_batch',
    ): "returns the VLM's verdicts to the caller; stores nothing",
    # P3 lifecycle mutations write the shared op_projects registry doc,
    # never the project's own item/image/etc indexes or state-dir files
    # -- the write-detection this sweep does (own_indexes / dir digest)
    # has nothing of the bound project's *data* to see, by design.
    (
        'DELETE',
        '',
    ): 'dry_run=true in the sweep; a real delete mutates the registry, not project data',
    ('PATCH', ''): 'mutates the shared project registry doc, not project data',
    ('POST', '/archive'): 'mutates the shared project registry doc, not project data',
    ('POST', '/unarchive'): 'mutates the shared project registry doc, not project data',
    ('POST', '/clone_settings'): 'mutates the shared project registry doc, not project data',
    ('POST', '/prompt_packs/validate'): 'dry-run report; never writes',
    (
        'POST',
        '/prompt_packs/test',
    ): 'runs a pack call over a stored crop and previews; never writes',
    ('POST', '/region_profiles/validate'): 'dry-run report; never writes',
    ('POST', '/region_profiles/validate_segmenter_prompt'): 'dry-run report; never writes',
    ('POST', '/open_vocab/validate'): 'dry-run report; never writes',
    ('POST', '/open_vocab/test'): 'runs a target over a stored image and previews; never writes',
    (
        'POST',
        '/region_profiles/test',
    ): 'runs a profile over a stored crop and previews; never writes',
}

# Routes whose ``{crop_id}`` is a seeded item other than ``-item-0001``,
# because they act on state that item does not have.
CROP_FOR: dict[tuple[str, str], str] = {
    ('POST', '/crops/{crop_id}/vlm_dismiss'): 'item-0004',
    ('POST', '/crops/{crop_id}/vlm_dismiss/undo'): 'item-0004',
    ('POST', '/crops/{crop_id}/region/undo'): 'item-0005',
    # an unlocked item: the default item-0001 is a validated set (locked)
    ('POST', '/crops/{crop_id}/reprocess'): 'item-0002',
}

# Routes that answer 5xx in the fixture for a reason that is not isolation.
EXPECTED_5XX: dict[tuple[str, str], str] = {
    ('POST', '/region_profiles/test'): (
        '502 segmenter_error: the sweep has no network, so the only leg of the draft '
        'profile cannot run (a total leg failure is a 502 by design); the item is read '
        'from the bound project before that'
    ),
    ('POST', '/open_vocab/test'): (
        '502 segmenter_error: the sweep has no segmenter (no network), so the call cannot '
        'run (an outage is never "no hits"); the image is read from the bound project first'
    ),
    ('DELETE', '/models/{model_name}'): (
        '502: the dead in-process Triton never confirms the unload, so the route '
        'refuses to delete the (own, ownership-checked) model dir'
    ),
}

# P3 lifecycle mutations (global_router; textually under SCOPED but never
# part of the scoped/alias double-mount -- see the module docstring at the
# top of src/routers/curation/projects.py). They act *on* a project via a
# plain ``project: str`` path parameter, not *within* one via
# ``bind_path_project``, so their OpenSearch calls (all against the shared
# ``op_projects`` registry doc) are correctly unbound (``try_current_project()
# is None``) rather than bound to the slug in the URL.
# Routes that GET documents from OTHER projects' ``configs`` index by design,
# each read inside a read-only bind of the project it reads: the model-sharing
# route names which projects run a detector profile on the model. Nothing
# else of another project (items, images, labels) may be reached, and no
# write.
CROSS_PROJECT_CONFIG_READS: frozenset[tuple[str, str]] = frozenset(
    {('PUT', '/models/{model_name}/sharing')}
)

UNBOUND_BY_DESIGN: frozenset[tuple[str, str]] = frozenset(
    {
        ('DELETE', ''),
        ('PATCH', ''),
        ('POST', '/archive'),
        ('POST', '/unarchive'),
        ('POST', '/clone_settings'),
        # BA-P2-5: pause/resume publish project.paused/project.resumed on
        # the *global* stream (project: null, target: slug) so every open
        # Cropwright tab learns about it, same rationale as the other
        # project.* lifecycle events above.
        ('POST', '/pause'),
        ('POST', '/resume'),
    }
)

# Mutating routes whose write this fixture cannot reach yet: the seeded
# state lacks what the route acts on. They are excused ONLY from the
# "wrote nothing" check; every isolation check (foreign indexes, foreign
# markers in the response, events, 5xx, cache parity, foreign files)
# still applies to them.
UNSEEDED_WRITES: dict[tuple[str, str], str] = {
    ('POST', '/bakeoff/run'): 'needs a resolvable contender model list, not seeded',
    ('POST', '/train/promote/{job_id}'): 'needs an exported best.onnx, not seeded',
    ('DELETE', '/models/{model_name}'): 'see EXPECTED_5XX: no live Triton to confirm the unload',
    ('POST', '/vlm/endpoints/active/rollback'): (
        'the seeded previous endpoint was deleted (409 previous_deleted, the marker this '
        'sweep plants); rollback success is covered by test_vlm_hot_switch_api.py'
    ),
    ('POST', '/prompt_packs/active/rollback'): (
        'a fresh per-slug config store has no prior activation to roll back to '
        '(409 no_previous); rollback success is covered by test_prompt_packs_router.py'
    ),
    ('POST', '/region_profiles/active/rollback'): (
        'a fresh per-slug config store has no prior activation to roll back to '
        '(409 no_previous); rollback success is covered by test_region_profiles_router.py'
    ),
    ('POST', '/open_vocab/active/rollback'): (
        'a fresh per-slug config store has no prior activation to roll back to '
        '(409 no_previous); rollback success is covered by test_open_vocab_router.py'
    ),
}


# Long-lived SSE streams: the per-project delivery they serve is proven by
# the event checks below and tests/projects/test_event_hub_project_filter.py.
STREAMING_ROUTES: frozenset[str] = frozenset({f'{SCOPED}/events', f'{SCOPED}/pipeline/events'})
