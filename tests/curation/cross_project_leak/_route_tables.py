"""Per-route exemptions and policy tables the isolation sweep checks against."""

from __future__ import annotations

from curation.cross_project_leak._constants import SCOPED


# Routes whose data is rows in PostgreSQL, not documents in an OpenSearch index: this sweep watches
# OpenSearch accesses and files, so it cannot see them. Their isolation is row-level security,
# proven in the database tier (tests/db/test_row_level_security.py, tests/config_store/). The
# embedding space routes (spaces, build, promote, retire, re-index runs) are proven per route in
# tests/embedding_spaces/test_space_routes.py (another project's space or run is a 404).
POSTGRES_BACKED_PREFIXES: tuple[str, ...] = (
    '/cluster_profiles',
    '/assist_loop_presets',
    '/near_dup_profiles',
    '/embedding',
)

# Mutating routes that write nothing by design (read-only work under POST,
# or an event with no data write), and why. Every other mutating route must
# really write in the sweep.
NO_WRITE: dict[tuple[str, str], str] = {
    ('POST', '/datasets/preview'): 'scans a dataset read-only; writes nothing',
    ('POST', '/datasets/imports'): (
        'the sweep sends no mapping, so it is refused before anything is queued; the '
        'import job is covered in tests/families_b/test_dataset_import_job.py'
    ),
    ('POST', '/classes/merge'): 'queues the class_merge job; the job relabels (tests/families_b)',
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
    ('POST', '/label_ops/{op_id}/undo'): (
        "the sweep has no such operation (404); another project's operation id is covered in "
        'test_label_ops_isolation'
    ),
    ('POST', '/events/publish'): 'publishes an event (checked separately), writes no data',
    ('POST', '/ingest/path_lookup'): 'read-only lookup under POST',
    (
        'POST',
        '/ingest/policy/preview',
    ): 'counts what a candidate policy would embed; writes nothing',
    ('POST', '/search/image'): 'kNN search by a stored or uploaded picture; stores nothing',
    ('POST', '/train/preflight'): 'read-only validation under POST',
    ('POST', '/keymap/validate'): 'dry-run report; writes nothing',
    (
        'PUT',
        '/ingest/policy',
    ): 'writes the project_settings row, which this OpenSearch sweep cannot see',
    (
        'PUT',
        '/vlm/policy',
    ): 'writes the project_settings row, which this OpenSearch sweep cannot see',
    ('PUT', '/keymap'): 'writes the project_keymaps row, which this OpenSearch sweep cannot see',
    (
        'POST',
        '/keymap/reset',
    ): 'writes the project_keymaps row, which this OpenSearch sweep cannot see',
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
    ('POST', '/ingest_profiles/validate'): 'dry-run report; never writes',
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
_CLASS_REGISTRY_IN_POSTGRES = (
    'the class registry is rows in PostgreSQL keyed by project_id, not an OpenSearch index; '
    'its per-project isolation is proven by tests/db/test_class_registry.py and the row-level '
    'security tests'
)
_CONFIG_STORE_IN_POSTGRES = (
    'the config store (documents, revisions, activations, settings) is rows in PostgreSQL keyed '
    'by project_id, not an OpenSearch index; its per-project isolation is proven by '
    'tests/db/test_config_repo.py and the row-level security tests'
)
_CONFIG_STORE_ROUTES: tuple[tuple[str, str], ...] = (
    *(
        (method, f'/{kind}{suffix}')
        for kind in ('prompt_packs', 'region_profiles', 'ingest_profiles', 'open_vocab')
        for method, suffix in (
            ('POST', ''),
            ('PUT', '/{name}'),
            ('DELETE', '/{name}'),
            ('POST', '/{name}/clone'),
            ('POST', '/{name}/activate'),
        )
    ),
    *(
        ('POST', f'/{kind}/deactivate')
        for kind in ('region_profiles', 'ingest_profiles', 'open_vocab')
    ),
    ('PUT', '/settings'),
    ('POST', '/vlm/endpoints/{name}/activate'),
    ('POST', '/vlm/endpoints/deactivate'),
)
_JOB_ROWS_IN_POSTGRES = (
    'the route queues, cancels or pauses a job, which is a row in the PostgreSQL jobs table, not '
    'a document in an OpenSearch index; its project scope is proven by tests/jobs and the '
    'row-level security tests, and the job itself writes later in a worker'
)
_NEAR_DUP_ROWS_IN_POSTGRES = (
    'near-duplicate runs, groups and decisions are rows in PostgreSQL keyed by project_id; '
    'isolation is proven by tests/db/test_near_dup_schema.py and the row-level security tests'
)
_NO_SUCH_ROW = (
    'the sweep has no such row (404); the route reads and writes PostgreSQL, which this '
    'OpenSearch sweep cannot see (row-level security tests)'
)
_NO_TRAINER = (
    'refused before any write: the sweep has no live gpu_train consumer (409), the same answer '
    'a project without a trainer gets'
)
UNSEEDED_WRITES: dict[tuple[str, str], str] = {
    **dict.fromkeys(
        (
            ('POST', '/near_duplicates/runs'),
            ('POST', '/source_backfill/runs'),
            ('POST', '/pipeline/auto_label/start'),
            ('POST', '/pipeline/auto_label/cancel'),
            ('POST', '/pipeline/auto_label/pause'),
            ('POST', '/pipeline/auto_label/resume'),
            ('POST', '/scores/compute'),
            ('POST', '/scores/cancel'),
            ('POST', '/select/diverse'),
            ('POST', '/select/cancel'),
            ('POST', '/viz/projection/rebuild'),
            ('POST', '/viz/projection/cancel'),
            ('POST', '/vlm/label_batch'),
            ('POST', '/crops/move'),
            ('PUT', '/crops/batch_label'),
            ('POST', '/probe/run'),
            ('POST', '/probe/cancel'),
            ('POST', '/clusters/runs'),
            ('POST', '/regions/fp_centroids/build'),
            ('POST', '/train/cancel/{job_id}'),
            ('POST', '/train/cancel_campaign/{campaign_id}'),
        ),
        _JOB_ROWS_IN_POSTGRES,
    ),
    ('POST', '/near_duplicates/decisions'): _NEAR_DUP_ROWS_IN_POSTGRES,
    ('DELETE', '/near_duplicates/decisions/{decision_id}'): _NO_SUCH_ROW,
    ('PUT', '/clusters/activation'): _NO_SUCH_ROW,
    ('POST', '/assist/loops/{loop_id}/pause'): _NO_SUCH_ROW,
    ('PATCH', '/assist/loops/{loop_id}/budget'): _NO_SUCH_ROW,
    ('POST', '/pipeline/auto_label'): (
        'the synchronous run with its defaults (no embed-missing, no VLM stage) over fully '
        'embedded seed items has nothing to write; its scoped reads are still swept, and the '
        'writing stages are covered by tests/curation/test_pipeline.py'
    ),
    ('POST', '/train/start'): _NO_TRAINER,
    ('POST', '/train/start_campaign'): _NO_TRAINER,
    ('PUT', '/models/{model_name}/sharing'): (
        'writes the model registry row in PostgreSQL, which this OpenSearch sweep cannot see'
    ),
    **dict.fromkeys(_CONFIG_STORE_ROUTES, _CONFIG_STORE_IN_POSTGRES),
    ('POST', '/pause'): 'writes projects.paused_at in Postgres; tests/projects/test_pause_pg.py',
    ('POST', '/resume'): 'writes projects.paused_at in Postgres; tests/projects/test_pause_pg.py',
    ('POST', '/classes'): _CLASS_REGISTRY_IN_POSTGRES,
    ('PUT', '/classes/{class_id}'): _CLASS_REGISTRY_IN_POSTGRES,
    ('POST', '/classes/{class_id}/restore'): _CLASS_REGISTRY_IN_POSTGRES,
    ('POST', '/classes/{class_id}/deprecate'): _CLASS_REGISTRY_IN_POSTGRES,
    ('POST', '/classes/merge'): _CLASS_REGISTRY_IN_POSTGRES,
    ('POST', '/classes/seed_from_detector'): _CLASS_REGISTRY_IN_POSTGRES,
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
    ('POST', '/ingest_profiles/active/rollback'): (
        'a fresh per-slug config store has no prior activation to roll back to '
        '(409 no_previous); rollback success is covered by test_ingest_profiles_router.py'
    ),
    ('POST', '/open_vocab/active/rollback'): (
        'a fresh per-slug config store has no prior activation to roll back to '
        '(409 no_previous); rollback success is covered by test_open_vocab_router.py'
    ),
}


# Long-lived SSE streams: the per-project delivery they serve is proven by
# the event checks below and tests/events/test_listener_fanout.py.
STREAMING_ROUTES: frozenset[str] = frozenset({f'{SCOPED}/events', f'{SCOPED}/pipeline/events'})

# Routes whose ``{job_id}`` is parsed as a uuid (a job row), so the sweep's marker id would be a 400.
UUID_JOB_ID: frozenset[tuple[str, str]] = frozenset(
    {('POST', '/train/promote/{job_id}'), ('POST', '/train/cancel/{job_id}')}
)
