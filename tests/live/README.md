# Live suite: what each module needs

The live suite (`tests/live/`, harness in `docker/test/README.md`) runs against one stack. Which
modules may run beside others is recorded in `_classes.py` and checked offline by
`tests/test_live_classification.py` (a module in no set fails; an isolated module that uses a
stack-acting helper fails). A module not in `ISOLATED` gets the `live_serial` marker.

Run it in two parts (pytest-xdist 3.8.0; `--dist loadfile` keeps every file on one worker, so a
module-scoped project fixture is built once and a file's tests keep their order; see
https://pytest-xdist.readthedocs.io/en/stable/distribution.html):

```bash
.venv/bin/python -m pytest tests/live -q --no-cov -m 'live and not live_serial' -n 3 --dist loadfile
.venv/bin/python -m pytest tests/live -q --no-cov -m 'live and live_serial'
```

The session fixture `seeded` wipes and reseeds the stack; under xdist the first worker does it and
the others wait on a `fcntl` lock (the recipe of the pytest-xdist how-to, "Making session-scoped
fixtures execute only once"). The other session fixtures only read, or reset a mode no isolated
module sets.

## Stack guard (return code 4)

Around every test of a stack-acting module (`_classes.STACK_ACTING`) the autouse fixture
`stack_healthy_around_stack_acting_tests` (`conftest.py`, logic in `_heal.py`) checks that `api`,
`jobs-worker`, `periodic-worker` and `jobs-beat` run and are healthy. If not, it heals once
(unpause, `up -d`, wait for health; every step is tried even when an earlier one times out) within
about 6 minutes, and otherwise ends the session with `pytest.exit(..., returncode=4)` and a table of
each service's State, Health and ExitCode. Preflight refusals stay code 3. Stack-acting modules run
without xdist (second command above); in an xdist worker the guard fails the test instead of
exiting, because the xdist docs do not say what `pytest.exit` does in a worker.

When you see code 4: check the Docker daemon first (`docker info`; a stalled daemon makes compose
calls hit their timeout), then the service the message names (`docker compose -p <project> ps -a`,
its logs). A job wait that times out while the job is `queued` prints the same service table.

## Classes

(a) isolated: its own project only, data read by its project or a unique id.
(b) stack-acting: stops, restarts, kills or reconfigures a service, switches a fake's mode, or
asserts stack-wide state.
(c) unsure: works on the seeded `default` project (shared cohorts, class registry, settings) or
starts a container of its own; stays serial until each test is shown to touch only what it creates.

| Module | Class | Line that decides it |
|---|---|---|
| test_live_combine | a | `ingest_project` x2 plus a combine target; the project list is filtered to the target |
| test_live_core_ingest | a | `/ingest` with a unique `image_id`; hits filtered by it (shared core indexes, only writer) |
| test_live_crop_space | a | `ingest_project('cropspace')` |
| test_live_detector_class | a | module `ingest_project('detcls')` |
| test_live_ingest | a | module `ingest_project('ingest')`; the events stream is the project's own (`{project.prefix}/events`) |
| test_live_preset_documents | a | `ingest_project('preset')`; the one deployment-wide kind uses a unique name |
| test_live_region_drain_narrowed | a | `ingest_project('narrow')`; `/ingest/region_drain` of its project |
| test_live_search | a | module `ingest_project('search')` |
| test_queue_delivery | a | own jobs and unique task ids; `compose logs` and `psql` are read-only (the filter is the id) |
| test_control_plane_smoke | b | `_compose.kill('jobs-worker')` (l.149), Postgres and broker restarts |
| test_job_load | b | `start_burst` of jobs and replica-count compose calls (l.85) |
| test_job_recovery_real_handlers | b | `_compose.kill('jobs-worker')` (l.176), pause, `restart api` (l.282) |
| test_live_assist_loop | b | `fake_vlm.post('/__control', ...)` (l.371) |
| test_live_bulk_restart | b | `_compose.compose('restart', 'api')` (l.68) |
| test_live_engine_unavailable | b | `engine_refuses(...)` (l.59, l.106) |
| test_live_models_promote | b | asserts the whole `/models/exports` list is unchanged (l.59-66) |
| test_live_train | b | `docker_ps_snapshot()` of the whole project (l.71) |
| test_live_train_promote | b | deployment-wide `/models/` list (l.45) and a train on `default` |
| test_live_vlm | b | `compose('up', '--force-recreate', 'api')` (l.273), `fake_vlm` mode |
| test_one_vs_n | b | `reconfigure(...)` (l.277) |
| test_sse_across_replicas | b | `reconfigure(...)` (l.51) |
| test_class_registry_live | c | eight processes write the `default` registry (`compose('exec', ...)`, l.80) |
| test_live_annotation, test_live_bulk_occ, test_live_capture_map, test_live_classes, test_live_clusters, test_live_embedding_state, test_live_export, test_live_generic_detector, test_live_image_counts, test_live_images, test_live_labels, test_live_ocr, test_live_open_vocab, test_live_paging, test_live_regions, test_live_seed_from_detector, test_live_served_facts, test_live_whole_photo | c | `api_client` / `INDEXES` / `PROJECT_PREFIX`: the seeded `default` project, which they read and several change (classes, labels, regions, holdout) |
| test_live_region_text | c | own project, but `compose('run', ...)` starts a detection worker container (l.55) |
