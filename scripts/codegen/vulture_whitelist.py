"""Reviewed false positives for ``check_vulture.py``.

vulture cannot see a name that a framework, a table of strings or a serialiser reaches.
Each name below is read as an attribute of ``_`` (vulture counts the read as a use) and
carries the reason on its line. Add a name only after checking that nothing in the scanned
trees calls it; dead code is deleted instead.
"""

from types import SimpleNamespace


_ = SimpleNamespace()


# Framework hooks and protocols (called by name from outside this repo)
_.do_GET  # http.server.BaseHTTPRequestHandler dispatches do_<VERB> by name
_.do_POST  # http.server.BaseHTTPRequestHandler dispatches do_<VERB> by name
_.log_message  # http.server.BaseHTTPRequestHandler hook, overridden to silence request logs
_.isatty  # file-like protocol: logging and tqdm probe stream.isatty()
_.model_post_init  # pydantic v2 hook, called by the model constructor
_.exc_text  # logging.LogRecord attribute the formatter reads; reset so a repeat is not re-rendered
_.atexit  # weakref.finalize attribute, read by the weakref module
_.postprocess  # ultralytics Predictor method replaced on the instance; the framework calls it
_.intra_op_num_threads  # onnxruntime SessionOptions attribute, read by the runtime
_.catIds  # pycocotools COCOeval.params attribute, read by the evaluator
_.connect_unix_socket  # httpcore AsyncNetworkBackend protocol method, called by the connection pool
_.downgrade  # Alembic calls a revision module's downgrade() by name
_.down_revision  # Alembic reads the revision module attributes by name
_.branch_labels  # Alembic reads the revision module attributes by name
_.depends_on  # Alembic reads the revision module attributes by name
_.get_col_spec  # SQLAlchemy UserDefinedType hook, called by the DDL compiler
_.get_next  # onnxruntime CalibrationDataReader protocol, called by the quantizer
_.rewind  # onnxruntime CalibrationDataReader protocol, called by the quantizer
_.class_agnostic  # ultralytics export argument, read through hasattr/getattr in src/ultralytics_patches
_.mask_resolution  # ultralytics export argument, read through hasattr/getattr in src/ultralytics_patches
_.pooler_scale  # ultralytics export argument, read through hasattr/getattr in src/ultralytics_patches
_.sampling_ratio  # ultralytics export argument, read through hasattr/getattr in src/ultralytics_patches
_.topk_all  # ultralytics export argument, read through hasattr/getattr in src/ultralytics_patches

# Looked up by string
_.preset_none  # docker/trainer/augment.py resolves preset_<id> through globals() for each catalog id
_.preset_balanced_default  # docker/trainer/augment.py resolves preset_<id> through globals() for each catalog id
_.preset_outdoor_scene  # docker/trainer/augment.py resolves preset_<id> through globals() for each catalog id
_.preset_heavy_tilt  # docker/trainer/augment.py resolves preset_<id> through globals() for each catalog id
_.preset_low_light  # docker/trainer/augment.py resolves preset_<id> through globals() for each catalog id
_.preset_text_targets  # docker/trainer/augment.py resolves preset_<id> through globals() for each catalog id
_.preset_text_targets_aggressive  # docker/trainer/augment.py resolves preset_<id> through globals() for each catalog id
_.FaissFlatAdapter  # scripts/bench/vectordb/run.py imports adapters by module and class name from a table
_.LanceAdapter  # scripts/bench/vectordb/run.py imports adapters by module and class name from a table
_.MilvusAdapter  # scripts/bench/vectordb/run.py imports adapters by module and class name from a table
_.OpenSearchAdapter  # scripts/bench/vectordb/run.py imports adapters by module and class name from a table
_.PgvectorAdapter  # scripts/bench/vectordb/run.py imports adapters by module and class name from a table
_.QdrantAdapter  # scripts/bench/vectordb/run.py imports adapters by module and class name from a table
_.write_abs  # public writer API used by examples/bakeoff/*/converters.py, which are outside the scanned trees

# Wire and job-file fields (dataclass or pydantic fields written in code, serialised by asdict or model_dump under the same name)
_.error_detail  # serialised to the wire or the job file by asdict/model_dump
_.skipped_known  # serialised to the wire or the job file by asdict/model_dump
_.walked  # serialised to the wire or the job file by asdict/model_dump
_.holdout_frozen  # serialised to the wire or the job file by asdict/model_dump
_.items_created  # serialised to the wire or the job file by asdict/model_dump
_.items_noop  # serialised to the wire or the job file by asdict/model_dump
_.items_reconciled_removed  # serialised to the wire or the job file by asdict/model_dump
_.items_updated  # serialised to the wire or the job file by asdict/model_dump
_.label_conflicts_locked  # serialised to the wire or the job file by asdict/model_dump
_.labels_written  # serialised to the wire or the job file by asdict/model_dump
_.parents_detected  # serialised to the wire or the job file by asdict/model_dump
_.proposals_created  # serialised to the wire or the job file by asdict/model_dump
_.proposals_merged  # serialised to the wire or the job file by asdict/model_dump
_.images_deleted  # serialised to the wire or the job file by asdict/model_dump
_.images_kept  # serialised to the wire or the job file by asdict/model_dump
_.items_kept_human_edited  # serialised to the wire or the job file by asdict/model_dump
_.items_reinstated  # serialised to the wire or the job file by asdict/model_dump
_.computed_groups  # serialised to the wire or the job file by asdict/model_dump
_.pinned_groups  # serialised to the wire or the job file by asdict/model_dump
_.pinned_overridden_by_holdout  # serialised to the wire or the job file by asdict/model_dump
_.pinned_split_conflicts  # serialised to the wire or the job file by asdict/model_dump
_.detector_class_id  # serialised to the wire or the job file by asdict/model_dump
_.updated_count  # serialised to the wire or the job file by asdict/model_dump
_.registry_snapshot_path  # serialised to the wire or the job file by asdict/model_dump
_.ocr_error  # serialised to the wire or the job file by asdict/model_dump

# Test seams and oracles: production code whose only caller is the named test. Each has a
# non-test reason to exist (state reset, a contract table the plan requires), so it stays.
_._reset_field_coverage_cache  # src/services/curation/field_coverage.py; called by tests/curation/test_config_store_settings_bridge.py to clear the module cache
_._reset_registry_for_tests  # src/services/detection/profile_registry.py; called by tests/conftest.py and the profile-registry tests to clear the registry
_.set_project_registry  # src/services/projects/registry.py; called by tests/conftest.py and the project tests to install the registry they need
_.reset_reachability_cache  # src/services/resource_links.py; called by tests/curation/test_resource_links.py to clear the cache
_.wait_for_inflight_refresh  # src/services/resource_links.py; awaited by tests/curation/test_resource_links.py for the background refresh
_.cross_project_access_count  # src/services/projects/guard_errors.py; asserted on by tests/projects/test_worker_leak.py and test_opensearch_guard.py
_.CHOICE_SOURCES  # src/routers/curation/config_vocabulary.py; read by tests/curation/test_config_vocabulary.py (contract check against the vocabulary)
_.ALL_STEPS  # src/services/curation/next_steps.py; enumerated by tests/curation/test_next_steps_paths.py so a new builder must be covered
_.QUEUE_PROFILE  # src/config/queue.py (plan 03a 1B-2); read by tests/config/test_queue_settings.py and test_job_kind_fields.py (consumers-from-compose check)
_.TABLE_USE  # src/db/table_use.py (plan 02d section 0.1); read by tests/db/test_research_schema_matches_plan.py (a new table must state its writer and reader)

# Planned callers that do not exist yet; each names the plan that owns the caller.
_.PlanReport  # src/schemas/processing_plan.py; response model of the plan-aware routes, docs/design/processing_plans_plan.md phase P2
_.ProcessingPlan  # src/schemas/processing_plan.py; request model of the plan-aware routes, docs/design/processing_plans_plan.md phase P2
_.to_request  # ProcessingPlan.to_request; the route that parses the plan, docs/design/processing_plans_plan.md phase P2
_.crop_key  # src/domain/storage_keys.py; docs/design/v070_rebuild/03a_stage1_foundation.md 1E-4 (crop sizing work); only tests/clients/storage/test_crop_key_variant.py calls it today

# Registered or read by a framework through a name, a decorator or a protocol
_._on_worker_init  # src/tasks/app.py; connected to Celery's worker_init signal by decorator
_._on_beat_init  # src/tasks/app.py; connected to Celery's beat_init signal by decorator
_._on_worker_process_shutdown  # src/tasks/app.py; connected to Celery's worker_process_shutdown signal by decorator
_.basic_auth  # scripts/flower/flowerconfig.py; an option Flower reads by name from its --conf file
_._send_wakes  # src/services/events/publish.py; a SQLAlchemy after_commit listener registered by decorator
_._drop_wakes  # src/services/events/publish.py; a SQLAlchemy after_rollback listener registered by decorator
_.StreamingResponse  # src/app/factory.py; named only in the string cast cast('StreamingResponse', response) (ruff TC006 requires the quotes)
_.list_all  # src/routers/curation/research_configs.py; route handler (GET /{kind}) registered by decorator inside the config-kind factory
_.put_one  # src/routers/curation/research_configs.py; route handler (PUT /{kind}/{name}) registered by decorator inside the factory
_.delete_one  # src/routers/curation/research_configs.py; route handler (DELETE /{kind}/{name}) registered by decorator inside the factory
_.__getattr__  # type: ignore[attr-defined]  # src/clients/__init__.py; PEP 562 module attribute hook
_.ServerLive  # docker/test/fake_triton.py; called by gRPC by Triton's method name
_.ServerReady  # docker/test/fake_triton.py; called by gRPC by Triton's method name
_.__qualname__  # src/tasks/runtime/register.py; assigned so Celery names the task by it
_.registry_sha  # src/services/training/jobs.py; stored in training_runs.spec, read by name in docker/trainer/job_manifest.py
_.tb  # src/utils/stage_timing.py; the __exit__ signature of the context manager protocol
_.update_ready  # src/clients/triton_router.py; the router is fed by src/services/serving_router.py; the callers (the pool and the encoders choosing a server per call) land with the post-tag two-card wave, docs/design/v070_rebuild/09g_research_gpu_pipeline.md
_.configure_router  # src/clients/triton_client_manager.py; the process start-up that builds the router from src/services/serving_router.py lands with the post-tag two-card wave, docs/design/v070_rebuild/09g_research_gpu_pipeline.md
_.build_router  # src/services/serving_router.py; see configure_router above
