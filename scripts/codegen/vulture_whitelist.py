"""Reviewed false positives for ``check_vulture.py``.

vulture cannot see a name that a framework, a table of strings or a serialiser reaches.
Each name below is read as an attribute of ``_`` (vulture counts the read as a use) and
carries the reason on its line. Add a name only after checking that nothing in the scanned
trees calls it; dead code is deleted instead.
"""

from types import SimpleNamespace


_ = SimpleNamespace()


# Framework hooks and protocols (called by name from outside this repo)
_.forward  # torch.nn.Module.forward, called by torch.onnx.export and __call__
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
_.backend_detail  # serialised to the wire or the job file by asdict/model_dump
_.holdout_frozen  # serialised to the wire or the job file by asdict/model_dump
_.images_created  # serialised to the wire or the job file by asdict/model_dump
_.images_reused  # serialised to the wire or the job file by asdict/model_dump
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
_.projection_version  # serialised to the wire or the job file by asdict/model_dump
_.computed_groups  # serialised to the wire or the job file by asdict/model_dump
_.pinned_groups  # serialised to the wire or the job file by asdict/model_dump
_.pinned_overridden_by_holdout  # serialised to the wire or the job file by asdict/model_dump
_.pinned_split_conflicts  # serialised to the wire or the job file by asdict/model_dump
_.detector_class_id  # serialised to the wire or the job file by asdict/model_dump
_.updated_count  # serialised to the wire or the job file by asdict/model_dump
_.registry_snapshot_path  # serialised to the wire or the job file by asdict/model_dump
_.ocr_error  # serialised to the wire or the job file by asdict/model_dump

# Held for a reason other than a read
_.curation_knn_warmup_task  # AppResources keeps the warmup task referenced so asyncio does not garbage-collect it

# Test seams and oracles (production code whose only caller is the test that needs the seam)
_._reset_for_tests  # clears module state between tests
_._reset_field_coverage_cache  # clears a module cache between tests
_._reset_registry_for_tests  # clears the profile registry between tests
_.to_source_frame  # region_eval_geometry helper kept for a crop-relative ingestion path; pinned by tests/curation/test_region_eval.py (its __all__ entry used to hide it)
_.set_project_registry  # installs the project registry a test needs (the conftest and 100+ tests call it)
_.reset_reachability_cache  # clears a module cache between tests
_.wait_for_inflight_refresh  # lets a test await the background refresh it triggered
_.cross_project_access_count  # the guard counter the isolation tests assert on
_.read_liveness  # reads back the heartbeat file write_liveness wrote; the per-project runtime tests assert on it
_._guard_condition_matches  # plain-Python twin of the painless guard, the oracle the parity tests compare against
_.CHOICE_SOURCES  # the choices_from table the config vocabulary contract test checks against the vocabulary
_.ALL_STEPS  # the registry the next-steps completeness test enumerates

# Landed ahead of their routes (#67 P1): wire models and their converter, first used by the
# /analyze plan executor in P2. Delete these entries in that change.
_.PlanReport  # src/schemas/processing_plan.py, response model of the plan-aware routes (P2)
_.ProcessingPlan  # src/schemas/processing_plan.py, request model of the plan-aware routes (P2)
_.to_request  # ProcessingPlan.to_request, called by the route that parses the plan (P2)

# Landed ahead of their callers (B2 stage 1): the storage seam is first used by the repository and
# job lanes of stage 2 onward. Delete these entries when the first production caller lands.
_.build_rooted_store  # src/clients/storage/roots.py, the BlobStore every storage consumer builds
_.storage_root_paths  # src/config/storage.py, the physical roots build_rooted_store maps
_.crop_key  # src/clients/storage/keys.py, the crop-cache key; its first caller is the crop sizing work

# Landed ahead of their consumers (B2 stage 1, lane 1B): the advisory-lock classes are first used
# by the periodic task bodies and the beat/migrate runners of later lanes, and QUEUE_PROFILE by
# the consumer-from-compose test. Delete each entry in the change that adds its production caller.
_.BEAT  # src/config/advisory_locks.py, session lock of the single beat (stage 6)
_.INGEST_DISPATCH  # src/config/advisory_locks.py, periodic ingest dispatcher body (stage 4)
_.OPEN_VOCAB_SWEEP  # src/config/advisory_locks.py, periodic open-vocabulary sweep (stage 4)
_.CACHE_PRUNE  # src/config/advisory_locks.py, periodic cache prune (stage 6)
_.GPU_ARBITER  # src/config/advisory_locks.py, periodic arbiter tick (stage 6)
_.CLUSTER_REFRESH  # src/config/advisory_locks.py, periodic clustering refresh (stage 6)
_.REAPER  # src/config/advisory_locks.py, the reaper tick (stage 3)
_.NEAR_DUP_RESCORE  # src/config/advisory_locks.py, periodic near-duplicate rescore (stage 4)
_.QUEUE_PROFILE  # src/config/queue.py, queue -> compose profile table the consumers-from-compose test reads
_._on_worker_init  # src/tasks/app.py, connected to Celery's worker_init signal by decorator
_._on_beat_init  # src/tasks/app.py, connected to Celery's beat_init signal by decorator
_.make_app  # src/tasks/runtime/gpu_app.py, called by the trainer, export and evaluator images (stage 5)
_.mark_queued  # src/clients/task_queue.py, called by start_job in the job service (stage 3)
_.clear_queued  # src/clients/task_queue.py, called by the job claim and the reaper (stage 3)
_.queued_state  # src/clients/task_queue.py, read by reaper rule c (stage 3)
_.set_inventory_writer  # src/tasks/runtime/gpu_app.py, called by the job runtime once it owns the database writer (stage 3)
_.refresh_inventory  # src/tasks/runtime/gpu_app.py, called on every job heartbeat (stage 3)
_.record_capabilities  # InventoryWriter protocol method, called by start_inventory on the registered writer
_.touch_devices  # InventoryWriter protocol method, called by refresh_inventory on the registered writer
_.upsert_devices  # InventoryWriter protocol method, called by start_inventory on the registered writer
_.capabilities  # parameter of the InventoryWriter protocol method, filled by the registered writer
_.bind_request_id  # src/core/logging.py, binds the id on both ContextVars; tests/integration/test_request_id_propagation.py asserts the binding
_.basic_auth  # scripts/flower/flowerconfig.py, an option Flower reads by name from its --conf file

# Landed ahead of their consumers (B2 stage 1, lane 1F): the contract models are first used by the
# engine export (5B) and the embedding catalog sync (ES-A). Delete each entry with its caller.
_.EngineOutputContract  # src/schemas/models.py, validated by the engine export (stage 5)
_.EncoderInputSpec  # src/schemas/models.py, validated by the embedding catalog sync (stage 2)
_.TABLE_USE  # src/db/table_use.py, pure data read only by tests/db/test_research_schema_matches_plan.py (a new table must state its writer and reader)

# Event publish and retention (B2 stage 3, lane 3B).
_.delete_older_than  # src/db/repos/events.py, called by reaper rule f (lane 3A)
_._send_wakes  # src/services/events/publish.py, a SQLAlchemy after_commit listener registered by decorator
_._drop_wakes  # src/services/events/publish.py, a SQLAlchemy after_rollback listener registered by decorator
_.StreamingResponse  # src/app/factory.py, named only in a string cast
# Landed ahead of their callers (B2 stage 2, lane 2A): the repository is wired by the registry,
# guard and lifecycle commits of the same lane. Delete each entry with its caller.
_.load_project  # src/db/repos/projects.py, called by the 0.6.0 upgrade step (lane 2E)
_.current_storage_prefix  # src/config/project_context.py, the bound project's storage key prefix; called by the storage users of lanes 2B-2D and stage 3
_.current_project_id  # src/config/project_context.py, the bound project's database id; called by the repositories of lanes 2B-2D
# Landed ahead of their consumers (B2 stage 2, lane 2C): the upgrade load (lane 2E) and the readers of
# the poll and the runtime reports. Delete each entry with its caller.
_.list_probes  # src/db/repos/config.py, read by the global snapshot load (2C-3)
_.list_runtime_reports  # src/db/repos/config.py, read by the applied-config view (2C-4)
_.upsert_runtime_report  # src/db/repos/config.py, written by the worker runtime report (2C-4)
_.load_activation_events  # src/db/repos/config_activations.py, called by the 0.6.0 upgrade (lane 2E)
_.load_vlm_endpoint  # src/db/repos/config.py, called by the 0.6.0 upgrade (lane 2E)
_.load_config_document  # src/db/repos/config.py, called by the 0.6.0 upgrade (lane 2E)
_.load_setting  # src/db/repos/settings.py, called by the 0.6.0 upgrade (lane 2E)
_.load_keymap  # src/db/repos/settings.py, called by the 0.6.0 upgrade (lane 2E)

# Embedding registry: public functions whose production callers land with the projects, config,
# upgrade and re-index work. Delete each entry with its first production caller.
_.sync_catalog  # src/services/embedding_spaces/catalog.py, called by scripts/db/migrate.py after Alembic
_.embedding_space  # src/services/embedding_spaces/resolve.py, called by every vector reader
_.note_config_epoch  # src/services/embedding_spaces/resolve.py, called by the config_epoch poll
_.default_space_rows  # src/services/embedding_spaces/spaces.py, called by the upgrade (the project create calls it already)
_.add_spaces  # src/db/repos/embedding_spaces.py, called with default_space_rows by the upgrade (the project create calls it already)
_.default_space_rows  # src/services/embedding_spaces/spaces.py, called by the project create and the upgrade
_.add_spaces  # src/db/repos/embedding_spaces.py, called with default_space_rows by the project create and the upgrade
_.register_space  # src/services/embedding_spaces/spaces.py, called by the space routes
_.begin_build  # src/services/embedding_spaces/spaces.py, called by the re-index job
_.mark_ready  # src/services/embedding_spaces/spaces.py, called by the re-index job
_.retire_space  # src/services/embedding_spaces/spaces.py, called by the space routes
_.finish_retire  # src/services/embedding_spaces/spaces.py, called by the re-index job in clear mode
_.list_project_spaces  # src/services/embedding_spaces/spaces.py, called by the space routes

# Route handlers registered by decorator inside a factory (one set per config kind).
_.list_all  # src/routers/curation/research_configs.py, GET /{kind}
_.get_one  # src/routers/curation/research_configs.py, GET /{kind}/{name}
_.put_one  # src/routers/curation/research_configs.py, PUT /{kind}/{name}
_.delete_one  # src/routers/curation/research_configs.py, DELETE /{kind}/{name}
