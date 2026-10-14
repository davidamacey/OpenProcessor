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
_.set_project_registry  # installs the project registry a test needs (the conftest and 100+ tests call it)
_.reset_reachability_cache  # clears a module cache between tests
_.wait_for_inflight_refresh  # lets a test await the background refresh it triggered
_.cross_project_access_count  # the guard counter the isolation tests assert on
_.read_liveness  # reads back the heartbeat file write_liveness wrote; the per-project runtime tests assert on it
_._guard_condition_matches  # plain-Python twin of the painless guard, the oracle the parity tests compare against
_.CHOICE_SOURCES  # the choices_from table the config vocabulary contract test checks against the vocabulary
_.ALL_STEPS  # the registry the next-steps completeness test enumerates
