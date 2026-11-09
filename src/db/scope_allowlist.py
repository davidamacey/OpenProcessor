"""Which modules may open a deployment-wide or a core transaction.

``project_tx`` needs no permission: it narrows access. ``deployment_tx`` sees every project's
rows and ``core_tx`` sees the rows that belong to no project, so only modules that must are
listed. Entries are ``fnmatch`` patterns on the importing
module's name. A module that needs a new entry adds one line here with its reason; a test module under ``tests/`` is always
allowed because it is never imported by production code.
"""

from __future__ import annotations


DEPLOYMENT_TX_MODULES: tuple[str, ...] = (
    'src.tasks.reaper',  # stuck-job recovery runs across all projects
    'src.tasks.periodic',  # beat tasks (cache prune, retention, ingest dispatch)
    'src.tasks.beat_schedule',
    'src.tasks.queue_liveness',  # the cluster consumer's liveness row belongs to no project
    'src.tasks.runtime',  # the job wrapper claims a job by id before it knows the project
    'src.services.events',  # event publish and the global event stream read across projects
    'src.services.periodic',  # the beat tick bodies read the projects they act on
    'src.services.jobs',  # job start and status across scopes, reaper rules
    'src.services.projects',  # the project registry, create, patch, delete and its job
    'src.services.embedding_spaces',  # the model catalog sync and the project lookup of the space resolver
    'src.services.classes',  # slug to project id before a project-scoped class registry read
    'src.services.upgrade',  # the 0.6.0 control-data step
    'src.services.training.model_registry',  # who owns, shares and serves a Triton model name crosses projects
    'src.services.training.trainer_reachability',  # reads worker_capabilities and gpu_devices, which belong to no project
    'src.services.training.bakeoff_pins',  # retention pins what any project's queued bake-off still scores
    'src.services.config_store',  # resolves a slug to its id; deployment-wide endpoints and profiles
    'src.services.model_export_jobs',  # deployment-wide export jobs (no project) and the models they build
    'scripts.db',  # migration runner and db verify
    'scripts.opensearch',  # the one-shot index migrate reads the projects table
    'scripts.admin',  # admin command-line tools
)

CORE_TX_MODULES: tuple[str, ...] = (
    'src.routers.persons',  # core face identities
    'src.routers.query_duplicates',
    'src.services.face_identity*',
    'src.services.duplicate_detection*',
    'src.services.core_ingest*',  # the core ingest services and routers
    'src.routers.ingest',
    'src.tasks.near_dup_core',  # the core near-duplicate and person-grouping handlers
    'src.services.near_duplicates.core*',  # core pending rows, written by core ingest
    'src.services.near_duplicates.content_key',  # the core exact-content rows (no project)
    'src.tasks.clustering',
)
