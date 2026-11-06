"""Where a 0.6.0 install keeps its control data: the one place that names the deleted indexes,
documents and files.

Nothing else in 0.7.0 reads any of these. The shapes are those the 0.6.0 writers produced
(``src/services/config_store/index.py``, ``projects/registry.py``, ``clients/curation_opensearch``,
the ``FileJob`` stores); the sections below name the writer each one came from.
"""

from __future__ import annotations


# --- OpenSearch -------------------------------------------------------------------------------
PROJECTS_INDEX = (
    'op_projects'  # one doc per project: ``project:<slug>``; ``meta:projects_revision``
)
GLOBAL_CONFIGS_INDEX = 'op_global_configs'  # VLM endpoints, probes, ``local_vlm:desired``
PROJECT_DOC_PREFIX = 'project:'
PROJECTS_REVISION_DOC_ID = 'meta:projects_revision'

# Config-store documents: ``<prefix>:<name>`` current, ``<prefix>:<name>@<rev>`` immutable copy.
CONFIG_PREFIX_TO_KIND: dict[str, str] = {
    'pack': 'prompt_pack',
    'profile': 'region_profile',
    'vlm': 'vlm_endpoint',
    'ovset': 'open_vocab_set',
    'ingest': 'ingest_profile',
}
ACTIVATION_DOC_PREFIX = 'activation:'
ACTIVATION_EVENT_DOC_PREFIX = 'activation_event:'
RUNTIME_DOC_PREFIX = 'runtime:'
LEASE_DOC_PREFIX = 'lease:'
KEYMAP_DOC_ID = 'keymap:default'
SETTINGS_DOC_ID = 'default'
CLASS_REGISTRY_DOC_ID = 'class_registry'  # main only: a development install of ``main`` has it
VIZ_STATE_DOC_ID = 'current'
LOCAL_DESIRED_DOC_ID = 'local_vlm:desired'

# Documents that are transient (leases, per-process runtime reports) are neither exported nor
# restored: a rollback starts the 0.6.0 services, which write their own.
TRANSIENT_DOC_PREFIXES = (LEASE_DOC_PREFIX, RUNTIME_DOC_PREFIX)

# Roles of a project's index map (``resources.indexes``) that are control data and go away.
CONTROL_ROLES = (
    'configs',
    'settings',
    'umap_state',
    'umap_viz_state',
    'classes',
    'labels_confirmed',
)
KEPT_ROLES = ('images', 'items')

# --- files ------------------------------------------------------------------------------------
CLASS_REGISTRY_FILE = 'class_registry.json'
CLASS_REGISTRY_RENAMED = 'class_registry.json.migrated-0.7.0'
PAUSE_FLAG = 'pipeline_paused.flag'
HOLDOUT_DIR = 'test_holdout'
LABEL_OPS_DIR = 'label_ops'
LAST_CLUSTERING_FILE = 'last_clustering.json'

# The job-file families under the shared jobs volume: (family, env var, default directory).
# Each keeps ``<base>/projects/<slug>/<job>/state.json`` (a ``FileJob``).
JOB_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ('bulk_ingest', 'OP_BULK_INGEST_DIR', '/jobs/bulk_ingest'),
    ('dataset_import', 'OP_DATASET_IMPORTS_DIR', '/jobs/imports'),
    ('reprocess', 'OP_REPROCESS_JOBS_DIR', '/jobs/reprocess'),
    ('probe', 'OP_PROBE_JOBS_DIR', '/jobs/probe'),
    ('scores', 'OP_SCORES_STATE_DIR', '/jobs/scores'),
    ('select', 'OP_SELECT_JOBS_DIR', '/jobs/select'),
    ('viz', 'OP_VIZ_JOBS_DIR', '/jobs/viz'),
    ('assist', 'OP_ASSIST_JOBS_DIR', '/jobs/assist'),
    ('assist_loop', 'OP_ASSIST_LOOP_JOBS_DIR', '/jobs/assist_loop'),
    ('region_drain', 'OP_REGION_DRAIN_STATE_DIR', '/jobs/region_drain'),
)
COMBINE_ENV = ('OP_COMBINE_JOBS_DIR', '/jobs/combine')
MODEL_REPO_ENV = ('OP_TRITON_MODEL_REPO', '/app/models')

# Statuses a family's ``state.json`` carries while the job is alive (the 0.6.0 ``ACTIVE_STATUSES``).
ACTIVE_JOB_STATUSES = frozenset({'queued', 'running', 'paused_backpressure', 'undoing', 'starting'})
