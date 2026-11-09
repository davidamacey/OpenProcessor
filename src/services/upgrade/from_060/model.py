"""What the load carries out of the bundle: plain frozen records in 0.7.0 vocabulary.

The parsers (``parse_*.py``) turn 0.6.0 documents and files into these; the writers
(``control_rows.py`` through the stage-2 repositories, ``direct_rows.py`` for tables with no
repository yet) turn them into rows. Nothing here knows an OpenSearch index name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import datetime as dt
    import uuid


class UnresolvedBuiltinError(RuntimeError):
    """A built-in configuration a project activated cannot be stored: its body does not resolve in
    the environment the upgrade runs in. The upgrade stops rather than drop the project's choice."""


ENV_NAME = 'env'  # the VLM endpoint built from OP_VLM_URL / OP_VLM_MODEL
ENV_REVISION = 1  # the one revision the upgrade stores for it


@dataclass(frozen=True)
class ProjectSrc:
    slug: str
    display_name: str
    description: str
    status: str  # 0.7.0 status; ``deleted`` never reaches here (a retired slug, not a project)
    pre_delete_status: str | None
    revision: int
    created_at: dt.datetime
    updated_at: dt.datetime
    origin: Any | None
    indexes: dict[str, str]  # role -> physical name, only ``images`` and ``items``
    model_prefix: str
    mlflow_experiment: str
    index_prefix: str
    storage_prefix: str
    paused_at: dt.datetime | None
    legacy_indexes: dict[str, str]  # control role -> name, for the sources to delete


@dataclass(frozen=True)
class RetiredSlug:
    slug: str
    created_at: dt.datetime
    retired_at: dt.datetime


@dataclass(frozen=True)
class ClassSrc:
    class_id: int
    name: str
    class_group: str | None
    deprecated: bool
    merged_into: int | None
    hotkey_letter: str | None
    notes: str
    added_at: dt.datetime


@dataclass(frozen=True)
class RegistrySrc:
    classes: tuple[ClassSrc, ...]
    next_class_id: int
    revision: int
    migrated_from: str  # ``class_registry_file`` or ``opensearch_registry_doc``
    migrated_at: dt.datetime


@dataclass(frozen=True)
class ConfigRevisionSrc:
    revision: int
    body: Any
    created_at: dt.datetime
    cloned_from: str | None


@dataclass(frozen=True)
class ConfigDocSrc:
    project_slug: str | None  # ``None``: deployment-wide
    kind: str
    name: str
    description: str
    revisions: tuple[ConfigRevisionSrc, ...]  # ascending; the last is the current one
    created_at: dt.datetime
    updated_at: dt.datetime


@dataclass(frozen=True)
class ActivationSrc:
    """One activation event of an axis; the last event of an axis is the current activation
    (an event to ``None`` leaves the axis off)."""

    axis: str
    from_name: str | None
    from_revision: int | None
    to_name: str | None  # ``None``: the axis was turned off
    to_revision: int | None  # ``None`` with ``to_name`` ``None``: off; with a name, a built-in
    kind: str
    at: dt.datetime


@dataclass(frozen=True)
class SettingsSrc:
    defaults: dict[str, Any]
    policies: dict[str, dict[str, Any]]  # ingest_policy, vlm_policy
    updated_at: dt.datetime | None
    updated_by: str | None


@dataclass(frozen=True)
class KeymapSrc:
    overrides: dict[str, Any]
    revision: int
    updated_at: dt.datetime | None


@dataclass(frozen=True)
class ProbeSrc:
    name: str
    revision: int
    probed_at: dt.datetime
    ok: bool
    result: Any


@dataclass(frozen=True)
class ViewStateSrc:
    projection_version: int
    scope: str
    cluster_id: int | None
    n_points: int
    n_components: int
    metric: str
    fitted_at: dt.datetime
    reducer_b64: str | None  # the clustering reducer to copy to storage, when one exists


@dataclass(frozen=True)
class HoldoutSrc:
    id: uuid.UUID
    is_current: bool
    source: str
    item_count: int
    frozen_at: dt.datetime
    frozen_by: str
    snapshot_key: str | None
    record: dict[str, Any]


@dataclass(frozen=True)
class LabelOpSrc:
    id: uuid.UUID
    legacy_id: str
    kind: str
    status: str
    params: dict[str, Any]
    created_at: dt.datetime
    created_by: str
    undone_at: dt.datetime | None
    members: tuple[str, ...]


@dataclass(frozen=True)
class ProjectData:
    """Everything one project owns in the bundle."""

    project: ProjectSrc
    registry: RegistrySrc | None
    configs: tuple[ConfigDocSrc, ...] = ()
    activations: tuple[ActivationSrc, ...] = ()
    settings: SettingsSrc | None = None
    keymap: KeymapSrc | None = None
    cluster_refresh: dict[str, Any] | None = None
    markers: dict[str, Any] = field(default_factory=dict)  # project_state key -> body
    view_state: ViewStateSrc | None = None
    holdouts: tuple[HoldoutSrc, ...] = ()
    label_ops: tuple[LabelOpSrc, ...] = ()


@dataclass(frozen=True)
class VlmEndpointSrc:
    doc: ConfigDocSrc
    probes: tuple[ProbeSrc, ...]


@dataclass(frozen=True)
class GlobalData:
    endpoints: tuple[VlmEndpointSrc, ...] = ()
    env_probe: ProbeSrc | None = None  # the last probe of the ``env`` built-in, if any
    local_vlm_desired: dict[str, Any] | None = None
    projects_revision: int = 0


@dataclass(frozen=True)
class JobRow:
    """A ``jobs`` row made from a 0.6.0 job directory or file set."""

    id: uuid.UUID
    project_slug: str | None
    kind: str
    status: str
    queue: str
    params: dict[str, Any]
    progress: dict[str, Any]
    result: Any | None
    error_code: str | None
    error_message: str | None
    created_at: dt.datetime
    started_at: dt.datetime | None
    finished_at: dt.datetime | None
    requested_by: str = 'upgrade_from_060'


@dataclass(frozen=True)
class IngestSourceSrc:
    id: uuid.UUID
    project_slug: str
    name: str
    kind: str
    config: dict[str, Any]
    schedule: dict[str, Any]
    enabled: bool
    created_at: dt.datetime
    updated_at: dt.datetime
    scan_state: dict[str, Any] | None
    last_poll_at: dt.datetime | None
    queue_image: bytes | None  # the SQLite image of ``queue.sqlite``; read by the ledger COPY


@dataclass(frozen=True)
class ImportSrc:
    id: uuid.UUID
    legacy_id: str
    project_slug: str
    job: JobRow
    format: str
    source_key: str
    mapping: Any
    options: Any
    scan: Any | None
    report: Any | None
    created_at: dt.datetime
    undone_at: dt.datetime | None
    undone: bool


@dataclass(frozen=True)
class TrainingRunSrc:
    id: uuid.UUID
    legacy_id: str
    project_slug: str
    job: JobRow
    spec: dict[str, Any]
    class_names: tuple[str, ...]
    family: str
    base_model: str
    artifact_prefix: str
    manifest: Any | None
    mlflow_run_id: str | None
    created_at: dt.datetime
    finished_at: dt.datetime | None


@dataclass(frozen=True)
class PromotionSrc:
    model_id: uuid.UUID
    promotion_id: uuid.UUID
    project_slug: str
    triton_name: str
    triton_version: int
    version: int
    labels: tuple[str, ...]
    family: str
    class_remap: Any | None
    training_run_legacy_id: str | None
    promoted_at: dt.datetime
    shared: bool
    artifact_key: str


@dataclass(frozen=True)
class ExportSrc:
    id: uuid.UUID
    project_slug: str
    kind: str
    class_names: tuple[str, ...]
    manifest_sha256: str
    artifact_prefix: str
    split_seed: int
    created_at: dt.datetime


@dataclass(frozen=True)
class CombineSrc:
    job: JobRow
    target_slug: str
    holdout_mode: str
    class_mapping: Any
    source_slugs: tuple[str, ...]
    created_at: dt.datetime


@dataclass
class LoadReport:
    """Counts of what the load wrote, in the vocabulary of the installer's report."""

    counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + n
