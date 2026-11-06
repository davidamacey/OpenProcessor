"""Every control-plane table as an ORM model, one module per concern.

Importing this package registers all tables on ``Base.metadata`` and imports nothing outside
SQLAlchemy and ``src.domain``.
"""

from src.db.models.classes import ClassRegistry, ProjectClass, SearchProjectionTask
from src.db.models.config import (
    ConfigActivation,
    ConfigActivationEvent,
    ConfigDocument,
    ConfigRevision,
    ConfigRuntimeReport,
    ProjectionState,
    ProjectKeymap,
    ProjectSettings,
    ProjectState,
    VlmEndpointProbe,
)
from src.db.models.datasets import (
    CombineRun,
    CombineSource,
    DatasetImport,
    DatasetImportImage,
    DatasetUpload,
    HoldoutFreeze,
    IngestSource,
    IngestSourceFile,
    LabelOperation,
    LabelOperationMember,
)
from src.db.models.jobs import (
    Event,
    GpuDevice,
    GpuReservation,
    Job,
    JobAttempt,
    JobItem,
    StreamWorkerClaim,
    WorkerCapability,
)
from src.db.models.ops import UpgradeRun
from src.db.models.projects import (
    DeploymentCounter,
    DeploymentSetting,
    Project,
    ProjectIndex,
    ProjectSlug,
)
from src.db.models.training import (
    BakeoffRun,
    DatasetExport,
    Model,
    ModelEngine,
    ModelShare,
    Promotion,
    TrainingRun,
)


__all__ = [
    'BakeoffRun',
    'ClassRegistry',
    'CombineRun',
    'CombineSource',
    'ConfigActivation',
    'ConfigActivationEvent',
    'ConfigDocument',
    'ConfigRevision',
    'ConfigRuntimeReport',
    'DatasetExport',
    'DatasetImport',
    'DatasetImportImage',
    'DatasetUpload',
    'DeploymentCounter',
    'DeploymentSetting',
    'Event',
    'GpuDevice',
    'GpuReservation',
    'HoldoutFreeze',
    'IngestSource',
    'IngestSourceFile',
    'Job',
    'JobAttempt',
    'JobItem',
    'LabelOperation',
    'LabelOperationMember',
    'Model',
    'ModelEngine',
    'ModelShare',
    'Project',
    'ProjectClass',
    'ProjectIndex',
    'ProjectKeymap',
    'ProjectSettings',
    'ProjectSlug',
    'ProjectState',
    'ProjectionState',
    'Promotion',
    'SearchProjectionTask',
    'StreamWorkerClaim',
    'TrainingRun',
    'UpgradeRun',
    'VlmEndpointProbe',
    'WorkerCapability',
]
