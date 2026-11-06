"""The GPU worker's card inventory: every NVML-visible card is recorded by UUID.

The NVML calls and the database writes are faked here; the real writer lands with the job
runtime. The parent process must never import ``pynvml`` (CUDA is initialised in the child).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tasks.runtime import gpu_app


REPO_ROOT = Path(__file__).resolve().parents[2]
UUIDS = ('GPU-aaaa', 'GPU-bbbb')


class _NvmlError(RuntimeError):
    pass


def _unsupported(_handle: int) -> tuple[int, int]:
    raise _NvmlError('not supported')


def fake_nvml(uuids: tuple[str, ...] = UUIDS) -> SimpleNamespace:
    """The subset of ``pynvml`` the inventory calls, for the given cards."""
    return SimpleNamespace(
        NVMLError=_NvmlError,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: len(uuids),
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUUID=lambda handle: uuids[handle],
        nvmlDeviceGetName=lambda handle: f'Card {handle}',
        nvmlDeviceGetCudaComputeCapability=lambda _h: (8, 6),
        nvmlDeviceGetMemoryInfo=lambda _h: SimpleNamespace(total=48 * 1024 * 1024 * 1024),
        nvmlDeviceGetNumaNodeId=lambda _h: 0,
        nvmlDeviceGetMaxPcieLinkGeneration=lambda _h: 4,
        nvmlDeviceGetMaxPcieLinkWidth=lambda _h: 16,
        nvmlDeviceGetJpgUtilization=_unsupported,
        nvmlSystemGetDriverVersion=lambda: '580.1',
        nvmlSystemGetCudaDriverVersion=lambda: 13000,
    )


class FakeWriter:
    def __init__(self) -> None:
        self.devices: dict[str, gpu_app.GpuDevice] = {}
        self.touched: list[tuple[str, ...]] = []
        self.capabilities: list[dict] = []

    def upsert_devices(self, devices: list[gpu_app.GpuDevice]) -> None:
        self.devices.update({d.gpu_uuid: d for d in devices})

    def touch_devices(self, uuids: list[str]) -> None:
        self.touched.append(tuple(uuids))

    def record_capabilities(self, capabilities: dict) -> None:
        self.capabilities.append(capabilities)


@pytest.fixture
def writer(monkeypatch: pytest.MonkeyPatch) -> FakeWriter:
    fake = FakeWriter()
    monkeypatch.setattr(gpu_app, '_inventory_writer', fake)
    monkeypatch.setattr(gpu_app, '_inventory', None)
    return fake


def test_a_worker_with_two_visible_cards_writes_two_rows_keyed_by_uuid(writer: FakeWriter) -> None:
    gpu_app.start_inventory(device_uuid=None, nvml=fake_nvml())
    assert set(writer.devices) == set(UUIDS)
    row = writer.devices['GPU-aaaa']
    assert (row.name, row.compute_capability, row.memory_total_mib) == ('Card 0', '8.6', 49152)
    assert (row.pcie_gen, row.pcie_width, row.hw_jpeg_decoder) == (4, 16, False)
    assert (row.driver_version, row.cuda_version) == ('580.1', '13.0')
    assert writer.capabilities == [{'gpu_uuids': list(UUIDS), 'device_uuid': None}]


def test_a_device_scoped_worker_names_its_card_in_the_capabilities(writer: FakeWriter) -> None:
    gpu_app.start_inventory(device_uuid='GPU-bbbb', nvml=fake_nvml())
    assert writer.capabilities == [{'gpu_uuids': list(UUIDS), 'device_uuid': 'GPU-bbbb'}]


def test_a_device_uuid_the_container_cannot_see_is_refused(writer: FakeWriter) -> None:
    with pytest.raises(RuntimeError, match='GPU-zzzz'):
        gpu_app.start_inventory(device_uuid='GPU-zzzz', nvml=fake_nvml())
    assert writer.devices == {}


def test_a_heartbeat_refreshes_every_card_so_a_long_lived_worker_keeps_its_rows(
    writer: FakeWriter,
) -> None:
    gpu_app.start_inventory(device_uuid=None, nvml=fake_nvml())
    gpu_app.refresh_inventory()
    gpu_app.refresh_inventory()
    assert writer.touched == [UUIDS, UUIDS]


def test_a_container_with_no_cards_records_nothing(writer: FakeWriter) -> None:
    gpu_app.start_inventory(device_uuid=None, nvml=fake_nvml(()))
    assert writer.devices == {}
    assert writer.capabilities == []
    gpu_app.refresh_inventory()
    assert writer.touched == []


def test_cards_without_a_registered_writer_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_app, '_inventory_writer', None)
    monkeypatch.setattr(gpu_app, '_inventory', None)
    with pytest.raises(RuntimeError, match='writer'):
        gpu_app.start_inventory(device_uuid=None, nvml=fake_nvml())


def test_make_app_keeps_pynvml_out_of_the_parent() -> None:
    probe = (
        "import sys; sys.path.insert(0, '.')\n"
        'from src.tasks.runtime.gpu_app import make_app\n'
        "app = make_app([], device_uuid='GPU-aaaa')\n"
        "print('pynvml' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, '-I', '-c', probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.stdout.strip() == 'False'
