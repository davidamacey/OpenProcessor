"""A ``gpu_devices`` card for the reservation tests."""

from __future__ import annotations

from src.db.repos import gpu_devices


HOST = 'trainer-host-1'


def card(gpu_uuid: str, nvml_index: int, host: str = HOST) -> gpu_devices.Card:
    return gpu_devices.Card(
        gpu_uuid=gpu_uuid,
        host=host,
        nvml_index=nvml_index,
        name='RTX A6000',
        compute_capability='8.6',
        memory_total_mib=49140,
        numa_node=None,
        pcie_gen=None,
        pcie_width=None,
        hw_jpeg_decoder=False,
        driver_version='580',
        cuda_version='13.0',
    )
