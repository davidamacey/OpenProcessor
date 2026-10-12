"""PE image encoder placement across GPUs.

``pe_image_encoder`` carries no ``gpus`` list, so Triton starts one instance on every GPU the
container sees and ``TRITON_GPU_ID`` in ``.env`` decides how many cards encode in parallel.
Every other model stays on the first visible card, which the NVIDIA runtime numbers in host
order, so it is the lowest listed id.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
PE = 'pe_image_encoder'
MEM_KEY = 'GPU_MIB_PER_INSTANCE'
# setup-openprocessor.sh budgets whole gigabytes of VRAM for the encoder on a card.
INSTALLER_PE_BUDGET_MIB = 2 * 1024


def _config(model: str) -> str:
    return (MODELS / model / 'config.pbtxt').read_text()


def _gpu_models() -> list[str]:
    return [
        cfg.parent.name
        for cfg in sorted(MODELS.glob('*/config.pbtxt'))
        if 'KIND_GPU' in cfg.read_text()
    ]


def test_pe_encoder_is_not_pinned_to_a_gpu() -> None:
    text = _config(PE)
    block = re.search(r'^instance_group\s*\[(.*?)^\]', text, re.S | re.M)
    assert block, 'no instance_group'
    assert 'KIND_GPU' in block.group(1)
    assert not re.search(r'^\s*gpus:', block.group(1), re.M), (
        'a gpus list pins the encoder to those cards regardless of TRITON_GPU_ID'
    )
    assert re.findall(r'^\s*count:\s*(\d+)', block.group(1), re.M) == ['1']


def test_every_other_gpu_model_stays_on_the_primary_card() -> None:
    others = [m for m in _gpu_models() if m != PE]
    assert others
    for model in others:
        pins = re.findall(r'^\s*gpus:\s*\[\s*([\d,\s]*)\]', _config(model), re.M)
        assert pins, f'{model}: no gpus pin, it would be copied to every visible card'
        assert all(p.strip() == '0' for p in pins), f'{model}: pinned to {pins}, expected [ 0 ]'


def test_pe_encoder_records_its_memory_per_instance() -> None:
    match = re.search(rf'key:\s*"{MEM_KEY}"\s*value:\s*{{\s*string_value:\s*"(\d+)"', _config(PE))
    assert match, f'{PE}: missing {MEM_KEY}'
    mib = int(match.group(1))
    assert 0 < mib <= INSTALLER_PE_BUDGET_MIB, (
        f'{mib} MiB per instance exceeds the {INSTALLER_PE_BUDGET_MIB} MiB the installer plans per card'
    )


def test_triton_service_passes_the_gpu_list_through_unsplit() -> None:
    compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
    devices = compose['services']['triton-server']['deploy']['resources']['reservations']['devices']
    # One string: the NVIDIA runtime accepts "0,2" in a single device id.
    assert devices[0]['device_ids'] == ['${TRITON_GPU_ID:-0}']


def test_env_template_documents_the_gpu_list() -> None:
    text = (ROOT / 'env.template').read_text()
    block = text.split('TRITON_GPU_ID=0', 1)[0].rsplit('\n\n', 1)[-1]
    assert 'comma' in block.lower()
    assert 'PE' in block


def test_triton_numbers_its_gpus_by_pci_bus() -> None:
    # CUDA's default order is fastest first, so an RTX A6000 would come before a lower-numbered
    # RTX 3080 Ti and the "gpus: [ 0 ]" models would land on a card other than the lowest id.
    compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
    env = compose['services']['triton-server'].get('environment') or []
    assert 'CUDA_DEVICE_ORDER=PCI_BUS_ID' in env
