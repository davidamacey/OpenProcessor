"""PE encoder placement across GPUs.

The frame encoder ``pe_image_encoder`` and the crop encoder ``pe_crop_encoder`` carry no ``gpus``
list, so Triton starts one instance of each on every GPU the container sees and ``TRITON_GPU_ID``
in ``.env`` decides how many cards encode in parallel. Every other model stays on the first
visible card, which the NVIDIA runtime numbers in host order, so it is the lowest listed id.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / 'export') not in sys.path:
    sys.path.insert(0, str(ROOT / 'export'))

import pe_image_constants
import pe_image_triton_config


MODELS = ROOT / 'models'
PE_MODELS = ('pe_image_encoder', 'pe_crop_encoder')
MEM_KEY = 'GPU_MIB_PER_INSTANCE'


def _config(model: str) -> str:
    return (MODELS / model / 'config.pbtxt').read_text()


def _gpu_models() -> list[str]:
    return [
        cfg.parent.name
        for cfg in sorted(MODELS.glob('*/config.pbtxt'))
        if 'KIND_GPU' in cfg.read_text()
    ]


def _installer_pe_budget_gb(crop_size: int) -> int:
    """The whole gigabytes setup-openprocessor.sh plans for the PE encoders on a card."""
    done = subprocess.run(
        [
            'bash',
            '-c',
            f'OP_SOURCE_ONLY=1 source "{ROOT / "setup-openprocessor.sh"}"; pe_need_gb "$1"',
            '_',
            str(crop_size),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(done.stdout.strip())


def _recorded_mib(model: str) -> int:
    match = re.search(
        rf'key:\s*"{MEM_KEY}"\s*value:\s*{{\s*string_value:\s*"(\d+)"', _config(model)
    )
    assert match, f'{model}: missing {MEM_KEY}'
    return int(match.group(1))


@pytest.mark.parametrize('model', PE_MODELS)
def test_pe_encoder_is_not_pinned_to_a_gpu(model: str) -> None:
    text = _config(model)
    block = re.search(r'^instance_group\s*\[(.*?)^\]', text, re.S | re.M)
    assert block, 'no instance_group'
    assert 'KIND_GPU' in block.group(1)
    assert not re.search(r'^\s*gpus:', block.group(1), re.M), (
        'a gpus list pins the encoder to those cards regardless of TRITON_GPU_ID'
    )
    assert re.findall(r'^\s*count:\s*(\d+)', block.group(1), re.M) == ['1']


def test_every_other_gpu_model_stays_on_the_primary_card() -> None:
    others = [m for m in _gpu_models() if m not in PE_MODELS]
    assert others
    for model in others:
        pins = re.findall(r'^\s*gpus:\s*\[\s*([\d,\s]*)\]', _config(model), re.M)
        assert pins, f'{model}: no gpus pin, it would be copied to every visible card'
        assert all(p.strip() == '0' for p in pins), f'{model}: pinned to {pins}, expected [ 0 ]'


def _nearest_gb_half_up(mib: int) -> int:
    """Python's round() rounds half to even (2.5 gives 2); the installer rounds half up."""
    return (mib * 2 + 1024) // 2048


@pytest.mark.parametrize('crop_size', [224, 336])
def test_pe_encoders_record_their_memory_and_the_installer_plans_it(crop_size: int) -> None:
    recorded = {'pe_image_encoder': _recorded_mib('pe_image_encoder')}
    if crop_size == 224:
        recorded['pe_crop_encoder'] = _recorded_mib('pe_crop_encoder')
    else:
        # The committed config is the 224 one; the 336 one is what the export renders.
        text = pe_image_triton_config.render_config(
            pe_image_triton_config.PETritonConfig(model_name='pe_crop_encoder', image_size=336)
        )
        match = re.search(rf'key:\s*"{MEM_KEY}"\s*value:\s*{{\s*string_value:\s*"(\d+)"', text)
        assert match
        recorded['pe_crop_encoder'] = int(match.group(1))
    assert all(mib > 0 for mib in recorded.values()), recorded
    # The installer's gate works in whole gigabytes; `gpu plan` checks the exact MiB.
    planned_gb = _installer_pe_budget_gb(crop_size)
    assert _nearest_gb_half_up(sum(recorded.values())) == planned_gb, (
        f'{recorded} is {sum(recorded.values())} MiB per card; the installer plans {planned_gb} GB'
    )
    assert planned_gb == {224: 2, 336: 3}[crop_size]


def test_the_planning_table_is_the_same_in_the_export_the_installer_and_the_cli() -> None:
    table = pe_image_constants.PE_CROP_GPU_MIB_PER_INSTANCE
    assert set(table) == set(pe_image_constants.CROP_IMAGE_SIZES)
    for size, mib in table.items():
        installer = subprocess.run(
            [
                'bash',
                '-c',
                f'OP_SOURCE_ONLY=1 source "{ROOT / "setup-openprocessor.sh"}"; pe_crop_mib "$1"',
                '_',
                str(size),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cli = subprocess.run(
            [
                'bash',
                '-c',
                f'source "{ROOT / "scripts/lib/cli_gpu.sh"}"; _env_value() {{ echo "$PE_CROP_SIZE"; }}; _gpu_crop_floor_mib',
            ],
            check=True,
            capture_output=True,
            text=True,
            env={'PE_CROP_SIZE': str(size), 'PATH': '/usr/bin:/bin'},
        )
        assert int(installer.stdout) == mib == int(cli.stdout), size
    assert _recorded_mib('pe_image_encoder') == pe_image_constants.PE_GPU_MIB_PER_INSTANCE


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
