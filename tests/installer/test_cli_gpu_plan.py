"""``openprocessor gpu plan`` explains and checks the PE encoder placement, driven through the
installer shim harness (nvidia-smi and docker are test doubles)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import test_cli_project
from test_cli_project import cli
from test_cli_vlm import set_env


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed

inst = test_cli_project.inst

TWO_A6000 = '0, NVIDIA RTX A6000, 49140, 100, 8.6\n2, NVIDIA RTX A6000, 49140, 100, 8.6\n'


def test_gpu_plan_lists_every_pe_card_and_the_primary(shimmed: Shimmed, inst: Path) -> None:
    shimmed.gpus(TWO_A6000)
    set_env(inst, TRITON_GPU_ID='0,2')
    done = cli(shimmed, inst, 'gpu', 'plan')
    assert done.returncode == 0, done.stderr
    assert 'GPU 0 role=primary' in done.stdout
    assert 'GPU 2 role=pe' in done.stdout


def test_gpu_plan_refuses_a_gpu_the_host_does_not_have(shimmed: Shimmed, inst: Path) -> None:
    shimmed.gpus(TWO_A6000)
    set_env(inst, TRITON_GPU_ID='0,1')
    done = cli(shimmed, inst, 'gpu', 'plan')
    assert done.returncode == 1
    assert 'GPU 1 is not present' in done.stdout + done.stderr


def test_gpu_plan_refuses_mixed_compute_capability(shimmed: Shimmed, inst: Path) -> None:
    shimmed.gpus('0, NVIDIA RTX A6000, 49140, 100, 8.6\n1, NVIDIA RTX 4090, 24564, 100, 8.9\n')
    set_env(inst, TRITON_GPU_ID='0,1')
    done = cli(shimmed, inst, 'gpu', 'plan')
    assert done.returncode == 1
    assert 'compute capabilities' in done.stdout + done.stderr
