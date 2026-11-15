"""``scripts/bench/frame_engine_hold.sh`` and ``frame_engine_faults.sh``, driven with ``--dry-run``
(they print every command they would run and start nothing) and a stand-in ``nvidia-smi`` on the
path: the lock, the bounded commands, the device guard, the load gate and the rows of each hold."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / 'scripts' / 'bench'
HOLD = BENCH / 'frame_engine_hold.sh'
FAULTS = BENCH / 'frame_engine_faults.sh'
BUS = '00000000:C1:00.0'
LABEL = re.compile(r'--label (\S+)')


@pytest.fixture
def bed(tmp_path: Path) -> dict[str, str]:
    """The environment of a dry run: a card list on the path and a load average file."""
    smi = tmp_path / 'bin' / 'nvidia-smi'
    smi.parent.mkdir()
    smi.write_text(
        '#!/bin/bash\n'
        'case "$*" in\n'
        f'  *query-gpu=index,pci.bus_id*) printf "0, 00000000:01:00.0\\n2, {BUS}\\n" ;;\n'
        '  *query-compute-apps*) [[ -n "${FAKE_APPS:-}" ]] && printf "%s\\n" "$FAKE_APPS" ;;\n'
        'esac\n'
    )
    smi.chmod(0o755)
    load = tmp_path / 'loadavg'
    load.write_text('1.50 1.00 1.00 1/100 1\n')
    return {**os.environ, 'PATH': f'{smi.parent}:{os.environ["PATH"]}', 'FE_LOADAVG': str(load)}


def hold(
    rows: str, env: dict[str, str], tmp_path: Path, *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(HOLD), '--dry-run', '--hold-file', str(tmp_path / 'hold'), '--rows', rows,
            '--lists', '/lists', '--out', str(tmp_path / 'out.jsonl'), '--gpu-index', '2',
            '--gpu-bus-id', BUS, '--capture-dir', '/cap', *extra,
        ],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip


def lines(out: str, tag: str) -> list[str]:
    return [ln for ln in out.splitlines() if ln.startswith(f'{tag}: ')]


def labels(out: str, kind: str) -> list[str]:
    """The labels of the loadgen rows of one kind (``engine`` or ``z``), in the order they run."""
    return [
        m.group(1)
        for ln in lines(out, 'RUN')
        if f'op-fe-loadgen {kind} ' in ln and (m := LABEL.search(ln))
    ]


def test_the_hold_takes_the_file_with_flock_o_and_a_bounded_wait(
    bed: dict[str, str], tmp_path: Path
) -> None:
    out = hold('h3', bed, tmp_path).stdout
    (lock,) = lines(out, 'LOCK')
    assert f'flock -o -w 60 -E 75 {tmp_path}/hold {HOLD} --locked' in lock
    assert lines(out, 'ENV') == ['ENV: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2']


def test_every_command_runs_under_timeout_kill_after_innermost(
    bed: dict[str, str], tmp_path: Path
) -> None:
    out = hold('h2', bed, tmp_path).stdout
    runs = lines(out, 'RUN')
    assert len(runs) > 100
    assert all(ln.startswith('RUN: timeout --kill-after=15s ') for ln in runs)
    assert all(re.match(r'RUN: timeout --kill-after=15s \d+ /', ln) for ln in runs), (
        'seconds then the program'
    )


def test_the_plans_come_from_the_parity_tool_for_every_plan_and_model_set(
    bed: dict[str, str], tmp_path: Path
) -> None:
    out = hold('h3', bed, tmp_path).stdout
    made = [ln for ln in lines(out, 'RUN') if 'gpu_prep_parity.py plan' in ln]
    assert len(made) == 6
    assert {
        (m.group(1), m.group(2))
        for ln in made
        if (m := re.search(r'--plan (\w+) --models (\w+)', ln))
    } == {(p, m) for p in ('frame', 'ingest') for m in ('real', 'null', 'capture')}


def test_arms_alternate_in_a_b_b_a_order_within_a_row(bed: dict[str, str], tmp_path: Path) -> None:
    out = hold('h2', bed, tmp_path, '--sets', 'coco').stdout
    arms = [
        lab.split(':')[1]
        for lab in labels(out, 'engine')
        if lab.startswith('h2:') and ':coco:frame:real:32:' in lab
    ]
    assert arms == ['python', 'native', 'native', 'python']
    runs = [lab for lab in labels(out, 'engine') if ':coco:ingest:real:8:' in lab]
    assert [r.rsplit(':', 1)[1] for r in runs] == ['r1', 'r1', 'r2', 'r2']


def test_h2_has_the_in_flight_ladder_z_rows_and_the_input_only_row(
    bed: dict[str, str], tmp_path: Path
) -> None:
    out = hold('h2', bed, tmp_path, '--sets', 'coco hires').stdout
    engine = labels(out, 'engine')
    ladder = {int(lab.split(':')[5]) for lab in engine if ':real:' in lab}
    assert ladder == {1, 8, 32, 64, 128}
    assert len([lab for lab in engine if ':coco:ingest:real:' in lab]) == 5 * 2 * 2
    assert sorted(lab for lab in labels(out, 'z') if ':coco:' in lab) == sorted(
        f'h2:z:coco:{plan}:real:{n}' for plan in ('frame', 'ingest') for n in (32, 64)
    )
    assert [lab for lab in engine if ':null:' in lab] == [
        'h2:native:coco:frame:null:32:r1',
        'h2:native:hires:frame:null:32:r1',
    ]
    ingest_z = [ln for ln in lines(out, 'RUN') if 'op-fe-loadgen z ' in ln and 'ingest_real' in ln]
    assert all(
        '--crop-counts ' in ln and ('/coco.json' in ln or '/hires.json' in ln) for ln in ingest_z
    )


def test_h3_runs_the_null_models_on_coco_and_the_20_mp_set_only(
    bed: dict[str, str], tmp_path: Path
) -> None:
    out = hold('h3', bed, tmp_path).stdout
    engine = labels(out, 'engine')
    assert {lab.split(':')[2] for lab in engine} == {'coco', 'hires'}
    assert {lab.split(':')[4] for lab in engine} == {'null'}
    assert {int(lab.split(':')[5]) for lab in engine} == {32, 64, 128}
    assert len(labels(out, 'z')) == 2 * 2 * 3, 'a Z0 row per set, plan and in-flight count'
    assert any('_null.json' in c for c in lines(out, 'COUNTS')), (
        'null counts: the detector fixes them'
    )


def test_every_row_names_the_card_the_stats_and_the_engines_cgroup(
    bed: dict[str, str], tmp_path: Path
) -> None:
    out = hold('h3', bed, tmp_path).stdout
    row = next(
        ln for ln in lines(out, 'RUN') if 'op-fe-loadgen engine ' in ln and 'h3:native:' in ln
    )
    assert f'--gpu-bus-id {BUS}' in row
    assert '--stats-url 127.0.0.1:4620' in row
    assert '--cpu-stat /sys/fs/cgroup/system.slice/docker-op-n0-engine-native.scope/cpu.stat' in row
    z = next(ln for ln in lines(out, 'RUN') if 'op-fe-loadgen z ' in ln)
    assert 'docker-op-n0-model-server.scope/cpu.stat' in z, (
        'in z mode the model server is the process measured'
    )
    assert '--cuda-device 0 --server-device-id 0' in z


def test_the_smoke_hold_checks_the_region_the_canvases_the_idle_ping_and_the_next_request(
    bed: dict[str, str], tmp_path: Path
) -> None:
    out = hold('s0', bed, tmp_path, '--sets', 'coco').stdout
    kinds = [ln.split(': ', 1)[0] for ln in out.splitlines() if ': ' in ln]
    text = '\n'.join(lines(out, 'RUN'))
    assert 'op-fe-loadgen regions --url 127.0.0.1:4620 --prefix op-fe-' in text
    assert lines(out, 'ARM') == ['ARM: /cap/ARM python_s0', 'ARM: /cap/ARM native_s0']
    assert len(re.findall(r'--models capture --concurrency 1', text)) == 2
    assert 'compare-inputs /cap/python_s0 /cap/native_s0' in text
    (kept,) = [ln for ln in text.splitlines() if 's0_inputs.json' in ln]
    assert kept.endswith('s0_inputs.json'), 'the comparison is kept in a file, not lost on stdout'
    assert lines(out, 'IDLE') == ['IDLE: 65']
    assert kinds.index('IDLE') > max(i for i, k in enumerate(kinds) if k == 'ARM')
    assert (
        text.rstrip()
        .splitlines()[-1]
        .endswith('--out ' + f'{tmp_path}/out.jsonl.d/s0_after_idle.npz')
    )


def test_the_parity_hold_measures_the_floor_with_the_engine_against_itself(
    bed: dict[str, str], tmp_path: Path
) -> None:
    out = hold('h1', bed, tmp_path, '--sets', 'coco').stdout
    text = '\n'.join(lines(out, 'RUN'))
    compares = [ln for ln in text.splitlines() if ' compare ' in ln]
    assert len(compares) == 4, 'plans F and C at 1 and 32 in flight'
    assert all('--floor-a' in c and '--floor-b' in c and '_python2.npz' in c for c in compares)
    first_captures = [
        ln
        for ln in text.splitlines()
        if 'capture --url 127.0.0.1:4622' in ln and '--limit 200' in ln
    ]
    assert len(first_captures) == 2 * 2 * 2, 'python and python again, four combinations'
    assert 'compare-inputs /cap/python_coco /cap/native_coco' in text


@pytest.mark.parametrize(
    ('script', 'options'),
    [
        (HOLD, ['--rows', 'h3', '--lists', '/l']),
        (FAULTS, ['--mode', 'kill', '--list', '/l/a.txt', '--plan-file', '/p.json']),
    ],
)
def test_the_wrong_card_stops_everything_before_any_command(
    script: Path, options: list[str], bed: dict[str, str], tmp_path: Path
) -> None:
    done = subprocess.run(
        [str(script), '--dry-run', '--hold-file', str(tmp_path / 'h'), *options,
         '--out', str(tmp_path / 'o'), '--gpu-index', '2', '--gpu-bus-id', '00000000:99:00.0'],
        env=bed, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert done.returncode == 1
    assert 'not the expected 00000000:99:00.0: nothing runs' in done.stderr
    assert lines(done.stdout, 'RUN') == []


def test_a_card_that_is_not_at_the_index_is_refused(bed: dict[str, str], tmp_path: Path) -> None:
    done = subprocess.run(
        [str(HOLD), '--dry-run', '--hold-file', str(tmp_path / 'h'), '--rows', 'h3', '--lists', '/l',
         '--out', str(tmp_path / 'o'), '--gpu-index', '7', '--gpu-bus-id', BUS],
        env=bed, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert done.returncode == 1
    assert 'no card at index 7' in done.stderr


def test_a_process_of_another_tenant_on_the_card_stops_the_hold(
    bed: dict[str, str], tmp_path: Path
) -> None:
    bed['FAKE_APPS'] = f'4242, {BUS}'
    done = hold('h3', bed, tmp_path)
    assert done.returncode == 1
    assert 'outside this test bed are on card 2: 4242' in done.stderr


def test_a_loaded_host_stops_the_hold_after_the_bounded_wait(
    bed: dict[str, str], tmp_path: Path
) -> None:
    Path(bed['FE_LOADAVG']).write_text('23.40 20.00 18.00 3/900 1\n')
    done = hold('h3', bed, tmp_path, '--load-wait-s', '0')
    assert done.returncode == 1
    assert 'the 1-minute load 23.40 is not under 16' in done.stderr
    assert lines(done.stdout, 'RUN') == []


def test_the_fault_hold_kills_the_engine_by_its_exact_name_each_time(
    bed: dict[str, str], tmp_path: Path
) -> None:
    done = subprocess.run(
        [str(FAULTS), '--dry-run', '--hold-file', str(tmp_path / 'h'), '--mode', 'kill', '--kills', '3',
         '--list', '/l/a.txt', '--plan-file', '/p.json', '--out', str(tmp_path / 'f.jsonl'),
         '--gpu-index', '2', '--gpu-bus-id', BUS],
        env=bed, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    runs = lines(done.stdout, 'RUN')
    kills = [ln for ln in runs if 'docker kill' in ln]
    assert kills == ['RUN: timeout --kill-after=15s 30 docker kill -s KILL op-n0-engine-native'] * 3
    loops = [ln for ln in runs if 'op-fe-loadgen engine' in ln]
    assert len(loops) == 3
    assert all('--in-flight 32' in ln and '--allow-failures' in ln for ln in loops)
    assert lines(done.stdout, 'LOCK')[0].startswith('LOCK: flock -o -w 60 -E 75 ')


def test_the_server_restart_hold_restarts_the_server_and_nothing_else(
    bed: dict[str, str], tmp_path: Path
) -> None:
    done = subprocess.run(
        [str(FAULTS), '--dry-run', '--hold-file', str(tmp_path / 'h'), '--mode', 'server-restart',
         '--list', '/l/a.txt', '--plan-file', '/p.json', '--out', str(tmp_path / 'f.jsonl'),
         '--gpu-index', '2', '--gpu-bus-id', BUS],
        env=bed, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert lines(done.stdout, 'RUN') == [
        'RUN: timeout --kill-after=15s 120 docker restart --time 30 op-n0-model-server'
    ]


def test_an_unknown_option_and_a_missing_one_are_refused(bed: dict[str, str]) -> None:
    unknown = subprocess.run(
        [str(HOLD), '--nope'], env=bed, capture_output=True, text=True, timeout=30, check=False
    )
    assert unknown.returncode == 1
    assert 'unknown argument: --nope' in unknown.stderr
    missing = subprocess.run(
        [str(HOLD), '--rows', 'h2'],
        env=bed,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert missing.returncode == 1
    assert '--HOLD-FILE is required' in missing.stderr


@pytest.mark.skipif(shutil.which('shellcheck') is None, reason='shellcheck is not installed')
def test_the_scripts_are_shellcheck_clean() -> None:
    done = subprocess.run(
        ['shellcheck', '--severity=style', '-x', *(str(p) for p in (HOLD, FAULTS, BENCH / 'frame_engine_bed.sh'))],
        capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout


def test_an_engines_regions_are_counted_by_its_instance_not_by_the_shared_prefix(
    tmp_path: Path,
) -> None:
    """Two engines share the model server: each holds one region, named by its instance."""
    fake = tmp_path / 'op-fe-loadgen'
    fake.write_text(
        '#!/bin/bash\nprintf "op-fe-0000-84-00-0-fe-native-77\nop-fe-0000-84-00-0-fe-python-12\n"\n'
    )
    fake.chmod(0o755)

    def count(instance: str) -> str:
        script = f'source {BENCH}/frame_engine_bed.sh; LOADGEN={fake}; engine_regions {instance}'
        done = subprocess.run(
            ['bash', '-c', script], capture_output=True, text=True, timeout=30, check=False
        )
        return done.stdout.strip()

    assert (count('fe-native'), count('fe-python'), count('fe-other')) == ('1', '1', '0')
