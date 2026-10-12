"""gpu_pe_placement: what `openprocessor gpu plan` says about TRITON_GPU_ID lists.

The table rows are ``index|name|total_mib|used_mib|compute_cap``.
"""

from __future__ import annotations

import subprocess

from installer_harness import REPO_ROOT


GPU_SH = REPO_ROOT / 'scripts' / 'lib' / 'gpu.sh'
A6000 = 'NVIDIA RTX A6000'
TI = 'NVIDIA GeForce RTX 3080 Ti'
HOST = [
    (0, A6000, 49140, 16000, '8.6'),
    (1, TI, 12288, 1000, '8.6'),
    (2, A6000, 49140, 13000, '8.6'),
]


def placement(
    ids: str,
    rows: list[tuple[int, str, int, int, str]] = HOST,
    pe_mib: int = 1536,
    running: int = 0,
    primary_mib: int = 0,
) -> tuple[int, dict[str, list[str]]]:
    table = '\n'.join('|'.join(str(c) for c in r) for r in rows)
    done = subprocess.run(
        [
            'bash',
            '-c',
            f'source "{GPU_SH}"; gpu_pe_placement "$1" "$2" "$3" "$4" "$5"',
            '_',
            ids,
            str(pe_mib),
            table,
            str(running),
            str(primary_mib),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    out: dict[str, list[str]] = {}
    for line in done.stdout.splitlines():
        key, _, value = line.partition('=')
        out.setdefault(key, []).append(value)
    return done.returncode, out


def test_one_gpu_is_the_primary_card() -> None:
    rc, out = placement('1')
    assert rc == 0
    assert out['primary'] == ['1']
    assert out['pe_gpus'] == ['1']
    assert out['gpu'][0].startswith('1 role=primary')


def test_two_a6000s_run_the_encoder_on_both_and_the_lowest_is_primary() -> None:
    rc, out = placement('2,0')
    assert rc == 0
    assert out['primary'] == ['0']
    assert out['pe_gpus'] == ['0,2']
    assert [g.split()[1] for g in out['gpu']] == ['role=primary', 'role=pe']
    assert 'warn' not in out


def test_three_gpus_warn_that_the_cards_are_different_models() -> None:
    rc, out = placement('0,1,2')
    assert rc == 0
    assert out['pe_gpus'] == ['0,1,2']
    assert any('different models' in w for w in out['warn'])


def test_a_missing_gpu_refuses_and_names_what_the_host_has() -> None:
    rc, out = placement('0,3')
    assert rc == 1
    assert any('GPU 3 is not present' in p and '0,1,2' in p for p in out['problem'])


def test_a_repeated_or_malformed_id_refuses() -> None:
    assert placement('0,0')[0] == 1
    rc, out = placement('0,x')
    assert rc == 1
    assert any("'x'" in p for p in out['problem'])
    assert placement('')[0] == 1


def test_cards_of_different_compute_capability_refuse() -> None:
    rows = [(0, A6000, 49140, 0, '8.6'), (1, 'NVIDIA GeForce RTX 4090', 24564, 0, '8.9')]
    rc, out = placement('0,1', rows)
    assert rc == 1
    assert any('compute capabilities' in p for p in out['problem'])


def test_a_card_without_room_for_an_instance_refuses_unless_triton_runs() -> None:
    rows = [(0, A6000, 49140, 0, '8.6'), (1, TI, 12288, 11500, '8.6')]
    rc, out = placement('0,1', rows)
    assert rc == 1
    assert any('GPU 1 has 788 MiB free' in p for p in out['problem'])
    rc, out = placement('0,1', rows, running=1)
    assert rc == 0
    assert any('Triton is running' in w for w in out['warn'])


def test_a_12_gb_card_is_refused_as_the_primary_but_serves_as_a_secondary() -> None:
    rows = [(0, A6000, 49140, 0, '8.6'), (1, TI, 12288, 2500, '8.6')]
    # standard profile: the primary needs 12 GB of free memory for every model
    rc, out = placement(
        '1,2', [(1, TI, 12288, 2500, '8.6'), (2, A6000, 49140, 0, '8.6')], primary_mib=12288
    )
    assert rc == 1
    assert any('GPU 1 has' in p and 'primary card' in p for p in out['problem'])
    rc, out = placement('0,1', rows, primary_mib=12288)
    assert rc == 0
    assert [g.split()[1] for g in out['gpu']] == ['role=primary', 'role=pe']
