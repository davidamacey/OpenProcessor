"""OP_OPENSEARCH_DATA is gone (OP_DATA_DIR replaced it). A .env that still sets it must stop the
installer and the CLI: otherwise OpenSearch would start on the empty named volume while the real
index sits at the old host path."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from _setup_dry_run_helpers import configure, dry
from installer_harness import REPO_ROOT


if TYPE_CHECKING:
    from pathlib import Path

    from installer_harness import Shimmed


def _install(shimmed: Shimmed, value: str | None) -> Path:
    assert configure(shimmed).returncode == 0
    inst = (shimmed.root / 'inst').resolve()
    if value is not None:
        with (inst / '.env').open('a') as fh:
            fh.write(f'OP_OPENSEARCH_DATA={value}\n')
    return inst


def _cli(shimmed: Shimmed, inst: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(inst / 'openprocessor'), 'status'],
        check=False,
        cwd=str(shimmed.root),
        env=shimmed.env(),
        capture_output=True,
        text=True,
        timeout=60,
        start_new_session=True,
    )


def test_a_host_path_value_is_refused_with_the_op_data_dir_steps(shimmed: Shimmed) -> None:
    inst = _install(shimmed, '/srv/opensearch-data')
    done = _cli(shimmed, inst)
    assert done.returncode == 2, done.stderr
    assert 'OP_OPENSEARCH_DATA=/srv/opensearch-data' in done.stderr
    assert 'empty' in done.stderr
    assert 'OP_DATA_DIR' in done.stderr
    assert '<dir>/opensearch' in done.stderr


def test_a_volume_name_value_is_refused_with_the_volume_steps(shimmed: Shimmed) -> None:
    inst = _install(shimmed, 'my_search_volume')
    done = _cli(shimmed, inst)
    assert done.returncode == 2, done.stderr
    assert 'OP_OPENSEARCH_DATA=my_search_volume' in done.stderr
    assert 'docker-compose.local.yml' in done.stderr


def test_the_installer_refuses_it_too(shimmed: Shimmed) -> None:
    _install(shimmed, '/srv/opensearch-data')
    done = dry(shimmed)
    assert done.returncode == 2, done.stderr
    assert 'OP_OPENSEARCH_DATA' in done.stderr


def test_without_the_key_nothing_is_refused(shimmed: Shimmed) -> None:
    inst = _install(shimmed, None)
    assert 'OP_OPENSEARCH_DATA' not in _cli(shimmed, inst).stderr


def test_the_compose_file_no_longer_reads_it() -> None:
    assert 'OP_OPENSEARCH_DATA' not in (REPO_ROOT / 'docker-compose.yml').read_text()


def test_old_host_path_full_and_named_volume_empty_is_stopped_by_the_retired_key_refusal(
    shimmed: Shimmed,
) -> None:
    """The data-dir volume check looks at named volumes with data; this case has an empty volume
    and a full host path, so the refusal on the retired key is the protection."""
    inst = _install(shimmed, '/srv/opensearch-data')
    assert _cli(shimmed, inst).returncode == 2
