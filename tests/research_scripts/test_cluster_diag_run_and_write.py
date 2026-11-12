"""run_and_write: the result file carries the command, git sha and load around the run."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from scripts.research.cluster_diag.run import run_and_write


if TYPE_CHECKING:
    from pathlib import Path


def test_result_file_wraps_the_run_output(tmp_path: Path) -> None:
    out = tmp_path / 'nested' / 'r.json'
    args = argparse.Namespace(cmd='diag', out=out)
    run_and_write(args, lambda _a: {'metric': 1.5})
    got = json.loads(out.read_text())
    assert got['command'] == 'diag'
    assert got['metric'] == 1.5
    assert len(got['load_avg_start']) == 3
    assert len(got['load_avg_end']) == 3
    assert 'git_sha' in got
