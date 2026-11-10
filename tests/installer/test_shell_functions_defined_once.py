"""Every shell function in the tracked tree is defined once per file.

bash silently keeps the last definition, so a duplicate is dead code that can drift from the live copy.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
_DEF = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\(\)\s*\{', re.M)


def _shell_files() -> list[Path]:
    out = subprocess.run(  # nosec B603 B607 - fixed git command, no user input
        ['git', 'ls-files', '*.sh', 'openprocessor', 'setup-openprocessor.sh'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / line for line in out.splitlines() if line]


def test_no_shell_function_is_defined_twice_in_one_file() -> None:
    dupes = {
        str(path.relative_to(ROOT)): sorted(
            n for n, c in Counter(_DEF.findall(path.read_text())).items() if c > 1
        )
        for path in _shell_files()
    }
    assert {f: names for f, names in dupes.items() if names} == {}
