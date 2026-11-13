#!/usr/bin/env python3
"""Pre-commit guard: a comment or docstring that cites a repo file cites one that exists.

A citation such as "see the plan in the design directory" goes stale when the document is
renamed or removed, and then points a reader at nothing. This reads the comments and
docstrings the process-id guard reads (same scope) and fails on a cited path that
starts with a top-level documentation or tooling directory (``docs``, ``scripts``,
``examples``, ``docker``, ``export``, ``config_templates``, ``monitoring``,
``contracts``) and is not a file in the tree. Globs, ``<placeholders>`` and URLs are
not paths and are skipped.

Today's hits live in ``baselines/cited_paths.txt`` (``path<TAB>hits<TAB>reason``)
and can only shrink (see ``_baseline.py``).
"""

from __future__ import annotations

import re
import sys
import tokenize
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline
import check_process_ids as ids


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / 'scripts' / 'codegen' / 'baselines' / 'cited_paths.txt'

CITATION = re.compile(
    r'(?<![\w/.\-:~])'
    r'((?:docs|scripts|examples|docker|export|config_templates|monitoring|contracts)/'
    r'[\w./\-]*[\w\-]\.(?:py|sh|md|mdx|yml|yaml|json|tsv|txt|toml|ts|svelte|pbtxt))\b'
)


def missing_citations(repo_root: Path, rel: str) -> int:
    return sum(
        1
        for text in ids.comment_texts(repo_root, rel)
        for cited in CITATION.findall(text)
        if not (repo_root / cited).is_file()
    )


def scan(repo_root: Path) -> tuple[dict[str, int], list[str]]:
    found: dict[str, int] = {}
    broken: list[str] = []
    for rel in ids.tracked_files(repo_root):
        if not ids.in_content_scope(rel):
            continue
        try:
            n = missing_citations(repo_root, rel)
        except (SyntaxError, tokenize.TokenError, UnicodeDecodeError, OSError):
            broken.append(rel)
            continue
        if n:
            found[rel] = n
    ids.save_text_caches()
    return found, broken


def main(argv: list[str] | None = None) -> int:
    return int(
        _baseline.run_scan_ratchet(
            argv,
            doc=__doc__,
            baseline_path=BASELINE_PATH,
            repo_root=REPO_ROOT,
            scan=scan,
            unit='missing citation(s)',
            unreadable='cannot be read, so its comments are unchecked',
            failure=(
                'a comment cites a file that is not in the tree. Cite the file that exists, '
                'or say what the comment needs to say in words.'
            ),
        )
    )


if __name__ == '__main__':
    sys.exit(main())
