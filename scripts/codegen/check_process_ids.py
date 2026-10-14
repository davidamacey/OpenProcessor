#!/usr/bin/env python3
"""Pre-commit guard: no work-package ids or history names in comments, docstrings or file names.

Process ids (wave, round and work-package numbers, review finding ids) describe
how a change was planned, not what the code does; they rot the moment the plan
is archived. Say why instead, or link a public issue (``#NNN``).

* Content: comments and docstrings of ``.py .ts .svelte .js .mjs .sh`` files and
  Dockerfiles under ``src scripts export docker models tests frontend/src
  frontend/e2e frontend/scripts``. Python is read with ``tokenize`` and ``ast``
  (string literals that are not docstrings are data and are ignored); the other
  languages by comment prefix (a whole-line comment or a trailing ``//``).
* File names: under ``tests scripts frontend/src frontend/e2e``, names that carry
  a round, wave or work-package number, a commit hash, or a review-pass name.

Today's hits live in ``baselines/process_ids.txt`` (``path<TAB>hits<TAB>reason``;
``path (file name)`` for name hits) and can only shrink (see ``_baseline.py``).
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / 'scripts' / 'codegen' / 'baselines' / 'process_ids.txt'

CONTENT_DIRS = (
    'src/',
    'scripts/',
    'export/',
    'docker/',
    'models/',
    'tests/',
    'frontend/src/',
    'frontend/e2e/',
    'frontend/scripts/',
)
VENDORED_PREFIXES = ('src/static/',)
NAME_DIRS = ('tests/', 'scripts/', 'frontend/src/', 'frontend/e2e/')
C_STYLE = ('.ts', '.svelte', '.js', '.mjs')
HASH_STYLE = ('.sh',)

ID_PATTERN = re.compile(
    r'\bW\d+(?:\.\d+)*[a-z]?\b'
    r'|\bR\d+-\d+\b'
    r'|\bWP-[A-Z]?\d[\w.]*'
    r'|\bround-\d+\b'
    r'|\bF-\d+\b'
    r'|\bChunk \d+\b'
    r'|\bWave \d+\b'
    r'|\bdqM\d+'
    r'|\bPC-\d+'
)
NAME_PATTERN = re.compile(
    r'(?:^|[_/])(?:r\d+|w\d+[a-z]?|wave\d+|round\d+)[_.]|[0-9a-f]{7}\.|dqM\d|logicMoves|interactivePass'
)
_TRAILING_SLASHES = re.compile(r'(?:^|\s)//\s?(.*)$')


def tracked_files(repo_root: Path) -> list[str]:
    out = subprocess.run(  # nosec B603 B607  fixed argv, git from PATH
        ['git', 'ls-files', '-z'], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split('\0') if p]


def _is_dockerfile(rel: str) -> bool:
    return rel.rsplit('/', 1)[-1].startswith('Dockerfile')


def _comment_texts_python(source: str, rel: str) -> list[str]:
    tree = ast.parse(source, filename=rel)
    texts = [
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.COMMENT
    ]
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                texts.append(doc)
    return texts


def _comment_texts_other(source: str, rel: str) -> list[str]:
    texts: list[str] = []
    hash_style = (
        rel.endswith(HASH_STYLE) or _is_dockerfile(rel) or '.' not in rel.rsplit('/', 1)[-1]
    )
    in_block = False
    for line in source.splitlines():
        stripped = line.strip()
        if hash_style:
            if stripped.startswith('#') and not stripped.startswith('#!'):
                texts.append(stripped)
            continue
        if in_block:
            texts.append(stripped)
            in_block = '*/' not in stripped
            continue
        if stripped.startswith('/*'):
            texts.append(stripped)
            in_block = '*/' not in stripped
        elif stripped.startswith(('//', '<!--')):
            texts.append(stripped)
        else:
            match = _TRAILING_SLASHES.search(line)
            if match:
                texts.append(match.group(1))
    return texts


def content_hits(repo_root: Path, rel: str) -> int:
    source = (repo_root / rel).read_text(encoding='utf-8')
    if rel.endswith('.py'):
        texts = _comment_texts_python(source, rel)
    else:
        texts = _comment_texts_other(source, rel)
    return sum(len(ID_PATTERN.findall(t)) for t in texts)


def in_content_scope(rel: str) -> bool:
    if rel.startswith(VENDORED_PREFIXES):
        return False
    if rel in {'openprocessor', 'setup-openprocessor.sh'}:
        return True
    if rel.endswith(('.py', *C_STYLE, *HASH_STYLE)) and rel.startswith(CONTENT_DIRS):
        return True
    return _is_dockerfile(rel)


def scan(repo_root: Path, files: list[str] | None = None) -> tuple[dict[str, int], list[str]]:
    """Return ``({key: hits}, unreadable files)``."""
    found: dict[str, int] = {}
    broken: list[str] = []
    for rel in files if files is not None else tracked_files(repo_root):
        if rel.startswith(NAME_DIRS) and NAME_PATTERN.search(rel):
            found[f'{rel} (file name)'] = 1
        if not in_content_scope(rel):
            continue
        try:
            n = content_hits(repo_root, rel)
        except (SyntaxError, tokenize.TokenError, UnicodeDecodeError, OSError):
            broken.append(rel)
            continue
        if n:
            found[rel] = n
    return found, broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE_PATH)
    parser.add_argument('--tighten', action='store_true', help='Lower baseline values, drop stale.')
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    found, broken = scan(args.root)
    baseline = _baseline.load(args.baseline)
    errors = [f'{rel}: cannot be read, so its comments are unchecked' for rel in broken]
    errors += _baseline.compare(found, baseline, unit='hit(s)')
    if args.tighten:
        changed = _baseline.tighten(args.baseline, found, baseline)
        sys.stderr.write(f'tightened {changed} baseline line(s)\n')
        errors = [e for e in errors if 'new violation' in e or 'grew past' in e or 'read' in e]
    if errors:
        sys.stderr.write(
            'ERROR: work-package ids or history names found. Say why in plain words, link a '
            'public issue (#NNN), and name files by what they test.\n'
        )
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
