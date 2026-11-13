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

import ast
import hashlib
import io
import json
import os
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


_BODY_FIELDS = ('body', 'orelse', 'finalbody', 'handlers', 'cases')


_Owner = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


def _docstring_owners(node: ast.AST) -> list[_Owner]:
    """Every module, class and function under ``node``, found through statement lists only.

    A docstring owner is a statement, so there is no need to visit every expression node the way
    ``ast.walk`` does (most of a tree); each statement-list field of the node types that can hold
    a nested definition (``if``, ``for``, ``try``, ``with``, ``match`` ...) is followed instead.
    """
    owners: list[_Owner] = []
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, _Owner):
            owners.append(current)
        for name in _BODY_FIELDS:
            children = getattr(current, name, None)
            if isinstance(children, list):
                stack.extend(children)
    return owners


def _comment_texts_python(source: str, rel: str) -> list[str]:
    tree = ast.parse(source, filename=rel)
    texts = [
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.COMMENT
    ]
    for node in _docstring_owners(tree):
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


class _TextCache:
    """Comments and docstrings per file, keyed by the file's content hash.

    The process-id guard and the cited-path guard read the same texts, and parsing every
    tracked Python file is almost all of their time. The file lives under ``<root>/.cache``
    (gitignored, so one per checkout). It is discarded whole when this script or the Python
    version changed, and an entry is used only when its content hash still matches, so a
    missing, stale or corrupt cache costs time and never changes a result.
    """

    def __init__(self, repo_root: Path) -> None:
        self.path = repo_root / '.cache' / 'codegen' / 'comment_texts.json'
        self.root = repo_root
        self.salt = hashlib.sha256(Path(__file__).read_bytes() + sys.version.encode()).hexdigest()
        self.files: dict[str, list[object]] = {}
        self.dirty = False
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if data.get('salt') == self.salt and isinstance(data.get('files'), dict):
                self.files = data['files']
        except (OSError, ValueError, AttributeError):
            self.files = {}

    def texts(self, rel: str, source: str) -> list[str]:
        digest = hashlib.sha256(source.encode()).hexdigest()
        hit = self.files.get(rel)
        if hit is not None and hit[0] == digest and isinstance(hit[1], list):
            return [str(t) for t in hit[1]]
        found = (
            _comment_texts_python(source, rel)
            if rel.endswith('.py')
            else _comment_texts_other(source, rel)
        )
        self.files[rel] = [digest, found]
        self.dirty = True
        return found

    def save(self) -> None:
        if not self.dirty:
            return
        files = {rel: v for rel, v in self.files.items() if (self.root / rel).is_file()}
        tmp = self.path.with_name(f'{self.path.name}.{os.getpid()}.tmp')
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({'salt': self.salt, 'files': files}), encoding='utf-8')
            tmp.replace(self.path)
        except OSError:
            tmp.unlink(missing_ok=True)
        self.dirty = False


_CACHES: dict[Path, _TextCache] = {}


def comment_texts(repo_root: Path, rel: str) -> list[str]:
    """The comments and docstrings of ``rel`` (cached by content; see :class:`_TextCache`)."""
    cache = _CACHES.get(repo_root)
    if cache is None:
        cache = _CACHES[repo_root] = _TextCache(repo_root)
    return cache.texts(rel, (repo_root / rel).read_text(encoding='utf-8'))


def save_text_caches() -> None:
    for cache in _CACHES.values():
        cache.save()


def content_hits(repo_root: Path, rel: str) -> int:
    return sum(len(ID_PATTERN.findall(t)) for t in comment_texts(repo_root, rel))


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
    save_text_caches()
    return found, broken


def main(argv: list[str] | None = None) -> int:
    return int(
        _baseline.run_scan_ratchet(
            argv,
            doc=__doc__,
            baseline_path=BASELINE_PATH,
            repo_root=REPO_ROOT,
            scan=scan,
            unit='hit(s)',
            unreadable='cannot be read, so its comments are unchecked',
            failure=(
                'work-package ids or history names found. Say why in plain words, link a '
                'public issue (#NNN), and name files by what they test.'
            ),
        )
    )


if __name__ == '__main__':
    sys.exit(main())
