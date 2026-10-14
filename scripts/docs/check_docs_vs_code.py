#!/usr/bin/env python3
"""Check the documentation against the code.

Five checks, each reporting every miss (not just the first):

* routes  - every ``METHOD /path`` written in a doc exists in the generated
  OpenAPI (``contracts/openapi/*.json``) or, for non-curation routes, in the
  assembled FastAPI app.
* env     - every ``OP_*`` token written in a doc exists in code, ``env.template``,
  compose files or the installer scripts.
* compose - every ``${VAR}`` a root compose file reads is assigned in ``env.template``
  (commented counts, a prose mention does not), so an operator can find every knob.
  Names the installer refuses (``RETIRED_ENV_NAMES``) are exempt.
* template - every non-``OP_`` setting ``env.template`` assigns is read by a compose file,
  script or code, so nothing advertised does nothing.
* links   - every relative link resolves to a file, and every ``#anchor`` that
  points into a markdown file matches a heading in that file.

Docs may write project-scoped paths without the ``/curation/projects/{project}``
prefix (the contract doc does); those are resolved against the prefixes below.

Usage:

    python3 scripts/docs/check_docs_vs_code.py            # all checks
    python3 scripts/docs/check_docs_vs_code.py routes env # a subset
    python3 scripts/docs/check_docs_vs_code.py --only docs/CURATION.md README.md

Exit status is 1 when any check finds a miss.
"""

from __future__ import annotations

import json
import re
import subprocess  # nosec B404 - fixed git invocation, no shell
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# History and generated files describe things that no longer exist or are
# produced from the code; they are not part of the documentation surface.
_EXCLUDED_FILES = {'CHANGELOG.md'}
_SKIP_DIR_NAMES = {
    '.git',
    '.venv',
    'venv',
    'node_modules',
    '__pycache__',
    '.claude',
    'artifacts_local',
    'build',
    '.docusaurus',
}
_DOC_SUFFIXES = {'.md', '.mdx'}

_METHODS = ('GET', 'POST', 'PUT', 'PATCH', 'DELETE')
_ROUTE_RE = re.compile(r'(?<![A-Za-z])(' + '|'.join(_METHODS) + r')[ \t|`*]+(/[A-Za-z0-9_/{}.\-]*)')
_ENV_RE = re.compile(r'\bOP_[A-Z0-9]+(?:_[A-Z0-9]+)*\b(?![_*<{])')

# Docs of side-car services describe those services' own HTTP surface, and the
# README documents Triton's management API. Neither is an OpenProcessor route.
_EXTERNAL_DOC_PREFIXES = ('docker/',)
_EXTERNAL_ROUTE_PREFIXES = ('/v2/',)

# Resolution prefixes for abbreviated paths, tried in order.
_ROUTE_PREFIXES = ('', '/curation', '/curation/projects/{project}')

_CODE_ENV_ROOTS = (
    'src',
    'scripts',
    'docker',
    'openprocessor',
    'config_templates',
    'frontend/e2e',
    'frontend/scripts',
)
_CODE_ENV_FILES = (
    'env.template',
    'setup-openprocessor.sh',
    'docker-compose.yml',
    'docker-compose.dev.yml',
    'docker-compose.gpu-arbiter.yml',
    'Makefile',
    'Dockerfile',
)


_STATE: dict[str, list[Path] | None] = {'only': None}


def _rel(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _tracked_paths() -> set[Path] | None:
    """Paths git tracks, or ``None`` outside a git checkout.

    Untracked local files (private notes, gitignored artifacts) are not part
    of the published documentation, so the checks ignore them.
    """
    try:
        out = subprocess.run(  # nosec B603 B607
            ['git', 'ls-files', '-z'], cwd=REPO_ROOT, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {REPO_ROOT / p for p in out.decode().split('\0') if p}


def doc_files() -> list[Path]:
    """Every doc file the checks apply to (or only ``--only`` paths)."""
    only = _STATE['only']
    if only is not None:
        return list(only)
    files: list[Path] = []
    tracked = _tracked_paths()
    for path in REPO_ROOT.rglob('*'):
        if path.suffix not in _DOC_SUFFIXES and path.suffix != '.json':
            continue
        if tracked is not None and path not in tracked:
            continue
        rel = path.relative_to(REPO_ROOT)
        if set(rel.parts) & _SKIP_DIR_NAMES:
            continue
        if path.suffix == '.json':
            # Diagram specs are the only json that carries prose.
            if 'architecture-diagrams' not in rel.parts:
                continue
        elif path.name in _EXCLUDED_FILES and len(rel.parts) == 1:
            continue
        if rel.parts[0] == 'tests' or rel.parts[:2] == ('docs', 'security'):
            continue
        files.append(path)
    return sorted(files)


def _norm(path: str) -> str:
    path = path.rstrip('.,;:)').rstrip('/') or '/'
    return re.sub(r'\{[^}]*\}', '{}', path)


def known_routes() -> set[tuple[str, str]]:
    """``(METHOD, normalised path)`` from the OpenAPI contract plus the app."""
    routes: set[tuple[str, str]] = set()
    for spec in sorted((REPO_ROOT / 'contracts' / 'openapi').glob('*.json')):
        for path, ops in json.loads(spec.read_text())['paths'].items():
            for method in ops:
                if method.upper() in _METHODS:
                    routes.add((method.upper(), _norm(path)))
    sys.path.insert(0, str(REPO_ROOT))
    from fastapi.routing import APIRoute, iter_route_contexts

    from src.main import app

    for route in iter_route_contexts(app.routes):
        if (
            isinstance(route.original_route, APIRoute)
            and route.path is not None
            and route.methods is not None
            and not route.path.startswith('/curation')
        ):
            for method in route.methods & set(_METHODS):
                routes.add((method, _norm(route.path)))
    return routes


def route_mentions(text: str) -> list[tuple[str, str]]:
    found = []
    for m in _ROUTE_RE.finditer(text):
        path = m.group(2)
        if path == '/' or path.startswith('//'):
            continue
        found.append((m.group(1), _norm(path.split('?')[0])))
    return found


def check_routes() -> list[str]:
    known = known_routes()
    errors = []
    for doc in doc_files():
        if _rel(doc).as_posix().startswith(_EXTERNAL_DOC_PREFIXES):
            continue
        text = doc.read_text(encoding='utf-8', errors='replace')
        for method, path in route_mentions(text):
            if path.startswith(_EXTERNAL_ROUTE_PREFIXES):
                continue
            if any((method, _norm(prefix + path)) in known for prefix in _ROUTE_PREFIXES):
                continue
            errors.append(f'{_rel(doc)}: {method} {path} is not a route')
    return errors


def _env_corpus() -> str:
    parts = []
    for name in _CODE_ENV_FILES:
        p = REPO_ROOT / name
        if p.is_file():
            parts.append(p.read_text(encoding='utf-8', errors='replace'))
    for root in _CODE_ENV_ROOTS:
        base = REPO_ROOT / root
        for p in base.rglob('*'):
            if not p.is_file() or set(p.relative_to(REPO_ROOT).parts) & _SKIP_DIR_NAMES:
                continue
            if p.suffix in {'.py', '.sh', '.mjs', '.yml', '.yaml', '.json', '.toml', ''}:
                parts.append(p.read_text(encoding='utf-8', errors='replace'))
    parts.extend(
        p.read_text(encoding='utf-8', errors='replace')
        for p in (
            *REPO_ROOT.glob('docker-compose*.yml'),
            *REPO_ROOT.glob('frontend/docker-compose*.yml'),
        )
    )
    return '\n'.join(parts)


def check_env() -> list[str]:
    corpus = _env_corpus()
    errors = []
    for doc in doc_files():
        text = doc.read_text(encoding='utf-8', errors='replace')
        for var in sorted(set(_ENV_RE.findall(text))):
            if var not in corpus:
                errors.append(f'{_rel(doc)}: {var} is not read by any code')  # noqa: PERF401  # the append carries a formatted message built next to its check
    return errors


_COMPOSE_VAR_RE = re.compile(r'(?<!\$)\$\{([A-Za-z_][A-Za-z0-9_]*)')
_TEMPLATE_ASSIGN_RE = re.compile(r'^#?[ \t]*([A-Z][A-Z0-9_]*)=', re.M)
_RETIRED_BLOCK_RE = re.compile(r'RETIRED_ENV_NAMES=\((.*?)\)', re.S)


def _template_names() -> set[str]:
    """Names env.template assigns, commented out or not; prose mentions do not count."""
    text = (REPO_ROOT / 'env.template').read_text(encoding='utf-8', errors='replace')
    return set(_TEMPLATE_ASSIGN_RE.findall(text))


def _retired_env_names() -> set[str]:
    """Names the installer refuses in a .env (renamed); compose may still pass them through."""
    script = REPO_ROOT / 'setup-openprocessor.sh'
    if not script.is_file():
        return set()
    block = _RETIRED_BLOCK_RE.search(script.read_text(encoding='utf-8', errors='replace'))
    return set(block.group(1).split()) if block else set()


def check_compose_vars() -> list[str]:
    documented = _template_names() | _retired_env_names()
    errors = []
    for compose in sorted(REPO_ROOT.glob('docker-compose*.yml')):
        text = compose.read_text(encoding='utf-8', errors='replace')
        for var in sorted(set(_COMPOSE_VAR_RE.findall(text))):
            if var not in documented:
                errors.append(  # noqa: PERF401  # the append carries a formatted message built next to its check
                    f'{_rel(compose)}: ${{{var}}} is not in env.template (commented is fine)'
                )
    return errors


def check_template_names() -> list[str]:
    """A non-OP_ setting in env.template that no compose file, script or code reads is
    advertised but does nothing (OP_ names are covered by tests/test_env_surface.py)."""
    template = (REPO_ROOT / 'env.template').read_text(encoding='utf-8', errors='replace')
    corpus = _env_corpus()
    errors = []
    for name in sorted(_template_names()):
        if name.startswith('OP_'):
            continue
        word = re.compile(rf'\b{re.escape(name)}\b')
        if len(word.findall(corpus)) <= len(word.findall(template)):
            errors.append(
                f'env.template: {name} is set by no compose file, script or code'
            )  # the append carries a formatted message built next to its check
    return errors


_LINK_RE = re.compile(r'(?<!\!)\]\(([^)\s]+)\)')
_HEADING_RE = re.compile(r'^#{1,6}\s+(.*?)\s*#*\s*$')


def slugify(heading: str) -> str:
    """GitHub-style heading anchor (also what Docusaurus produces)."""
    text = re.sub(r'`([^`]*)`', r'\1', heading)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text).strip().lower()
    text = re.sub(r'[^\w\- ]', '', text, flags=re.UNICODE)
    return text.replace(' ', '-')


def anchors_of(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
            continue
        m = None if in_fence else _HEADING_RE.match(line)
        if not m:
            continue
        slug = slugify(m.group(1))
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        anchors.add(slug if n == 0 else f'{slug}-{n}')
    for m in re.finditer(r'\{#([\w\-]+)\}', path.read_text(encoding='utf-8', errors='replace')):
        anchors.add(m.group(1))
    return anchors


def _resolve_link(doc: Path, path_part: str) -> Path | None:
    base = (
        REPO_ROOT / path_part.lstrip('/') if path_part.startswith('/') else doc.parent / path_part
    )
    for cand in (base, base.with_suffix('.md'), base.with_suffix('.mdx')):
        if cand.exists():
            return cand
    return None


def check_links() -> list[str]:
    errors = []
    for doc in doc_files():
        if doc.suffix not in _DOC_SUFFIXES:
            continue
        rel = _rel(doc)
        text = doc.read_text(encoding='utf-8', errors='replace')
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        for target in _LINK_RE.findall(text):
            if re.match(r'^[a-z][a-z0-9+.\-]*:', target):
                continue  # http:, https:, mailto:, pathname:, ...
            path_part, _, frag = target.partition('#')
            if path_part.startswith('/') and 'docs-site' in rel.parts:
                continue  # site-absolute routes are validated by the site build
            if not path_part:
                dest: Path = doc
            else:
                resolved = _resolve_link(doc, path_part)
                if resolved is None:
                    errors.append(f'{rel}: link target missing: {target}')
                    continue
                dest = resolved
            is_doc = dest.is_file() and dest.suffix in _DOC_SUFFIXES
            if frag and is_doc and frag not in anchors_of(dest):
                errors.append(f'{rel}: anchor not found: {target}')
    return errors


CHECKS = {
    'routes': check_routes,
    'env': check_env,
    'compose': check_compose_vars,
    'template': check_template_names,
    'links': check_links,
}


def set_only(paths: list[Path] | None) -> None:
    _STATE['only'] = paths


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if '--only' in args:
        i = args.index('--only')
        set_only([(REPO_ROOT / a).resolve() for a in args[i + 1 :]])
        args = args[:i]
    names = args or list(CHECKS)
    failed = False
    for name in names:
        errors = CHECKS[name]()
        print(f'{name}: {len(errors)} problem(s)')
        for err in errors:
            print(f'  {err}')
        failed = failed or bool(errors)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
