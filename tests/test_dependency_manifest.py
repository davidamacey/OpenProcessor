"""Guard against the shipped-image-can't-run-its-own-code class of bug
(see §0.5 of the OSS completion plan): ``scikit-learn``, ``umap-learn``,
``hdbscan`` and ``joblib`` were imported (lazily, inside functions) by
curation clustering code and appeared in **no** dependency manifest --
not ``requirements.txt``, not ``pyproject.toml``. Because the imports
are lazy, ``import src.main`` stayed silent and the whole test suite
passed in a dev venv that happened to have them hand-installed. The
first call to a real clustering endpoint in the shipped container would
``ImportError``.

This test converts that whole class of bug into two hard failures:

1. ``pyproject.toml``'s ``[project.dependencies]`` and
   ``requirements.txt`` must declare the exact same set of top-level
   packages (name only -- version pins/extras may legitimately differ).
2. Every top-level (module-scope, i.e. *not* inside a function/method --
   lazy imports are exactly the pattern that hid this bug, so they are
   deliberately *not* exempt just for being lazy) third-party import
   anywhere under ``src/`` must resolve to a package declared in at
   least one of the two manifests, or be explicitly allowlisted below
   with a documented reason.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'

# Python's stdlib + the project's own top-level packages. Anything in
# here is never expected in a dependency manifest.
_STDLIB_AND_LOCAL = {
    '__future__',
    'abc',
    'argparse',
    'array',
    'ast',
    'asyncio',
    'base64',
    'binascii',
    'bisect',
    'builtins',
    'collections',
    'concurrent',
    'configparser',
    'contextlib',
    'contextvars',
    'copy',
    'csv',
    'ctypes',
    'dataclasses',
    'datetime',
    'decimal',
    'difflib',
    'dis',
    'email',
    'enum',
    'errno',
    'functools',
    'gc',
    'getpass',
    'glob',
    'gzip',
    'hashlib',
    'heapq',
    'hmac',
    'html',
    'http',
    'imaplib',
    'importlib',
    'inspect',
    'io',
    'ipaddress',
    'itertools',
    'json',
    'logging',
    'math',
    'mimetypes',
    'multiprocessing',
    'numbers',
    'operator',
    'os',
    'pathlib',
    'pickle',
    'platform',
    'pprint',
    'queue',
    'random',
    're',
    'sched',
    'secrets',
    'select',
    'shutil',
    'signal',
    'site',
    'socket',
    'socketserver',
    'sqlite3',
    'ssl',
    'stat',
    'statistics',
    'string',
    'struct',
    'subprocess',
    'sys',
    'tarfile',
    'tempfile',
    'textwrap',
    'threading',
    'time',
    'timeit',
    'token',
    'tokenize',
    'trace',
    'traceback',
    'types',
    'typing',
    'typing_extensions',
    'unicodedata',
    'unittest',
    'urllib',
    'uuid',
    'venv',
    'warnings',
    'weakref',
    'xml',
    'zipfile',
    'zlib',
    'src',  # the project itself
}

# Import name -> pip distribution name, for the cases where they differ.
_IMPORT_TO_DIST = {
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'opensearchpy': 'opensearch-py',
    'yaml': 'pyyaml',
    'sklearn': 'scikit-learn',
    'google': 'protobuf',
    'faiss': 'faiss-gpu-cu12',
    'onnxruntime': 'onnxruntime-gpu',
    'umap': 'umap-learn',
    # facebookresearch/perception_models installs a top-level `core`
    # package under the `perception_models` pip distribution name --
    # see the verification note in src/clients/pe_encoder.py.
    'core': 'perception_models',
}

# Third-party top-level imports under src/ that are intentionally *not*
# required to appear in either manifest, with the reason each is safe.
_ALLOWLIST = {
    # Pillow is a transitive dependency of ultralytics; requirements.txt
    # documents this in a comment rather than pinning it directly.
    'PIL': 'transitive dependency of ultralytics (see requirements.txt comment)',
    # pydantic is a transitive dependency of fastapi/pydantic-settings --
    # both of which pin a compatible pydantic themselves.
    'pydantic': 'transitive dependency of fastapi/pydantic-settings',
    # grpc (the grpcio package) is pulled in by tritonclient[all].
    'grpc': "provided by tritonclient's [all] extra",
    # cachetools is an optional, soft dependency: src/services/curation/
    # image_serving.py imports it inside a try/except ImportError with a
    # small in-repo LRU stand-in as the fallback, by design.
    'cachetools': 'optional soft dependency with an in-repo fallback on ImportError',
    # GPU-accelerated clustering overlay (see docker-compose.gpu-clustering.yml
    # / `make cluster-gpu`): src/services/curation/clustering/backend.py's
    # _try_import_cuml() wraps both in a try/except and returns (None, None)
    # on failure, with every caller falling back to sklearn/umap on CPU.
    # Not part of the base image by design.
    'cuml': 'optional GPU-clustering overlay dependency with a CPU (sklearn/umap) fallback',
    'cupy': 'optional GPU-clustering overlay dependency with a CPU (sklearn/umap) fallback',
    # src/services/training/gpu_arbiter.py's _docker_client() wraps the
    # import (and the socket connection) in a try/except Exception and
    # returns None on any failure, falling back to the sentinel-only path.
    # The GPU arbiter is a documented no-op on main (see completion plan §3.4).
    'docker': 'optional docker-SDK path with a graceful None fallback (GPU arbiter, currently a no-op)',
    # src/services/curation/autolabel/job.py falls back to polling
    # (_watch_state_file_poll) inside a try/except ImportError.
    'inotify_simple': 'optional inotify-based watch with a polling fallback on ImportError',
    # First-party, not a pip package: src/routers/curation/bakeoff.py lazily
    # imports the stdlib-only scripts.curation.bakeoff.profile inside its
    # handlers. The API image ships scripts/ (Dockerfile COPY) and the router
    # already reads scripts/curation/bakeoff/baselines.json from it.
    'scripts': 'first-party repo package shipped in the API image (bake-off profiles)',
}

# Files whose dependency lists are legitimately out of scope for the
# pyproject.toml <-> requirements.txt reconciliation (different purpose,
# not meant to mirror the main runtime manifest).
_RECONCILIATION_ALLOWLIST: dict[str, str] = {}


def _normalize(name: str) -> str:
    name = name.split('#', 1)[0].strip()
    # PEP 508 direct reference, e.g. "perception_models @ git+https://...":
    # only the distribution name on the left of `@` is comparable.
    name = name.split('@', 1)[0].strip()
    name = re.split(r'[\[<>=!~;\s]', name)[0].strip()
    return name.lower().replace('_', '-')


def _pyproject_dependencies() -> set[str]:
    with (REPO_ROOT / 'pyproject.toml').open('rb') as f:
        data = tomllib.load(f)
    return {_normalize(d) for d in data['project']['dependencies'] if _normalize(d)}


def _requirements_txt_dependencies() -> set[str]:
    lines = (REPO_ROOT / 'requirements.txt').read_text().splitlines()
    return {n for line in lines if (n := _normalize(line))}


def test_pyproject_and_requirements_txt_agree() -> None:
    pyproject_deps = _pyproject_dependencies() | set(_RECONCILIATION_ALLOWLIST)
    requirements_deps = _requirements_txt_dependencies() | set(_RECONCILIATION_ALLOWLIST)

    only_pyproject = sorted(pyproject_deps - requirements_deps)
    only_requirements = sorted(requirements_deps - pyproject_deps)

    assert not only_pyproject, (
        'pyproject.toml [project.dependencies] declares packages missing from '
        f'requirements.txt: {only_pyproject}'
    )
    assert not only_requirements, (
        'requirements.txt declares packages missing from pyproject.toml '
        f'[project.dependencies]: {only_requirements}'
    )


class _TopLevelImportVisitor(ast.NodeVisitor):
    """Collects import roots that appear at *module* scope -- i.e. not
    nested inside a function/method body. Lazy imports are exactly the
    pattern that hid the scikit-learn/umap-learn/hdbscan/joblib bug, so
    this test does not exempt them; it walks function bodies too, just
    tracking whether we're inside one for informational purposes only.
    All imports anywhere in the file are collected -- lazy or not --
    because a package imported *only* lazily still needs to be in a
    manifest or the first real call ImportErrors in production.
    """

    def __init__(self) -> None:
        self.roots: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.roots.add(alias.name.split('.')[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self.roots.add(node.module.split('.')[0])


def _collect_third_party_import_roots() -> dict[str, list[Path]]:
    """Return ``{import_root: [files that import it]}`` for every
    non-stdlib, non-local import root found anywhere under ``src/``."""
    roots: dict[str, list[Path]] = {}
    for path in sorted(SRC_ROOT.rglob('*.py')):
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except SyntaxError:
            continue
        visitor = _TopLevelImportVisitor()
        visitor.visit(tree)
        for root in visitor.roots:
            if root in _STDLIB_AND_LOCAL:
                continue
            roots.setdefault(root, []).append(path.relative_to(REPO_ROOT))
    return roots


def test_every_third_party_import_under_src_is_declared_or_allowlisted() -> None:
    declared = _pyproject_dependencies() | _requirements_txt_dependencies()
    import_roots = _collect_third_party_import_roots()

    undeclared = {}
    for root, files in sorted(import_roots.items()):
        if root in _ALLOWLIST:
            continue
        dist_name = _normalize(_IMPORT_TO_DIST.get(root, root))
        if dist_name not in declared:
            undeclared[root] = files

    assert not undeclared, (
        'third-party modules imported under src/ with no entry in requirements.txt, '
        'pyproject.toml [project.dependencies], or the _ALLOWLIST in this test:\n'
        + '\n'.join(
            f'  {root} (imported by {", ".join(str(f) for f in files[:3])}'
            f'{", ..." if len(files) > 3 else ""})'
            for root, files in undeclared.items()
        )
    )
