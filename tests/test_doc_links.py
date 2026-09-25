"""Guard against dangling relative markdown links: 42 references across 36
files once pointed at a design doc (``docs/design/curation_design_rationale.md``)
that did not exist on this branch.
Three of those were *live* links that rendered broken in
``docs/README.md``, ``docs/ARCHITECTURE.md`` and ``CLAUDE.md``. This
test converts that whole class of bug into a hard failure so it cannot
silently recur.

Walks every ``*.md`` file in the repository (skipping VCS/venv/cache
directories that legitimately don't ship real content), extracts every
``](target.md)`` / ``](target.md#fragment)``-shaped relative link
target, resolves it against the *linking file's own directory*
(standard markdown relative-link semantics), and asserts the resolved
path exists. Absolute paths (``/...``) and URLs (``http(s)://``,
``mailto:``) are out of scope — this only polices relative,
repo-internal links.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories that legitimately contain no real repo content, or whose
# markdown files are generated/vendored and not part of this project's
# own documentation graph.
_SKIP_DIR_NAMES = {
    '.git',
    '.venv',
    'venv',
    'node_modules',
    '.pytest_cache',
    '__pycache__',
    '.mypy_cache',
    '.ruff_cache',
    '.claude',
}

# A markdown inline link's target: ``](target)``. Deliberately does not
# try to handle titled links (``](target "title")``) or link-reference
# definitions — none are used for internal doc cross-references in this
# repo, and a stray unmatched one would show up as a resolution failure
# below rather than a silent miss.
_LINK_RE = re.compile(r'\]\(([^)\s]+)\)')

# Only a target that looks like a real relative filesystem path ending
# in ``.md`` is in scope. This intentionally rejects anything containing
# non-path characters (e.g. stray unicode/prose accidentally captured by
# `_LINK_RE` inside a code span) rather than trying to special-case it.
_RELATIVE_MD_PATH_RE = re.compile(r'^[\w./\-]+\.md$')


def _tracked_markdown_files() -> set[Path] | None:
    """Markdown files git tracks, or ``None`` outside a git checkout.

    Gitignored local artifacts (e.g. ``artifacts_local/``) aren't part of
    the repository, so their links are not this test's concern.
    """
    import subprocess  # nosec B404 - fixed git invocation, no shell

    try:
        out = subprocess.run(  # nosec B603 B607
            ['git', 'ls-files', '-z', '--', '*.md'],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {REPO_ROOT / p for p in out.decode().split('\0') if p}


def _iter_markdown_files() -> list[Path]:
    tracked = _tracked_markdown_files()
    files = []
    for path in REPO_ROOT.rglob('*.md'):
        if tracked is not None and path not in tracked:
            continue
        rel_parts = set(path.relative_to(REPO_ROOT).parts)
        if rel_parts & _SKIP_DIR_NAMES:
            continue
        files.append(path)
    return sorted(files)


def _iter_relative_md_links(md_file: Path) -> list[tuple[str, Path]]:
    """Return ``(original_target, resolved_path)`` pairs for every
    relative ``*.md`` link found in ``md_file``."""
    text = md_file.read_text(encoding='utf-8', errors='replace')
    found = []
    for match in _LINK_RE.finditer(text):
        target = match.group(1)
        path_part = target.split('#', 1)[0]
        if not path_part:
            continue  # pure same-file anchor, e.g. `](#section)`
        if path_part.startswith(('http://', 'https://', 'mailto:', '/')):
            continue
        if not _RELATIVE_MD_PATH_RE.match(path_part):
            continue
        resolved = (md_file.parent / path_part).resolve()
        found.append((target, resolved))
    return found


def _collect_all_links() -> list[tuple[Path, str, Path]]:
    links = []
    for md_file in _iter_markdown_files():
        for target, resolved in _iter_relative_md_links(md_file):
            links.append((md_file, target, resolved))
    return links


def test_no_broken_relative_markdown_links() -> None:
    all_links = _collect_all_links()
    assert all_links, 'expected to find at least one relative markdown link to check'

    broken = [
        (md_file.relative_to(REPO_ROOT), target)
        for md_file, target, resolved in all_links
        if not resolved.is_file()
    ]
    assert not broken, 'dangling relative markdown links found:\n' + '\n'.join(
        f'  {md_file}: ]({target})' for md_file, target in broken
    )
