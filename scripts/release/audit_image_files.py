#!/usr/bin/env python3
"""Pre-publish audit of a built image against the committed tree.

Two checks, both fail-closed:

1. Every file inside a directory the final stage of the Dockerfile COPYs from the
   build context must map back to a file tracked in git at the release revision.
   A `<dir>/__pycache__/<stem>.<tag>.pyc` is accepted only when `<dir>/<stem>.py` is itself
   a tracked file of the same COPY'd tree (build-time imports compile the shipped sources);
   a .pyc without its tracked source, or a __pycache__ with no tracked source dir, fails.
2. No model artifact may be in the image, unless the path matches
   scripts/release/image_audit_allowlist.txt:
   - Extension rule: a file named *.onnx, *.plan, *.engine, *.pt, *.pth or *.safetensors
     is forbidden anywhere in the image when it is larger than 1 MiB. Real weights are MBs
     to GBs; path-config .pth files and library test fixtures (vim's tutor.pt) are tiny.
     Inside a model-repo root any untracked file is already caught by check 1.
   - Project-model rule: a `<a>__<b>` directory is forbidden only as a direct child of a
     Triton model-repository root: /opt/openprocessor/model_repo_seed and every
     destination of a final-stage Dockerfile `COPY models/`. Elsewhere `__` is ordinary
     (scipy test__numdiff.py, torch linalg__powsum.h).

The 0.6.0 API image shipped 24 untracked files under model_repo_seed; this is the
guard that would have stopped it. Standard library only.
"""

from __future__ import annotations

import argparse
import fnmatch
import posixpath
import re
import subprocess
import sys
from pathlib import Path


FORBIDDEN_SUFFIXES = ('.onnx', '.plan', '.engine', '.pt', '.pth', '.safetensors')
# `<project>__<model>`: a name with an interior double underscore, not a dunder name.
PROJECT_MODEL_DIR = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*__[A-Za-z0-9][A-Za-z0-9._-]*$')


def _logical_lines(text: str) -> list[str]:
    lines: list[str] = []
    buf = ''
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buf and (not stripped or stripped.startswith('#')):
            continue
        if stripped.endswith('\\'):
            buf += stripped[:-1] + ' '
            continue
        lines.append((buf + stripped).strip())
        buf = ''
    if buf:
        lines.append(buf.strip())
    return lines


def _stage_name(from_args: str) -> str | None:
    words = [w for w in from_args.split() if not w.startswith('--')]
    return words[2] if len(words) == 3 and words[1].upper() == 'AS' else None


def _final_stage_copies(
    dockerfile_text: str, target: str | None = None
) -> list[tuple[list[str], str]]:
    """Return [(sources, absolute_dest)] for the COPYs of the final stage, or of the stage named
    `target` (a `docker build --target` build ends there)."""
    copies: list[tuple[list[str], str]] = []
    workdir = '/'
    found = target is None
    for line in _logical_lines(dockerfile_text):
        word, _, rest = line.partition(' ')
        word = word.upper()
        if word == 'FROM':
            if target is not None:
                if found:
                    break
                found = _stage_name(rest) == target
            copies, workdir = [], '/'
        elif word == 'WORKDIR':
            workdir = posixpath.normpath(posixpath.join(workdir, rest.strip()))
        elif word == 'COPY':
            args = rest.split()
            if any(a.startswith('--from=') for a in args):
                continue
            args = [a for a in args if not a.startswith('--')]
            if len(args) < 2:
                continue
            dest = posixpath.normpath(posixpath.join(workdir, args[-1]))
            copies.append((args[:-1], dest + ('/' if args[-1].endswith('/') else '')))
    if not found:
        raise ContextMismatchError(f'the Dockerfile has no stage {target!r}')
    return copies


def _expected_and_roots(
    dockerfile_text: str, context: str, tracked: set[str], target: str | None = None
) -> tuple[set[str], set[str]]:
    copies = _final_stage_copies(dockerfile_text, target)
    prefix = '' if context in ('', '.') else context.rstrip('/') + '/'
    expected: set[str] = set()
    roots: set[str] = set()
    for sources, dest in copies:
        dest_is_dir = dest.endswith('/') or len(sources) > 1
        dest_dir = dest.rstrip('/')
        for src in sources:
            clean = (
                posixpath.normpath(prefix + src.removeprefix('./'))
                if src not in ('.', './')
                else prefix.rstrip('/') or '.'
            )
            below = f'{clean}/'
            is_dir = src.endswith('/') or src in ('.', './')
            if is_dir and not any(clean == '.' or t.startswith(below) for t in tracked):
                raise ContextMismatchError(
                    f'COPY source {src!r} has no tracked files under context {context!r}; '
                    '--context must be the build context directory relative to the repo root'
                )
            if is_dir or any(t.startswith(below) for t in tracked):
                roots.add(dest_dir)
                for t in tracked:
                    if clean == '.' or t.startswith(below):
                        rel = t if clean == '.' else t[len(below) :]
                        expected.add(f'{dest_dir}/{rel}')
                continue
            for t in tracked:
                if fnmatch.fnmatchcase(t, clean):
                    name = posixpath.basename(t)
                    expected.add(f'{dest_dir}/{name}' if dest_is_dir else dest_dir)
    return expected, roots


def directory_copy_roots(
    dockerfile_text: str, context: str, tracked: set[str], target: str | None = None
) -> list[str]:
    """In-image directories populated by a directory COPY in the final (or target) stage."""
    return sorted(_expected_and_roots(dockerfile_text, context, tracked, target)[1])


def _pyc_has_tracked_source(path: str, expected: set[str]) -> bool:
    directory, name = posixpath.split(path)
    parent, cache = posixpath.split(directory)
    if cache != '__pycache__' or not name.endswith('.pyc'):
        return False
    stem = name.removesuffix('.pyc').split('.')[0]
    return f'{parent}/{stem}.py' in expected


def untracked_files(
    dockerfile_text: str,
    context: str,
    tracked: set[str],
    listing: set[str] | list[str],
    target: str | None = None,
) -> list[str]:
    """Files of `listing` under a directory-COPY destination that git does not track."""
    expected, roots = _expected_and_roots(dockerfile_text, context, tracked, target)
    bad: list[str] = []
    for path in sorted(set(listing)):
        if not any(path.startswith(f'{r}/') for r in roots) or path in expected:
            continue
        if _pyc_has_tracked_source(path, expected):
            continue
        bad.append(path)
    return bad


LARGE_FILE_BYTES = 1024 * 1024
SEED_ROOT = '/opt/openprocessor/model_repo_seed'


class ContextMismatchError(ValueError):
    """A directory COPY source has no tracked files: --context is not repo-relative."""


def model_roots(dockerfile_text: str, target: str | None = None) -> list[str]:
    """Model-repository roots: the seed dir plus every final-stage `COPY models/` destination."""
    roots = {SEED_ROOT}
    for sources, dest in _final_stage_copies(dockerfile_text, target):
        if any(posixpath.basename(posixpath.normpath(src)) == 'models' for src in sources):
            roots.add(dest.rstrip('/'))
    return sorted(roots)


def forbidden_files(
    paths: list[str] | set[str],
    allowlist: list[str],
    large: set[str] | frozenset[str] = frozenset(),
    model_roots: list[str] | tuple[str, ...] = (SEED_ROOT,),
) -> list[str]:
    """Model artifacts and `<project>__<model>` directories not on the allowlist.

    `large` holds the paths known to exceed LARGE_FILE_BYTES; a model extension is
    forbidden only for those.
    """
    bad: list[str] = []
    for path in sorted(set(paths)):
        name = posixpath.basename(path.rstrip('/'))
        if name.endswith(FORBIDDEN_SUFFIXES):
            hit = path in large
        else:
            parent = posixpath.dirname(path.rstrip('/'))
            hit = parent in model_roots and bool(PROJECT_MODEL_DIR.match(name))
        if hit and not any(fnmatch.fnmatchcase(path, pat) for pat in allowlist):
            bad.append(path)
    return bad


def load_allowlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith('#')
    ]


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout  # nosec B603


class ListingError(RuntimeError):
    """The in-image file listing failed; a partial listing is never treated as a pass."""


def _image_listing(docker: str, image: str, find_expr: str) -> list[str]:
    # Root, so a non-root image user cannot hide unreadable directories from the audit.
    cmd = [
        docker, 'run', '--rm', '--network', 'none', '--user', '0:0',
        '--entrypoint', 'sh', image, '-c', f'find {find_expr}',
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # nosec B603
    if proc.returncode != 0:
        tail = '\n'.join(proc.stderr.strip().splitlines()[-10:])
        raise ListingError(
            f'listing {image} failed (exit {proc.returncode}); command: {" ".join(cmd)}\n'
            f'stderr tail:\n{tail}'
        )
    return [ln for ln in proc.stdout.splitlines() if ln]


def main() -> int:
    try:
        return _audit()
    except (ListingError, ContextMismatchError) as exc:
        print(f'image audit cannot complete: {exc}', file=sys.stderr)
        return 1


def _audit() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', required=True)
    ap.add_argument('--dockerfile', required=True, help='path of the Dockerfile on disk')
    ap.add_argument('--context', required=True, help='build context dir, relative to the repo')
    ap.add_argument('--repo', required=True)
    ap.add_argument('--rev', required=True)
    ap.add_argument('--target', help='the Dockerfile stage the image was built to (--target)')
    ap.add_argument('--docker-bin', default='docker')
    ap.add_argument('--allowlist', required=True)
    args = ap.parse_args()

    tracked = set(
        _run(['git', '-C', args.repo, 'ls-tree', '-r', '--name-only', args.rev]).split('\n')
    )
    text = Path(args.dockerfile).read_text()
    problems: list[str] = []

    for root in directory_copy_roots(text, args.context, tracked, args.target):
        listing = _image_listing(args.docker_bin, args.image, f'{root} -type f')
        problems += [
            f'untracked in image (not in git at {args.rev}): {p}'
            for p in untracked_files(text, args.context, tracked, listing, args.target)
        ]

    everything = _image_listing(
        args.docker_bin,
        args.image,
        r'/ -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o '
        r'\( -type f -o -type d \) -print',
    )
    large = set(
        _image_listing(
            args.docker_bin,
            args.image,
            r'/ -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o '
            f'-type f -size +{LARGE_FILE_BYTES}c -print',
        )
    )
    allowlist = load_allowlist(Path(args.allowlist))
    forbidden = forbidden_files(everything, allowlist, large, model_roots(text, args.target))
    problems += [f'forbidden model artifact in image: {p}' for p in forbidden]

    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
