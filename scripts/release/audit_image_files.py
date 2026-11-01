#!/usr/bin/env python3
"""Pre-publish audit of a built image against the committed tree.

Two checks, both fail-closed:

1. Every file inside a directory the final stage of the Dockerfile COPYs from the
   build context must map back to a file tracked in git at the release revision.
   (`__pycache__` is accepted only when the Dockerfile compiles bytecode itself.)
2. No file anywhere in the image may be a model artifact (*.onnx, *.plan, *.engine,
   *.pt, *.pth, *.safetensors) or a `<project>__<model>` directory, unless it
   matches scripts/release/image_audit_allowlist.txt.

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
COMPILES_BYTECODE = re.compile(r'\b(compileall|py_compile)\b')


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


def _final_stage_copies(dockerfile_text: str) -> tuple[list[tuple[list[str], str]], bool]:
    """Return ([(sources, absolute_dest)], compiles_bytecode) for the final stage."""
    copies: list[tuple[list[str], str]] = []
    workdir = '/'
    compiles = False
    for line in _logical_lines(dockerfile_text):
        word, _, rest = line.partition(' ')
        word = word.upper()
        if word == 'FROM':
            copies, workdir, compiles = [], '/', False
        elif word == 'WORKDIR':
            workdir = posixpath.normpath(posixpath.join(workdir, rest.strip()))
        elif word == 'RUN' and COMPILES_BYTECODE.search(rest):
            compiles = True
        elif word == 'COPY':
            args = rest.split()
            if any(a.startswith('--from=') for a in args):
                continue
            args = [a for a in args if not a.startswith('--')]
            if len(args) < 2:
                continue
            dest = posixpath.normpath(posixpath.join(workdir, args[-1]))
            copies.append((args[:-1], dest + ('/' if args[-1].endswith('/') else '')))
    return copies, compiles


def _expected_and_roots(
    dockerfile_text: str, context: str, tracked: set[str]
) -> tuple[set[str], set[str], bool]:
    copies, compiles = _final_stage_copies(dockerfile_text)
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
            if any(t.startswith(below) for t in tracked) or src.endswith('/') or src in ('.', './'):
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
    return expected, roots, compiles


def directory_copy_roots(dockerfile_text: str, context: str, tracked: set[str]) -> list[str]:
    """In-image directories populated by a directory COPY in the final stage."""
    return sorted(_expected_and_roots(dockerfile_text, context, tracked)[1])


def untracked_files(
    dockerfile_text: str, context: str, tracked: set[str], listing: set[str] | list[str]
) -> list[str]:
    """Files of `listing` under a directory-COPY destination that git does not track."""
    expected, roots, compiles = _expected_and_roots(dockerfile_text, context, tracked)
    bad: list[str] = []
    for path in sorted(set(listing)):
        if not any(path.startswith(f'{r}/') for r in roots) or path in expected:
            continue
        if compiles and '/__pycache__/' in path and path.endswith('.pyc'):
            continue
        bad.append(path)
    return bad


def forbidden_files(paths: list[str] | set[str], allowlist: list[str]) -> list[str]:
    """Model artifacts and `<project>__<model>` directories not on the allowlist."""
    bad: list[str] = []
    for path in sorted(set(paths)):
        name = posixpath.basename(path.rstrip('/'))
        hit = name.endswith(FORBIDDEN_SUFFIXES) or bool(PROJECT_MODEL_DIR.match(name))
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
    except ListingError as exc:
        print(f'image audit cannot complete: {exc}', file=sys.stderr)
        return 1


def _audit() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', required=True)
    ap.add_argument('--dockerfile', required=True, help='path of the Dockerfile on disk')
    ap.add_argument('--context', required=True, help='build context dir, relative to the repo')
    ap.add_argument('--repo', required=True)
    ap.add_argument('--rev', required=True)
    ap.add_argument('--docker-bin', default='docker')
    ap.add_argument('--allowlist', required=True)
    args = ap.parse_args()

    tracked = set(
        _run(['git', '-C', args.repo, 'ls-tree', '-r', '--name-only', args.rev]).split('\n')
    )
    text = Path(args.dockerfile).read_text()
    problems: list[str] = []

    for root in directory_copy_roots(text, args.context, tracked):
        listing = _image_listing(args.docker_bin, args.image, f'{root} -type f')
        problems += [
            f'untracked in image (not in git at {args.rev}): {p}'
            for p in untracked_files(text, args.context, tracked, listing)
        ]

    everything = _image_listing(
        args.docker_bin,
        args.image,
        r'/ -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o '
        r'\( -type f -o -type d \) -print',
    )
    allowlist = load_allowlist(Path(args.allowlist))
    problems += [
        f'forbidden model artifact in image: {p}' for p in forbidden_files(everything, allowlist)
    ]

    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
