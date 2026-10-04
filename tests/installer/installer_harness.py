"""Shared harness for the installer tests (installer plan section 9.2).

Nothing here talks to a real Docker daemon, GPU or network: every external
tool the installer uses (docker, curl, nvidia-smi, ss, df) is replaced by a
PATH shim from tests/installer/shims/ that logs its argv to a file and
answers from fixture files. The fake release is built with the real
scripts/release/build_deploy_bundle.sh, so the bundle format is tested too.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / 'setup-openprocessor.sh'
SHIMS = Path(__file__).resolve().parent / 'shims'
RELEASE = 'v9.9.9'
CW_TAG = 'v1.2.3'
PROJECT = 'opinst-test'

# One 48 GB card with nothing on it.
GPU_48 = '0, NVIDIA RTX A6000, 49140, 0, 8.6\n'
# This host's shape: A6000, 3080 Ti, A6000.
GPU_HOST = (
    '0, NVIDIA RTX A6000, 49140, 0, 8.6\n'
    '1, NVIDIA GeForce RTX 3080 Ti, 12288, 0, 8.6\n'
    '2, NVIDIA RTX A6000, 49140, 0, 8.6\n'
)

CW_COMPOSE = """services:
  cropwright:
    image: ${CROPWRIGHT_IMAGE:-davidamacey/cropwright:1.2.3}
    container_name: ${CROPWRIGHT_CONTAINER_NAME:-cropwright}
    ports:
      - '${CROPWRIGHT_BIND_ADDRESS:-0.0.0.0}:${CROPWRIGHT_PORT:-5184}:8080'
    networks:
      - api
networks:
  api:
    external: true
    name: ${OP_DOCKER_NETWORK:-openprocessor_triton_net}
"""
CW_ENV_EXAMPLE = 'CROPWRIGHT_PORT=5184\nPUBLIC_API_PREFIX=/curation\n'


def run_bash(
    script: str, *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run a bash snippet, returning the CompletedProcess (never raises)."""
    return subprocess.run(
        ['bash', '-c', script],
        check=False,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        start_new_session=True,
    )


def fake_digest(name: str) -> str:
    return 'sha256:' + hashlib.sha256(name.encode()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_files(src: Path) -> list[str]:
    files: list[str] = []
    for raw in (src / 'release-manifest.txt').read_text().splitlines():
        parts = raw.split()
        if not parts or parts[0].startswith('#'):
            continue
        path = parts[0]
        if path.endswith('**'):
            base = src / path[:-2]
            files.extend(str(p.relative_to(src)) for p in base.rglob('*') if p.is_file())
        elif (src / path).is_file():
            files.append(path)
    return files


def image_key_refs() -> list[tuple[str, str]]:
    """(key, repo[:tag]) for every images.lock key in scripts/lib/image_keys.sh."""
    out = subprocess.run(
        [
            'bash',
            '-c',
            f'source "{REPO_ROOT}/scripts/lib/image_keys.sh"; for k in $(image_keys); do '
            'echo "$k $(image_key_field "$k" kind) $(image_key_field "$k" image)$(image_key_field "$k" source)"; done',
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    refs = []
    for line in out.splitlines():
        key, kind, ref = line.split()
        refs.append((key, f'davidamacey/{ref}' if kind == 'build' else ref))
    return refs


def build_fake_release(root: Path, *, lock_override: str | None = None) -> Path:
    """Build release assets + raw files + a Cropwright release under root."""
    src = root / 'src'
    for rel in _manifest_files(REPO_ROOT):
        dest = src / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dest)

    lock_lines = [f'{key}={ref}@{fake_digest(key)}' for key, ref in image_key_refs()]
    (src / 'images.lock').write_text(lock_override or '\n'.join(lock_lines) + '\n')

    cw_dir = root / 'release' / 'cw' / CW_TAG
    cw_dir.mkdir(parents=True)
    (cw_dir / 'docker-compose.yml').write_text(CW_COMPOSE)
    (cw_dir / '.env.example').write_text(CW_ENV_EXAMPLE)
    (cw_dir / 'SHA256SUMS').write_text(
        f'{_sha(cw_dir / "docker-compose.yml")}  docker-compose.yml\n'
        f'{_sha(cw_dir / ".env.example")}  .env.example\n'
    )
    (src / 'cropwright.lock').write_text(
        f'tag={CW_TAG}\n'
        f'image=davidamacey/cropwright@{fake_digest("cropwright")}\n'
        f'sha256sums_sha256={_sha(cw_dir / "SHA256SUMS")}\n'
    )

    assets = root / 'release' / 'assets' / RELEASE
    env = {**os.environ, 'ALLOW_UNPINNED_LOCK': '1' if lock_override else '0'}
    subprocess.run(
        [
            'bash',
            str(REPO_ROOT / 'scripts/release/build_deploy_bundle.sh'),
            RELEASE,
            str(src),
            str(assets),
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    shutil.copytree(src, root / 'release' / 'raw' / RELEASE)
    return root / 'release'


@dataclass
class Shimmed:
    root: Path
    release: Path
    bin: Path = field(init=False)
    state: Path = field(init=False)
    log: Path = field(init=False)
    home: Path = field(init=False)

    def __post_init__(self) -> None:
        self.bin = self.root / 'bin'
        self.state = self.root / 'state'
        self.log = self.root / 'shim.log'
        self.home = self.root / 'home'
        for d in (self.bin, self.state, self.home):
            d.mkdir(parents=True, exist_ok=True)
        for shim in SHIMS.iterdir():
            dest = self.bin / shim.name
            shutil.copy2(shim, dest)
            dest.chmod(0o755)
        self.log.touch()
        self.gpus(GPU_48)

    # ---- fixture knobs -------------------------------------------------
    def gpus(self, csv: str | None) -> None:
        f = self.state / 'gpus.csv'
        if csv is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(csv)

    def containers(self, rows: list[tuple[str, str, str, str]]) -> None:
        (self.state / 'containers.tsv').write_text(''.join('|'.join(r) + '\n' for r in rows))

    def flag(self, name: str, content: str = '') -> None:
        (self.state / name).write_text(content)

    def wrap_argv_loggers(self, names: list[str]) -> None:
        """Shadow common tools with wrappers that log argv, then exec the
        real binary -- a secret passed as an argument would land in the log."""
        for name in names:
            real = shutil.which(name, path='/usr/bin:/bin')
            if real is None:
                continue
            w = self.bin / name
            w.write_text(
                f'#!/bin/bash\nprintf "%s %s\\n" {name} "$*" >> "$SHIM_LOG"\nexec {real} "$@"\n'
            )
            w.chmod(0o755)

    # ---- running -------------------------------------------------------
    def env(self, **extra: str) -> dict[str, str]:
        env = {
            'PATH': f'{self.bin}:/usr/bin:/bin',
            'HOME': str(self.home),
            'LANG': 'C.UTF-8',
            'SHIM_STATE': str(self.state),
            'SHIM_LOG': str(self.log),
            'SHIM_RELEASE': str(self.release),
            'SHIM_LATEST': RELEASE,
            'OP_ARTIFACT_BASE_URL': 'https://release.test/assets',
            'OP_RAW_BASE_URL': 'https://release.test/raw',
            'CW_ARTIFACT_BASE_URL': 'https://cw.test/assets',
            'CW_RAW_BASE_URL': 'https://cw.test/raw',
        }
        env.update(extra)
        return env

    def run(
        self,
        args: list[str],
        *,
        piped: bool = False,
        xtrace: bool = False,
        script: Path = SCRIPT,
        timeout: int = 180,
        cwd: Path | None = None,
        **extra: str,
    ) -> subprocess.CompletedProcess:
        bash = ['bash', '-x'] if xtrace else ['bash']
        if piped:
            cmd = ['bash', '-c', f'cat "{script}" | {" ".join(bash)} -s -- "$@"', '_', *args]
        else:
            cmd = [*bash, str(script), *args]
        return subprocess.run(
            cmd,
            check=False,
            cwd=str(cwd or self.root),
            env=self.env(**extra),
            capture_output=True,
            text=True,
            timeout=timeout,
            start_new_session=True,
        )

    def log_lines(self, prefix: str = '') -> list[str]:
        return [ln for ln in self.log.read_text().splitlines() if ln.startswith(prefix)]

    def mutating_docker_calls(self) -> list[str]:
        f = self.state / 'mutations.log'
        return f.read_text().splitlines() if f.exists() else []
