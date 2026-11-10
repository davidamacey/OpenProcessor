"""Shared loaders for the compose contract tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / 'docker-compose.yml'


class ComposeLoader(yaml.SafeLoader):
    """Reads Compose's merge tags (`!reset`, `!override`) as the plain value they wrap.

    Docs: https://docs.docker.com/reference/compose-file/merge/#reset-value and
    #replace-value. The tags only steer how Compose merges files; a static check of one file
    reads the value itself.
    """


def _construct_wrapped(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


ComposeLoader.add_multi_constructor('!', _construct_wrapped)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.load(fh, Loader=ComposeLoader)


def _load_compose() -> dict[str, Any]:
    return _load_yaml(COMPOSE_PATH)


def _services() -> dict[str, dict[str, Any]]:
    return _load_compose()['services']


def render(env: dict[str, str], *files: str) -> dict[str, Any]:
    args = ['docker', 'compose', '-p', 'op-6b-check', '--profile', '*']
    for name in files or ('docker-compose.yml',):
        args += ['-f', str(REPO_ROOT / name)]
    done = subprocess.run(
        [*args, 'config', '--format', 'json'],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env={'PATH': '/usr/bin:/bin:/usr/local/bin', 'HOME': '/tmp', **env},
        cwd=REPO_ROOT,
    )
    return json.loads(done.stdout)['services']
