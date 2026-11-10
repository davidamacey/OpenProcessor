"""Pin the built-image live-harness overlay: no source mounts, no builds."""

from __future__ import annotations

from compose._helpers import REPO_ROOT, render


def test_built_overlay_has_no_source_mounts_or_builds() -> None:
    files = ('docker/test/compose.yml', 'docker/test/compose.built.yml')
    cfg = render({'OP_TEST_IMAGE_TAG': 'x1'}, *files)
    for name, svc in cfg.items():
        if name in {'opensearch', 'postgres', 'valkey'}:
            continue
        assert 'build' not in svc, name
        assert svc['image'].endswith(':x1'), name
        for vol in svc.get('volumes') or []:
            src = vol.get('source', '')
            assert not src.startswith(str(REPO_ROOT / 'src')), (name, src)
            assert not src.startswith(str(REPO_ROOT / 'scripts')), (name, src)
            assert not src.endswith('/VERSION'), (name, src)
