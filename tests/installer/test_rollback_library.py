"""Rollback across a change of the installer's own library."""

from __future__ import annotations

from typing import TYPE_CHECKING

from installer_harness import PROJECT
from test_lifecycle import install, reset_mutations


if TYPE_CHECKING:
    from installer_harness import Shimmed


def test_rollback_to_a_release_whose_library_lacks_newer_functions(shimmed: Shimmed) -> None:
    # The restored files include scripts/lib/image_keys.sh of the OLD release; a
    # rollback must keep using the running installer's own already-loaded
    # functions (compose_image_refs did not exist before the export image).
    assert install(shimmed).returncode == 0
    assert install(shimmed, '--port-base', '4970').returncode == 0
    inst = shimmed.root / 'inst'
    backup = sorted((inst / 'backups').iterdir())[-1]
    lib = backup / 'scripts' / 'lib' / 'image_keys.sh'
    text = lib.read_text()
    start = text.index('# compose_image_refs')
    old = text[:start].replace('[[ -n "${_IMAGE_KEYS_SH_LOADED:-}" ]] && return 0\n', '')
    assert 'compose_image_refs' not in old
    lib.write_text(old)
    reset_mutations(shimmed)
    result = shimmed.run(['--rollback', '--unattended', '--dir', 'inst'])
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'command not found' not in result.stderr
    assert 'compose_image_refs' not in (inst / 'scripts' / 'lib' / 'image_keys.sh').read_text()
    assert [
        ln
        for ln in shimmed.mutating_docker_calls()
        if ln.endswith(' up -d --remove-orphans') and f'-p {PROJECT} ' in ln
    ]
