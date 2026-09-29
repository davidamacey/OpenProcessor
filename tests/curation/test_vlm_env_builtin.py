"""The ``env`` built-in endpoint (W9.2): the operator's own ``OP_VLM_*``
settings as a read-only endpoint, so a stack that never opens the picker
keeps working exactly as before."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.services.labeling.vlm_catalog import load_catalog
from src.services.labeling.vlm_endpoint_body import VlmProbeRecord
from src.services.labeling.vlm_endpoints import env_endpoint, probe_fingerprint


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        'OP_VLM_URL',
        'OP_VLM_MODEL',
        'OP_VLM_API_KEY',
        'OP_VLM_MAX_IMAGES_PER_CALL',
        'OP_VLM_OPEN_IMAGES_PER_CALL',
    ):
        monkeypatch.delenv(key, raising=False)


def _probe(root: str | None = None, **over: object) -> dict:
    record = VlmProbeRecord(ok=True, probed_at='2026-09-28T12:00:00+00:00', root=root, **over)  # type: ignore[arg-type]
    return {'fingerprint': None, 'record': record.model_dump()}


def test_no_url_means_no_builtin() -> None:
    assert env_endpoint() is None


def test_the_builtin_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1/')
    monkeypatch.setenv('OP_VLM_MODEL', 'local-vlm')
    monkeypatch.setenv('OP_VLM_MAX_IMAGES_PER_CALL', '4')
    endpoint = env_endpoint()
    assert endpoint is not None
    assert (endpoint.name, endpoint.source, endpoint.revision) == ('env', 'env', None)
    assert endpoint.body.base_url == 'http://vlm:8000/v1'  # trailing slash stripped
    assert endpoint.body.model == 'local-vlm'
    assert endpoint.body.max_images_per_call == 4
    assert endpoint.body.allow_external is True  # the operator's own choice
    assert endpoint.body.api_key_ref == 'env:OP_VLM_API_KEY'


def test_a_key_file_named_env_takes_the_place_of_the_environment_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / 'env').write_text('file-key')
    monkeypatch.setenv('OP_VLM_SECRETS_DIR', str(tmp_path))
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'local-vlm')
    endpoint = env_endpoint()
    assert endpoint is not None
    assert endpoint.body.api_key_ref == 'secret:env'


def test_the_ref_follows_the_body_so_an_env_change_is_a_new_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'a')
    first = env_endpoint()
    assert first is not None
    assert first.ref == env_endpoint().ref  # type: ignore[union-attr]
    monkeypatch.setenv('OP_VLM_MODEL', 'b')
    second = env_endpoint()
    assert second is not None
    assert second.ref != first.ref
    assert first.ref.startswith('env@')


def test_the_resolved_model_is_the_probe_root_when_the_probe_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'local-vlm')
    bare = env_endpoint()
    assert bare is not None
    assert bare.model_id == 'local-vlm'
    assert bare.status == 'unprobed'

    first = load_catalog()[0]
    doc = _probe(first.hf_repo)
    doc['fingerprint'] = probe_fingerprint(bare.body)
    probed = env_endpoint(doc)
    assert probed is not None
    assert probed.model_id == first.hf_repo
    assert probed.status == 'ready'
    # the catalog entry is recognised by the served root
    assert probed.body.catalog_id == first.id


def test_a_probe_of_a_different_body_is_not_attributed_to_the_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'local-vlm')
    bare = env_endpoint()
    assert bare is not None
    stale = _probe('org/old')
    stale['fingerprint'] = 'deadbeefdead'  # taken when the URL or model was different
    endpoint = env_endpoint(stale)
    assert endpoint is not None
    assert endpoint.last_probe is None
    assert endpoint.model_id == 'local-vlm'
    assert endpoint.status == 'unprobed'


def test_json_mode_auto_follows_the_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'local-vlm')
    bare = env_endpoint()
    assert bare is not None
    assert bare.json_mode_on is True  # unprobed: on, as it always was
    doc = _probe(json_mode_supported=False)
    doc['fingerprint'] = probe_fingerprint(bare.body)
    off = env_endpoint(doc)
    assert off is not None
    assert off.json_mode_on is False
