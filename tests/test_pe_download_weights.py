"""Tests for ``export/download_pe_weights.py`` (pinned PE-Core checkpoint fetch).

Both PE exporters resolve their checkpoint through this module, so a wrong
pin or a checksum check that silently passes would let a re-uploaded
checkpoint change the embedding space under an existing index. No network:
``hf_hub_download`` is replaced with a fake that records its arguments.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import download_pe_weights as dl  # noqa: E402


def _pin_for_file(path: Path, *, size: bool = True) -> dl.PECheckpointPin:
    data = path.read_bytes()
    return dl.PECheckpointPin(
        repo_id='facebook/PE-Test',
        filename=path.name,
        revision='abc123',
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data) if size else None,
    )


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Replace ``huggingface_hub.hf_hub_download``; returns the recorded call."""
    pytest.importorskip('huggingface_hub')
    import huggingface_hub

    calls: dict = {}
    blob = tmp_path / 'blob.pt'
    blob.write_bytes(b'pe-weights')

    def fake_download(**kwargs: object) -> str:
        calls.update(kwargs)
        return str(blob)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download)
    calls['_blob'] = blob
    return calls


class TestPins:
    def test_l14_336_is_pinned_to_a_commit_and_checksum(self) -> None:
        pin = dl.pin_for('PE-Core-L14-336')
        assert pin.repo_id == 'facebook/PE-Core-L14-336'
        assert pin.filename == 'PE-Core-L14-336.pt'
        assert pin.revision == 'bafb0f76541d399057e980a25947f67acec76575'
        assert pin.sha256 == '0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8'
        assert pin.size_bytes == 2_684_747_432

    def test_unknown_variant_follows_the_perception_models_convention(self) -> None:
        pin = dl.pin_for('PE-Core-B16-224')
        assert (pin.repo_id, pin.filename) == ('facebook/PE-Core-B16-224', 'PE-Core-B16-224.pt')
        assert pin.revision is None
        assert pin.sha256 is None

    def test_default_variant_matches_both_exporters(self) -> None:
        import export_pe_image_encoder
        import export_pe_text_encoder

        assert dl.DEFAULT_VARIANT == export_pe_image_encoder.PE_VARIANT
        assert dl.DEFAULT_VARIANT == export_pe_text_encoder.PE_VARIANT


class TestVerify:
    def test_matching_file_passes(self, tmp_path: Path) -> None:
        f = tmp_path / 'w.pt'
        f.write_bytes(b'x' * 1000)
        dl.verify_checkpoint(f, _pin_for_file(f))

    def test_sha_mismatch_raises(self, tmp_path: Path) -> None:
        f = tmp_path / 'w.pt'
        f.write_bytes(b'x' * 1000)
        pin = _pin_for_file(f, size=False)
        f.write_bytes(b'y' * 1000)
        with pytest.raises(dl.ChecksumMismatchError, match='SHA-256'):
            dl.verify_checkpoint(f, pin)

    def test_truncated_file_fails_fast_on_size(self, tmp_path: Path) -> None:
        f = tmp_path / 'w.pt'
        f.write_bytes(b'x' * 1000)
        pin = _pin_for_file(f)
        f.write_bytes(b'x' * 10)
        with pytest.raises(dl.ChecksumMismatchError, match='truncated'):
            dl.verify_checkpoint(f, pin)

    def test_unpinned_variant_verifies_nothing(self, tmp_path: Path) -> None:
        f = tmp_path / 'w.pt'
        f.write_bytes(b'anything')
        dl.verify_checkpoint(f, dl.pin_for('PE-Core-S16-384'))

    def test_streaming_hash_matches_hashlib(self, tmp_path: Path) -> None:
        f = tmp_path / 'w.pt'
        data = bytes(range(256)) * 300
        f.write_bytes(data)
        assert dl.sha256_file(f, chunk_size=1000) == hashlib.sha256(data).hexdigest()


class TestDownload:
    def test_downloads_the_pinned_revision(self, fake_hub: dict, monkeypatch) -> None:
        monkeypatch.setitem(dl.PINS, 'PE-Core-L14-336', _pin_for_file(fake_hub['_blob']))
        path = dl.download_checkpoint('PE-Core-L14-336')
        assert path == fake_hub['_blob']
        assert fake_hub['revision'] == 'abc123'
        assert fake_hub['repo_id'] == 'facebook/PE-Test'

    def test_a_corrupt_download_is_rejected(self, fake_hub: dict, monkeypatch) -> None:
        pin = _pin_for_file(fake_hub['_blob'])
        monkeypatch.setitem(
            dl.PINS,
            'PE-Core-L14-336',
            dl.PECheckpointPin(pin.repo_id, pin.filename, pin.revision, '0' * 64, None),
        )
        with pytest.raises(dl.ChecksumMismatchError):
            dl.download_checkpoint('PE-Core-L14-336')

    def test_revision_override_drops_the_checksum(self, fake_hub: dict) -> None:
        # The real pin's checksum cannot match the fake blob; an override
        # must not try to apply it.
        path = dl.download_checkpoint('PE-Core-L14-336', revision='deadbeef')
        assert path == fake_hub['_blob']
        assert fake_hub['revision'] == 'deadbeef'

    def test_no_verify_skips_the_checksum(self, fake_hub: dict) -> None:
        assert dl.download_checkpoint('PE-Core-L14-336', verify=False) == fake_hub['_blob']


class TestResolve:
    def test_explicit_path_is_verified_against_the_pin(self, tmp_path: Path) -> None:
        f = tmp_path / 'PE-Core-L14-336.pt'
        f.write_bytes(b'not the real checkpoint')
        with pytest.raises(dl.ChecksumMismatchError):
            dl.resolve_checkpoint('PE-Core-L14-336', f)
        assert dl.resolve_checkpoint('PE-Core-L14-336', f, verify=False) == f

    def test_missing_explicit_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            dl.resolve_checkpoint('PE-Core-L14-336', tmp_path / 'nope.pt')


class TestCli:
    def test_verify_file_mode_exit_codes(self, tmp_path: Path, capsys) -> None:
        f = tmp_path / 'w.pt'
        f.write_bytes(b'wrong bytes')
        assert dl.main(['--verify-file', str(f)]) == 1
        assert dl.main(['--verify-file', str(f), '--variant', 'PE-Core-S16-384']) == 0
        assert capsys.readouterr().out.strip().endswith('w.pt')
