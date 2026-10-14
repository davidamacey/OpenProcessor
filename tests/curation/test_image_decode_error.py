"""Issue #297: an undecodable image reports the decode error, not the patched-open
side effect. Ultralytics replaces ``PIL.Image.open`` with a wrapper that, when Pillow
fails, tries to import ``pi_heif``; where that package is absent the wrapper raises
``No module named 'pi_heif'`` and hides Pillow's own error."""

from __future__ import annotations

import io
import sys

import pytest
from PIL import Image, UnidentifiedImageError

from src.services.curation.image_decode import ImageDecodeError, decode_image


@pytest.fixture
def ultralytics_open_without_heif(monkeypatch: pytest.MonkeyPatch) -> None:
    """The container's situation: Ultralytics' ``Image.open`` is installed and
    ``pi_heif`` cannot be imported."""
    patches = pytest.importorskip('ultralytics.utils.patches')
    checks = pytest.importorskip('ultralytics.utils.checks')
    monkeypatch.setattr(patches, '_pil_plugins_registered', False)
    monkeypatch.setattr(checks, 'check_requirements', lambda *_a, **_k: True)
    monkeypatch.setitem(sys.modules, 'pi_heif', None)  # `from pi_heif import ...` raises
    monkeypatch.setattr(Image, 'open', patches.image_open)


def _png(size: int = 4, *, noise: bool = False) -> bytes:
    buf = io.BytesIO()
    image = (
        Image.effect_noise((size, size), 80).convert('RGB')
        if noise
        else Image.new('RGB', (size, size), (1, 2, 3))
    )
    image.save(buf, format='PNG')
    return buf.getvalue()


@pytest.mark.usefixtures('ultralytics_open_without_heif')
def test_text_bytes_report_a_stable_decode_reason() -> None:
    with pytest.raises(ImageDecodeError) as err:
        decode_image(b'this is a text file named .jpg')

    assert 'pi_heif' not in str(err.value)
    assert str(err.value) == 'not a decodable image'


@pytest.mark.usefixtures('ultralytics_open_without_heif')
def test_the_original_pillow_error_stays_chained() -> None:
    with pytest.raises(ImageDecodeError) as err:
        decode_image(b'garbage')

    chain: list[BaseException] = []
    link: BaseException | None = err.value.__cause__
    while link is not None:
        chain.append(link)
        link = link.__context__
    assert any(isinstance(e, UnidentifiedImageError) for e in chain)


@pytest.mark.usefixtures('ultralytics_open_without_heif')
def test_a_valid_image_still_decodes() -> None:
    assert decode_image(_png()).pil.size == (4, 4)


def test_a_truncated_image_keeps_its_own_truthful_message() -> None:
    whole = _png(64, noise=True)
    with pytest.raises(ImageDecodeError) as err:
        decode_image(whole[: len(whole) // 2])

    assert 'pi_heif' not in str(err.value)
    assert str(err.value) != 'not a decodable image'  # Pillow identified it; say what failed
