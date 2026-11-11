"""POST /embed/image only embeds; it must not advertise an index flag it cannot honour."""

from src.routers.embed import router
from src.schemas.embed import ImageEmbeddingResponse


def _image_route():
    return next(r for r in router.routes if getattr(r, 'path', '') == '/embed/image')


def test_embed_image_has_no_index_or_image_id_parameter() -> None:
    names = {p.name for p in _image_route().dependant.query_params}
    assert names == {'use_cache'}


def test_embed_image_response_has_no_indexed_field() -> None:
    assert set(ImageEmbeddingResponse.model_fields) == {'embedding', 'embedding_norm', 'status'}
