"""Step ``embedding_spaces``: the three ``pe_core_l14_336_v1`` spaces of every loaded project.

A 0.6.0 project already has these vectors under today's field names, so the registry records
what is there: ``item`` on ``pe_embedding`` of the items index (state ``embedding_state``),
``box`` on the nested ``region_box_embeddings.embedding`` of the items index, ``frame`` on
``pe_embedding`` of the images index (no state field). The rows are the ones a project create
writes (``default_space_rows``, lane ES-A) with ``created_by = 'upgrade'`` and one difference:
``knn_method`` is read from each live index mapping, never from configuration, because an old
project's fields are faiss, a new one's Lucene, and a field's engine is fixed at creation.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from src.db.repos import embedding_spaces as repo
from src.domain.embedding_spaces import KnnMethodSpec, Target
from src.services.embedding_spaces.catalog import load_catalog
from src.services.embedding_spaces.spaces import default_space_rows


if TYPE_CHECKING:
    import uuid

    from src.services.upgrade.from_060.steps import StepContext

ORDER = 20
MODEL_KEY = 'pe_core_l14_336'
MODEL_VERSION = 1
CREATED_BY = 'upgrade'


class MappingError(RuntimeError):
    """A live index lacks, or does not describe as expected, the vector field of a 0.6.0 project."""


def field_of(mappings: dict[str, Any], vector_field: str) -> dict[str, Any]:
    """The mapping entry of ``vector_field`` (a dotted path through ``properties``)."""
    node: Any = mappings
    for part in vector_field.split('.'):
        node = node.get('properties', {}).get(part)
        if node is None:
            raise MappingError(f'the live mapping has no field {vector_field}')
    return dict(node)


def knn_method_of(field: dict[str, Any]) -> KnnMethodSpec:
    """The ``KnnMethodSpec`` of a ``knn_vector`` mapping entry: the method as it is."""
    method = field.get('method') or {}
    params = method.get('parameters') or {}
    raw: dict[str, Any] = {
        'engine': method.get('engine', 'faiss'),
        'space_type': method.get('space_type', 'cosinesimil'),
        'mode': field.get('mode'),
        'compression_level': field.get('compression_level'),
        'data_type': field.get('data_type'),
    }
    for key in ('m', 'ef_construction'):
        if key in params:
            raw[key] = int(params[key])
    try:
        return KnnMethodSpec(**raw)
    except ValidationError as exc:
        raise MappingError(
            f'the live k-NN method {method} is not a registry method: {exc}'
        ) from exc


async def _model_version(ctx: StepContext) -> repo.ModelVersionRow:
    """The catalog row, inserted from the shipped catalog when ``db-migrate`` has not synced it."""

    def find(sync_session: Any) -> repo.ModelVersionRow | None:
        found = repo.get_model_version(sync_session, MODEL_KEY, MODEL_VERSION)
        if found is None:
            wanted = [
                e for e in load_catalog() if (e.model_key, e.version) == (MODEL_KEY, MODEL_VERSION)
            ]
            if not wanted:
                return None
            found = repo.insert_model_version(sync_session, wanted[0])
        return found

    row = await ctx.session.run_sync(find)
    if row is None:
        raise MappingError(f'the embedding catalog has no {MODEL_KEY} version {MODEL_VERSION}')
    return row


async def _methods(
    ctx: StepContext, indexes: dict[str, str], rows: list[repo.NewSpace]
) -> list[repo.NewSpace]:
    """``rows`` with each space's ``knn_method`` (and a check of ``dim``) from its live mapping."""
    mappings = {
        role: (await ctx.reader.definition(indexes[role]))[0] for role in ('items', 'images')
    }
    out = []
    for row in rows:
        field = field_of(mappings[row.index_role], row.vector_field)
        if int(field['dimension']) != row.dim:
            msg = (
                f'{row.vector_field} has dimension {field["dimension"]}, the catalog says {row.dim}'
            )
            raise MappingError(msg)
        out.append(dataclasses.replace(row, knn_method=knn_method_of(field)))
    return out


async def run(ctx: StepContext) -> None:
    version = await _model_version(ctx)
    for pd in ctx.parsed.projects:
        project_id: uuid.UUID = ctx.project_ids[pd.project.slug]
        rows = default_space_rows(project_id, version, created_by=CREATED_BY)
        await repo.add_spaces(ctx.session, await _methods(ctx, pd.project.indexes, rows))
        ctx.report.add('embedding_spaces', len(rows))


async def verify(ctx: StepContext) -> list[str]:
    """A space per (project, target) and the recorded engine equals the live mapping."""
    problems: list[str] = []
    for pd in ctx.parsed.projects:
        project = pd.project
        stored = {
            s.target: s
            for s in await repo.list_spaces(ctx.session, ctx.project_ids[project.slug])
            if s.space_key.startswith(MODEL_KEY)
        }
        for target in (Target.ITEM, Target.BOX, Target.FRAME):
            space = stored.get(target)
            if space is None:
                problems.append(f'{project.slug}: no {target.value} space')
                continue
            mappings, _ = await ctx.reader.definition(project.indexes[space.index_role])
            live = knn_method_of(field_of(mappings, space.vector_field))
            if space.knn_method.engine != live.engine:
                problems.append(
                    f'{project.slug}/{target.value}: recorded engine {space.knn_method.engine} '
                    f'but the live mapping says {live.engine}'
                )
    return problems
