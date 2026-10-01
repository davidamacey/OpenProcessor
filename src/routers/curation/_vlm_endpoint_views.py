"""Builders for the VLM endpoint wire shapes (W9, §7.8): summaries, docs and
the form schema. No routes live here."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.routers.curation._vlm_endpoint_models import (
    LOCALITY_LABELS,
    SOURCE_LABELS,
    STATUS_LABELS,
    Choice,
    SecretRef,
    VlmEndpointDoc,
    VlmEndpointFieldSchema,
    VlmEndpointGroup,
    VlmEndpointLabels,
    VlmEndpointSchema,
    VlmEndpointSummary,
    VlmProbeResult,
)
from src.services.labeling.vlm_endpoint_body import FIELD_RANGES
from src.services.labeling.vlm_endpoints import api_key_present, list_secret_refs
from src.services.labeling.vlm_url_policy import (
    Locality,
    acompute_locality,
    external_warning,
    sends_images_externally,
)


if TYPE_CHECKING:
    from src.routers.curation._config_common_models import ValidationReport
    from src.services.labeling.vlm_endpoint_body import VlmProbeRecord
    from src.services.labeling.vlm_endpoints import VlmEndpoint

NO_KEY_HELP = (
    'A reference to a key file, never the key itself. Write the file on the host with '
    '`openprocessor vlm key set <slug>`, then pick it here.'
)


def probe_wire(record: VlmProbeRecord | None) -> VlmProbeResult | None:
    return None if record is None else VlmProbeResult.model_validate(record.model_dump())


async def locality_of(endpoint: VlmEndpoint, *, cached: bool = True) -> Locality | None:
    """``None`` when the URL is unusable (it then has no locality)."""
    try:
        return await acompute_locality(endpoint.body.base_url, cached=cached)
    except ValueError:
        return None


def _key_present(endpoint: VlmEndpoint) -> bool:
    return api_key_present(endpoint.body.api_key_ref, is_env_builtin=endpoint.source == 'env')


async def summary_of(endpoint: VlmEndpoint, *, active_in: list[str]) -> VlmEndpointSummary:
    locality = await locality_of(endpoint)
    sends = locality is not None and sends_images_externally(locality)
    return VlmEndpointSummary(
        name=endpoint.name,
        source=endpoint.source,
        read_only=endpoint.source == 'env',
        revision=endpoint.revision,
        etag=endpoint.etag,
        description=endpoint.description,
        base_url=endpoint.body.base_url,
        model=endpoint.model_id,
        catalog_id=endpoint.body.catalog_id,
        locality=locality,
        sends_images_externally=sends,
        warning=external_warning(endpoint.body.base_url, locality) if locality else None,
        api_key_ref=endpoint.body.api_key_ref,
        api_key_present=_key_present(endpoint),
        status=endpoint.status,
        last_probe_at=endpoint.last_probe.probed_at if endpoint.last_probe else None,
        active_in=active_in,
        updated_at=endpoint.updated_at,
    )


async def doc_of(
    endpoint: VlmEndpoint, *, active_in: list[str], validation: ValidationReport | None = None
) -> VlmEndpointDoc:
    locality = await locality_of(endpoint)
    sends = locality is not None and sends_images_externally(locality)
    return VlmEndpointDoc(
        name=endpoint.name,
        source=endpoint.source,
        read_only=endpoint.source == 'env',
        revision=endpoint.revision,
        etag=endpoint.etag,
        description=endpoint.description,
        body=endpoint.body,
        api_key_present=_key_present(endpoint),
        locality=locality,
        sends_images_externally=sends,
        warning=external_warning(endpoint.body.base_url, locality) if locality else None,
        last_probe=probe_wire(endpoint.last_probe),
        created_at=endpoint.created_at,
        updated_at=endpoint.updated_at,
        cloned_from=endpoint.cloned_from,
        active_in=active_in,
        validation=validation,
    )


def secret_refs_wire() -> list[SecretRef]:
    return [
        SecretRef(
            ref=ref,
            present=True,
            choice=Choice(id=ref, label=ref.removeprefix('secret:')),
        )
        for ref in list_secret_refs()
    ]


def labels_wire() -> VlmEndpointLabels:
    return VlmEndpointLabels(status=STATUS_LABELS, locality=LOCALITY_LABELS, source=SOURCE_LABELS)


def endpoint_schema() -> VlmEndpointSchema:
    lo, hi = FIELD_RANGES['max_images_per_call']
    rows = [
        VlmEndpointFieldSchema(
            field='base_url',
            label='Endpoint URL',
            group='connection',
            type='string',
            default='',
            help='The OpenAI-compatible base URL, for example http://vlm:8000/v1.',
        ),
        VlmEndpointFieldSchema(
            field='model',
            label='Model id',
            group='connection',
            type='string',
            default='',
            help='The model id sent with every request.',
        ),
        VlmEndpointFieldSchema(
            field='api_key_ref',
            label='API key',
            group='connection',
            type='string',
            default=None,
            choices_from='secret_refs',
            empty_choice=Choice(id=None, label='No key'),
            help=NO_KEY_HELP,
        ),
        VlmEndpointFieldSchema(
            field='max_images_per_call',
            label='Images per call',
            group='limits',
            type='int',
            default=8,
            min=lo,
            max=hi,
            help='Hard cap on images in one request; must not exceed what the server accepts.',
        ),
        VlmEndpointFieldSchema(
            field='open_images_per_call',
            label='Images per open-vocabulary call',
            group='limits',
            type='int',
            default=3,
            min=FIELD_RANGES['open_images_per_call'][0],
            max=FIELD_RANGES['open_images_per_call'][1],
            advanced=True,
            help='Smaller chunks for the denser open-vocabulary prompt; empty uses min(3, cap).',
        ),
        VlmEndpointFieldSchema(
            field='timeout_s',
            label='Request timeout (seconds)',
            group='limits',
            type='float',
            default=240.0,
            min=FIELD_RANGES['timeout_s'][0],
            max=FIELD_RANGES['timeout_s'][1],
            advanced=True,
        ),
        VlmEndpointFieldSchema(
            field='requests_per_second',
            label='Requests per second',
            group='limits',
            type='float',
            default=500.0,
            min=FIELD_RANGES['requests_per_second'][0],
            max=FIELD_RANGES['requests_per_second'][1],
            advanced=True,
            help='Per process.',
        ),
        VlmEndpointFieldSchema(
            field='json_mode',
            label='JSON mode',
            group='behavior',
            type='enum',
            default='auto',
            enum=[
                Choice(id='auto', label='Automatic (from the last test)'),
                Choice(id='on', label='Always on'),
                Choice(id='off', label='Off'),
            ],
            advanced=True,
            help='Ask the server for JSON-only replies. Some servers reject it.',
        ),
        VlmEndpointFieldSchema(
            field='allow_external',
            label='This endpoint is outside this deployment',
            group='behavior',
            type='bool',
            default=False,
            help=(
                'Required for an endpoint outside this deployment: crops are sent to it. '
                'Tick it only if you accept that.'
            ),
        ),
        VlmEndpointFieldSchema(
            field='catalog_id',
            label='Local catalog model',
            group='behavior',
            type='string',
            default=None,
            choices_from='vlm_catalog',
            empty_choice=Choice(id=None, label='Not from the catalog'),
            advanced=True,
            help='Only used to show what is known about the model and to pair it with profiles.',
        ),
    ]
    return VlmEndpointSchema(
        fields=rows,
        groups=[
            VlmEndpointGroup(id='connection', label='Connection'),
            VlmEndpointGroup(id='limits', label='Limits'),
            VlmEndpointGroup(id='behavior', label='Behavior'),
        ],
    )


__all__ = [
    'doc_of',
    'endpoint_schema',
    'labels_wire',
    'locality_of',
    'probe_wire',
    'secret_refs_wire',
    'summary_of',
]
