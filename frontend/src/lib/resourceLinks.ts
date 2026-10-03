/**
 * Links the app offers to everything else behind its origin: the bundled
 * docs and the API's own interactive docs are same-origin nginx proxies
 * (nginx.conf), so these paths are constants owned by this repo. The
 * monitoring dashboards are NOT: they come only from the served
 * `GET /settings` `monitoring_links`, and MLflow from `mlflowBaseUrl`.
 */
import { externalHref } from '$lib/mlflowLink';
import type { MonitoringLinksServed } from '$lib/curationSettings';

export interface ResourceLink {
  key: string;
  label: string;
  href: string;
}

/** Same-origin proxy paths; keep in step with nginx.conf. */
export const SAME_ORIGIN_RESOURCES: readonly ResourceLink[] = [
  { key: 'docs', label: 'Documentation', href: '/cropwright/' },
  { key: 'swagger', label: 'API reference (Swagger UI)', href: '/docs' },
  { key: 'redoc', label: 'API reference (ReDoc)', href: '/redoc' },
  { key: 'openapi', label: 'OpenAPI JSON', href: '/openapi.json' },
];

const MONITORING = [
  { key: 'grafana', label: 'Grafana' },
  { key: 'prometheus', label: 'Prometheus' },
  { key: 'opensearch_dashboards', label: 'OpenSearch' },
] as const;

/** Served monitoring dashboards that carry an absolute http(s) URL. */
export function monitoringResourceLinks(
  served: MonitoringLinksServed | null | undefined,
): ResourceLink[] {
  return MONITORING.flatMap((s) => {
    const href = externalHref(served?.[s.key]);
    return href ? [{ key: s.key, label: s.label, href }] : [];
  });
}

export function resourceLinks(
  served: MonitoringLinksServed | null | undefined,
  mlflowHref: string | null,
): ResourceLink[] {
  return [
    ...SAME_ORIGIN_RESOURCES,
    ...monitoringResourceLinks(served),
    ...(mlflowHref ? [{ key: 'mlflow', label: 'MLflow', href: mlflowHref }] : []),
  ];
}
