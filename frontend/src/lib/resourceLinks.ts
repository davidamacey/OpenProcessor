/**
 * What the Resources menu and the dashboards row show: exactly the served
 * `GET /settings` `resource_links`, in the served order, behind one
 * client-owned entry. Nothing is guessed: no host, port or path is built
 * here, and an empty or failed settings read leaves only Documentation.
 */
import { externalHref } from '$lib/mlflowLink';
import type { ResourceLink } from '$lib/curationSettings';

/** The one client-owned entry: the bundled documentation is this repo's own
 *  docs container, proxied same-origin by nginx.conf (`^~ /cropwright/`),
 *  so the backend knows nothing of it and never serves it. */
export const DOCS_ENTRY = {
  id: 'docs',
  label: 'Documentation',
  href: '/cropwright/',
} as const;

/** A served URL as an `href`: an absolute http(s) URL, or a root-relative
 *  path with a single leading slash (a docs entry, resolved against our own
 *  origin). Svelte does not sanitize `href`, so `javascript:`, `data:`,
 *  protocol-relative `//host` and anything else is refused (null). */
export function safeHref(u: string | null | undefined): string | null {
  if (!u) return null;
  if (u.startsWith('/')) return /^\/[^/\\]/.test(u) || u === '/' ? u : null;
  return externalHref(u);
}

export interface ResourceView {
  id: string;
  label: string;
  kind: ResourceLink['kind'] | 'client';
  /** null = render a muted row, not an anchor. */
  href: string | null;
  /** Why a row has no link ("not configured"); empty when it has one. */
  note: string;
  /** Served help text, shown as the row's tooltip. */
  hint: string;
  /** The server reports the configured service is not answering. */
  notRunning: boolean;
}

export function resourceViews(served: readonly ResourceLink[]): ResourceView[] {
  return [
    {
      id: DOCS_ENTRY.id,
      label: DOCS_ENTRY.label,
      kind: 'client',
      href: DOCS_ENTRY.href,
      note: '',
      hint: '',
      notRunning: false,
    },
    ...served.map((l): ResourceView => {
      const href = safeHref(l.url);
      return {
        id: l.id,
        label: l.label,
        kind: l.kind,
        href,
        note: href ? '' : l.url ? 'link unavailable' : 'not configured',
        hint: l.hint,
        notRunning: href !== null && l.reachable === false,
      };
    }),
  ];
}
