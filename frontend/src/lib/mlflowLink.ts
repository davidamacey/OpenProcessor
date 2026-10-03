/**
 * Per-run MLflow links (`mlflow_run_url`) and other served external URLs
 * (a model's `license_url`) render only when they are absolute http(s).
 * The dashboard link of the Resources menu is a served `resource_links`
 * entry instead (see `$lib/resourceLinks`).
 */
/** A served URL as an `href`, only when it is an absolute http(s) URL:
 *  Svelte does not sanitize `href`, so a `javascript:` value would reach
 *  the DOM verbatim. `null` means render the text without a link. */
export function externalHref(u: string | null | undefined): string | null {
  if (!u) return null;
  try {
    const { protocol } = new URL(u);
    return protocol === 'http:' || protocol === 'https:' ? u : null;
  } catch {
    return null;
  }
}
