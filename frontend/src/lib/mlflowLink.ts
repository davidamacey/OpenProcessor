/**
 * The MLflow dashboard link is the served `mlflow_public_url` of
 * `GET /health` (`OP_MLFLOW_PUBLIC_URL`, null when unset), and a run links
 * to its own served `mlflow_run_url` (null unless that public URL is set).
 * Nothing is derived from a run URL's origin, a port or an env override.
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
