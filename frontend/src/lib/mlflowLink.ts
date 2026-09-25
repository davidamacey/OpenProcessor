/**
 * T1 (visual audit 2026-09-24): the "MLflow" dashboard link was built as
 * `http://<host>:5000`, an MLflow that has none of the runs. Runs link to
 * whatever MLflow the backend put in each run's served `mlflow_run_url`,
 * so the dashboard link now uses that URL's origin. An explicit
 * `PUBLIC_MLFLOW_URL` still wins; with neither, there is no link.
 *
 * TODO(backend): serve the public MLflow base URL directly (e.g. on
 * `/health` or a config endpoint, from `OP_MLFLOW_PUBLIC_URL`) so pages
 * without a run list (e.g. /bakeoff) can link to it too.
 */
export function mlflowBaseUrl(
  runUrls: ReadonlyArray<string | null | undefined>,
  envUrl?: string | null,
): string | null {
  if (envUrl) return envUrl;
  for (const u of runUrls) {
    if (!u) continue;
    try {
      const parsed = new URL(u);
      if (parsed.protocol === 'http:' || parsed.protocol === 'https:')
        return parsed.origin;
    } catch {
      // not an absolute URL; try the next run
    }
  }
  return null;
}
