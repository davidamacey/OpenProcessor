// Wire types of `GET /quality/readiness` (`DatasetReadiness` and its parts in
// `contracts/openapi/curation.json`, from `src/schemas/curation/quality.py`).

export type ReadinessSeverity = 'ok' | 'warn' | 'block';
export type ReadinessVerdict = 'ready' | 'warn' | 'blocked';

/** Where a check's work is: a project page and, for the Workspace, its address query. */
export interface ReadinessLink {
  /** `workspace`, `export`, `quality/audit` or `classes`. */
  section: string;
  /** The search string of the page address, without the `?`; null for none. */
  query: string | null;
}

export interface ReadinessClassRow {
  class_name: string;
  count: number;
  precision?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
}

export interface ReadinessCheck {
  id: string;
  label: string;
  severity: ReadinessSeverity;
  message: string;
  count: number | null;
  link: ReadinessLink | null;
  classes: ReadinessClassRow[];
}

export interface ClassCoverage {
  class_name: string;
  validated: number;
  machine_only: number;
  detector_only: number;
  test: number;
}

export interface DatasetReadiness {
  verdict: ReadinessVerdict;
  checks: ReadinessCheck[];
  coverage: ClassCoverage[];
  computed_at: string;
}
