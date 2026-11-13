/**
 * `GET {P}/next_steps` (`WorkflowSteps`): the curation loop in order, which
 * step is next, and the view the Workspace opens on. Served text is rendered
 * verbatim; the browser never derives a step or a default.
 */

export type WorkflowStepId =
  | 'ingest'
  | 'classes'
  | 'cluster'
  | 'activate'
  | 'label'
  | 'hotkeys'
  | 'near_dups'
  | 'export'
  | 'train'
  | 'probe'
  | 'review_disagreements'
  | 'audit';

export interface WorkflowStep {
  id: WorkflowStepId;
  label: string;
  detail: string;
  done: boolean;
  /** A project sub-path: `ingest`, `classes`, `workspace`, `quality/audit`, ... */
  section: string;
  /** Optional search string for the link, e.g. `view=clusters`. */
  query?: string | null;
  /** A write the step offers instead of a page: `adopt_detector_classes` is
   *  `POST /label_ops/adopt_detector`. Null when the step is done or has none. */
  action?: 'adopt_detector_classes' | null;
}

export interface WorkflowSteps {
  steps: WorkflowStep[];
  next: WorkflowStep | null;
  workspace_default_view: 'gallery' | 'clusters';
}
