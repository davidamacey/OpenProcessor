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
  | 'review_disagreements';

export interface WorkflowStep {
  id: WorkflowStepId;
  label: string;
  detail: string;
  done: boolean;
  /** A project sub-path: `ingest`, `classes`, `workspace`, `quality/near-duplicates`, ... */
  section: string;
  /** Optional search string for the link, e.g. `view=clusters`. */
  query?: string | null;
}

export interface WorkflowSteps {
  steps: WorkflowStep[];
  next: WorkflowStep | null;
  workspace_default_view: 'gallery' | 'clusters';
}
