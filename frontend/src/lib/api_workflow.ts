/** The curation loop's next steps and the Workspace's default view. */
import { apiFetch, scoped } from '$lib/api';
import type { WorkflowSteps } from '$lib/types_workflow';

export function getNextSteps(signal?: AbortSignal): Promise<WorkflowSteps> {
  return apiFetch<WorkflowSteps>(`${scoped()}/next_steps`, { signal });
}
