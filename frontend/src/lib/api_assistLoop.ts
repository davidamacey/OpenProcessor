/**
 * The auto-label loop (`{prefix}/assist/loops`): start (a dry run by default),
 * list, read, the node table and a node's members, cancel, resume and rollback.
 */
import { apiFetch, qs, scoped } from '$lib/api';
import type {
  AssistLoopRequest,
  AssistLoopState,
  LoopMembersPage,
  LoopNodePage,
  LoopRollbackResponse,
} from '$lib/types_assistLoop';

export function createAssistLoop(
  body: AssistLoopRequest,
  signal?: AbortSignal,
): Promise<AssistLoopState> {
  return apiFetch<AssistLoopState>(
    `${scoped()}/assist/loops`,
    { method: 'POST', body: JSON.stringify(body) },
    signal,
  );
}

export function listAssistLoops(
  opts: { active?: boolean; limit?: number } = {},
  signal?: AbortSignal,
): Promise<{ loops: AssistLoopState[] }> {
  return apiFetch<{ loops: AssistLoopState[] }>(
    `${scoped()}/assist/loops${qs({ active: opts.active, limit: opts.limit })}`,
    {},
    signal,
  );
}

export function getAssistLoop(
  loopId: string,
  signal?: AbortSignal,
): Promise<AssistLoopState> {
  return apiFetch<AssistLoopState>(
    `${scoped()}/assist/loops/${encodeURIComponent(loopId)}`,
    {},
    signal,
  );
}

export function listLoopNodes(
  loopId: string,
  opts: { level?: number; decision?: string; after?: string | null; limit?: number } = {},
  signal?: AbortSignal,
): Promise<LoopNodePage> {
  return apiFetch<LoopNodePage>(
    `${scoped()}/assist/loops/${encodeURIComponent(loopId)}/nodes${qs({
      level: opts.level,
      decision: opts.decision,
      after: opts.after,
      limit: opts.limit,
    })}`,
    {},
    signal,
  );
}

export function listLoopNodeMembers(
  loopId: string,
  nodePath: string,
  opts: { after?: string | null; limit?: number } = {},
  signal?: AbortSignal,
): Promise<LoopMembersPage> {
  return apiFetch<LoopMembersPage>(
    `${scoped()}/assist/loops/${encodeURIComponent(loopId)}/nodes/${encodeURIComponent(nodePath)}/members${qs(
      {
        after: opts.after,
        limit: opts.limit,
      },
    )}`,
    {},
    signal,
  );
}

export function cancelAssistLoop(loopId: string): Promise<AssistLoopState> {
  return apiFetch<AssistLoopState>(
    `${scoped()}/assist/loops/${encodeURIComponent(loopId)}/cancel`,
    { method: 'POST' },
  );
}

export function resumeAssistLoop(loopId: string): Promise<AssistLoopState> {
  return apiFetch<AssistLoopState>(
    `${scoped()}/assist/loops/${encodeURIComponent(loopId)}/resume`,
    { method: 'POST' },
  );
}

export function rollbackAssistLoop(
  loopId: string,
  body: { node_paths: string[] | null; dry_run: boolean },
): Promise<LoopRollbackResponse> {
  return apiFetch<LoopRollbackResponse>(
    `${scoped()}/assist/loops/${encodeURIComponent(loopId)}/rollback`,
    {
      method: 'POST',
      body: JSON.stringify(body),
    },
  );
}
