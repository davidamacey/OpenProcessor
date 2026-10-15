import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import {
  ProposalApplyController,
  type ProposalApplyDeps,
} from '$lib/assistRun/proposalApplyController.svelte';
import type { AssistApplyRequest, AssistApplyResponse } from '$lib/types_assist';

const SCOPE_NOW = { count: 400, overlap: 390, jaccard: 0.95, changed: false };

function reply(body: AssistApplyRequest): AssistApplyResponse {
  return {
    dry_run: body.dry_run !== false,
    class_name: body.class_name,
    selected: 380,
    skipped_locked: 20,
    updated: body.dry_run === false ? 380 : 0,
    updated_ids: [],
    conflicts: [],
    scope_now: SCOPE_NOW,
  };
}

function deps(over: Partial<ProposalApplyDeps> = {}): ProposalApplyDeps {
  return {
    apply: vi.fn(async (_id: string, body: AssistApplyRequest) => reply(body)),
    ...over,
  };
}

describe('ProposalApplyController', () => {
  it('asks the server for the dry-run count before anything is written', async () => {
    const d = deps();
    const c = new ProposalApplyController('asr_1', d);
    c.choose('widget_a');
    await c.preview();
    const [id, body] = vi.mocked(d.apply).mock.calls[0]!;
    expect(id).toBe('asr_1');
    expect(body).toEqual({
      class_name: 'widget_a',
      label_source: 'human',
      dry_run: true,
    });
    expect(c.dry?.selected).toBe(380);
    expect(c.dry?.skipped_locked).toBe(20);
    expect(c.applied).toBeNull();
  });

  it('applies only on the explicit call, with dry_run false, and reports the served result', async () => {
    const d = deps();
    const onapplied = vi.fn();
    const c = new ProposalApplyController('asr_1', d, onapplied);
    c.choose('widget_a');
    await c.preview();
    expect(vi.mocked(d.apply)).toHaveBeenCalledTimes(1);
    await c.apply();
    const body = vi.mocked(d.apply).mock.calls[1]![1];
    expect(body).toEqual({
      class_name: 'widget_a',
      label_source: 'human',
      dry_run: false,
    });
    expect(c.applied?.updated).toBe(380);
    expect(onapplied).toHaveBeenCalledOnce();
  });

  it('renders the served scope_changed refusal and only then offers the override', async () => {
    const refusal = new ApiError(409, 'u', {
      detail: {
        error: 'scope_changed',
        message: 'only 62% of the scope is what the run saw',
        scope_now: { count: 300, overlap: 186, jaccard: 0.62, changed: true },
      },
    });
    const d = deps({
      apply: vi.fn(async (_id: string, body: AssistApplyRequest) => {
        if (body.allow_changed_scope !== true) throw refusal;
        return reply(body);
      }),
    });
    const c = new ProposalApplyController('asr_1', d);
    c.choose('widget_a');
    await c.preview();
    expect(c.error).toBe('only 62% of the scope is what the run saw');
    expect(c.errorCode).toBe('scope_changed');
    expect(c.dry).toBeNull();
    await c.preview({ allowChangedScope: true });
    expect(c.dry?.selected).toBe(380);
    await c.apply();
    const body = vi.mocked(d.apply).mock.calls.at(-1)![1];
    expect(body).toMatchObject({ dry_run: false, allow_changed_scope: true });
    expect(c.error).toBeNull();
    expect(c.applied?.updated).toBe(380);
  });

  it('shows class_not_in_registry as served', async () => {
    const d = deps({
      apply: vi.fn(async () => {
        throw new ApiError(409, 'u', {
          detail: { error: 'class_not_in_registry', message: 'widget_c is not a class' },
        });
      }),
    });
    const c = new ProposalApplyController('asr_1', d);
    c.choose('widget_c');
    await c.preview();
    expect(c.errorCode).toBe('class_not_in_registry');
    expect(c.error).toBe('widget_c is not a class');
  });

  it('choosing another class drops the previous dry run', async () => {
    const c = new ProposalApplyController('asr_1', deps());
    c.choose('widget_a');
    await c.preview();
    c.choose('widget_b');
    expect(c.dry).toBeNull();
  });
});
