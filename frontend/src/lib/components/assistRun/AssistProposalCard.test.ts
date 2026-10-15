import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { ApiError } from '$lib/api_errors';
import type { AssistApplyRequest, AssistApplyResponse } from '$lib/types_assist';
import { assistRun, proposal } from '$lib/test/fixtures/assist';

const applyAssistRun =
  vi.fn<(id: string, body: AssistApplyRequest) => Promise<AssistApplyResponse>>();
vi.mock('$lib/api_assist', () => ({
  applyAssistRun: (id: string, body: AssistApplyRequest) => applyAssistRun(id, body),
}));

import AssistProposalCard from '$components/assistRun/AssistProposalCard.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  applyAssistRun.mockReset();
});

const SCOPE = { count: 400, overlap: 390, jaccard: 0.95, changed: false };

function served(body: AssistApplyRequest): AssistApplyResponse {
  return {
    dry_run: body.dry_run !== false,
    class_name: body.class_name,
    selected: 380,
    skipped_locked: 20,
    updated: body.dry_run === false ? 380 : 0,
    scope_now: SCOPE,
  };
}

async function render(
  run = assistRun('completed', { result: { proposal: proposal() } }),
) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(AssistProposalCard, { target, props: { run } });
  await vi.waitFor(() => flushSync());
  return run;
}

const q = (id: string) =>
  target.querySelector(`[data-testid="${id}"]`) as HTMLElement | null;
const settle = () => vi.waitFor(() => flushSync());

describe('AssistProposalCard', () => {
  it('shows the proposed class, the agreement figure and the votes', async () => {
    applyAssistRun.mockImplementation(async (_id, b) => served(b));
    await render();
    expect(q('proposal-class')?.textContent).toContain('widget_a');
    expect(q('proposal-agreement')?.textContent).toContain('83%');
    expect(q('proposal-agreement')?.textContent).toContain('12');
    expect(target.textContent).toContain('widget_b');
    expect(target.textContent).toContain('widget_c');
  });

  it('asks the server for the dry-run count and shows it on the explicit apply button', async () => {
    applyAssistRun.mockImplementation(async (_id, b) => served(b));
    await render();
    await vi.waitFor(() => {
      flushSync();
      expect(q('proposal-apply')?.textContent).toContain('380');
    });
    expect(applyAssistRun.mock.calls[0]![1].dry_run).toBe(true);
    expect(q('proposal-locked')?.textContent).toContain('20');
    // The dry run wrote nothing: only the click applies.
    expect(applyAssistRun.mock.calls.every(([, b]) => b.dry_run === true)).toBe(true);
  });

  it('applies on click with dry_run false and says what was written', async () => {
    applyAssistRun.mockImplementation(async (_id, b) => served(b));
    await render();
    await vi.waitFor(() => {
      flushSync();
      expect((q('proposal-apply') as HTMLButtonElement).disabled).toBe(false);
    });
    (q('proposal-apply') as HTMLButtonElement).click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('proposal-applied')).not.toBeNull();
    });
    expect(applyAssistRun.mock.calls.at(-1)![1]).toMatchObject({
      class_name: 'widget_a',
      dry_run: false,
    });
    expect(q('proposal-applied')?.textContent).toContain('380');
  });

  it('marks a proposal stale from the served scope and renders the scope_changed refusal', async () => {
    applyAssistRun.mockRejectedValue(
      new ApiError(409, 'u', {
        detail: {
          error: 'scope_changed',
          message: 'only 62% of the scope is what the run saw',
        },
      }),
    );
    await render(
      assistRun('completed', {
        result: { proposal: proposal() },
        scope_now: { count: 300, overlap: 186, jaccard: 0.62, changed: true },
      }),
    );
    await vi.waitFor(() => {
      flushSync();
      expect(q('proposal-error')).not.toBeNull();
    });
    expect(q('proposal-stale')?.textContent).toContain('186');
    expect(q('proposal-error')?.textContent).toContain('only 62% of the scope');
    expect(q('proposal-apply')).toBeNull();
    // The override is offered only after the refusal, and counts first.
    applyAssistRun.mockImplementation(async (_id, b) => served(b));
    (q('proposal-anyway') as HTMLButtonElement).click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('proposal-apply')).not.toBeNull();
    });
    expect(applyAssistRun.mock.calls.at(-1)![1]).toMatchObject({
      allow_changed_scope: true,
      dry_run: true,
    });
  });

  it('says no class passed the gate and offers the distribution instead of a default', async () => {
    applyAssistRun.mockImplementation(async (_id, b) => served(b));
    await render(
      assistRun('completed', {
        result: {
          proposal: proposal({ class_name: null, passes_gate: false, agreement: 0.41 }),
        },
      }),
    );
    expect(q('proposal-class')).toBeNull();
    expect(q('proposal-nogate')?.textContent).toContain('41%');
    expect(q('proposal-apply')).toBeNull();
    (q('proposal-use-widget_b') as HTMLButtonElement).click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('proposal-apply')?.textContent).toContain('widget_b');
    });
  });

  it('offers to add a proposed class the registry lacks', async () => {
    applyAssistRun.mockRejectedValue(
      new ApiError(409, 'u', {
        detail: { error: 'class_not_in_registry', message: 'widget_a is not a class' },
      }),
    );
    await render(
      assistRun('completed', {
        result: { proposal: proposal({ in_registry: false }) },
      }),
    );
    await settle();
    expect(q('proposal-add-class')).not.toBeNull();
  });

  it('shows attribute suggestions as suggestions, not labels', async () => {
    applyAssistRun.mockImplementation(async (_id, b) => served(b));
    await render();
    const attrs = q('proposal-attributes')!;
    expect(attrs.textContent).toContain('finish');
    expect(attrs.textContent).toContain('matte');
    expect(attrs.textContent?.toLowerCase()).toContain('suggestions');
  });
});
