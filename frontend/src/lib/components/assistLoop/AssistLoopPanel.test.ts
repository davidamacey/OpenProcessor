import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { ApiError } from '$lib/api_errors';
import { AssistLoopController } from '$lib/assistLoop/assistLoopController.svelte';
import { loopDeps, loopNode, loopState } from '$lib/test/fixtures/assistLoop';
import { EMPTY_METHODS } from '$lib/strategies';
import { strategiesStore } from '$stores/strategies.svelte';
import AssistLoopPanel from '$components/assistLoop/AssistLoopPanel.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;
let ctl: AssistLoopController;

beforeEach(() => {
  strategiesStore.loaded = true;
  strategiesStore.methods = EMPTY_METHODS;
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  ctl?.dispose();
  target?.remove();
});

const q = (id: string) =>
  target.querySelector(`[data-testid="${id}"]`) as HTMLElement | null;
const click = (id: string) => {
  (q(id) as HTMLElement).click();
  flushSync();
};
const settle = () => vi.waitFor(() => flushSync());

async function render(deps = loopDeps(), opts: { open?: boolean } = { open: true }) {
  ctl = new AssistLoopController(deps);
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(AssistLoopPanel, { target, props: { ctl } });
  await vi.waitFor(() => expect(ctl.supported).not.toBeNull());
  flushSync();
  if (opts.open && q('loop-panel-toggle')?.getAttribute('aria-expanded') !== 'true') {
    click('loop-panel-toggle');
  }
}

describe('AssistLoopPanel', () => {
  it('is absent when the backend does not serve the loop routes', async () => {
    await render(
      loopDeps({
        list: async () => {
          throw new ApiError(404, '/x', { detail: 'Not Found' });
        },
      }),
      { open: false },
    );
    expect(q('loop-panel')).toBeNull();
  });

  it('shows the served dry run as the default step and starts only on request', async () => {
    const create = vi.fn(async (b: { dry_run: boolean }) =>
      loopState(b.dry_run ? 'dry_run' : 'queued'),
    );
    await render(loopDeps({ list: async () => ({ loops: [] }), create }));
    await vi.waitFor(() => {
      flushSync();
      expect(q('loop-plan-roots')?.textContent?.trim()).toBe('12');
    });
    expect(q('loop-plan-crops')?.textContent).toContain('380 crops');
    expect(q('loop-plan-fits')?.textContent?.trim()).toBe('yes');
    expect(create.mock.calls.every((c) => (c[0] as { dry_run: boolean }).dry_run)).toBe(
      true,
    );
    expect(q('loop-plan')?.textContent).toContain('No second opinion');
    (q('loop-start') as HTMLButtonElement).click();
    await vi.waitFor(() =>
      expect(create.mock.calls.at(-1)?.[0]).toMatchObject({ dry_run: false }),
    );
  });

  it('prints a refusal of the dry run verbatim and keeps Start disabled', async () => {
    const create = vi.fn(async () => {
      throw new ApiError(409, '/x', {
        detail: { error: 'assist_loop_busy', message: 'a loop is already live' },
      });
    });
    await render(loopDeps({ list: async () => ({ loops: [] }), create }));
    await vi.waitFor(() => {
      flushSync();
      expect(q('loop-plan-error')?.textContent).toContain('a loop is already live');
    });
    expect((q('loop-start') as HTMLButtonElement).disabled).toBe(true);
  });

  it('shows budget, level counts and a cancel control for a live loop, without a start form', async () => {
    await render(
      loopDeps({
        list: async () => ({ loops: [loopState('running')] }),
        get: async () => loopState('running'),
      }),
    );
    await settle();
    expect(q('loop-status')?.textContent).toContain('Running');
    expect(q('loop-pool-vlm')?.textContent).toContain('500 / 2,000');
    expect(q('loop-levels')?.textContent).toContain('Progress by level');
    expect(q('loop-cancel')).not.toBeNull();
    expect(q('loop-resume')).toBeNull();
    expect(q('loop-start')).toBeNull();
    expect((q('loop-rollback-all') as HTMLButtonElement).disabled).toBe(true);
  });

  it('says why a paused loop stopped and offers resume, not cancel', async () => {
    await render(
      loopDeps({
        list: async () => ({
          loops: [
            loopState('paused', {
              pause_reason: 'audit',
              audit: {
                status: 'fail',
                sampled: 40,
                agreed: 30,
                agreement: 0.75,
                lower_bound: 0.63,
                crops_spent: 40,
              },
            }),
          ],
        }),
      }),
    );
    await settle();
    expect(q('loop-status')?.textContent).toContain('second opinion disagreed');
    expect(q('loop-resume')).not.toBeNull();
    expect(q('loop-cancel')).toBeNull();
    const audit = q('loop-audit')?.textContent?.replace(/\s+/g, ' ') ?? '';
    expect(audit).toContain('30 of 40 sampled labels agreed');
    expect(audit).toContain('75.0 percent');
    expect(audit).toContain('lower bound 63.0 percent');
  });

  it('states that the audit was off rather than leaving it blank', async () => {
    await render();
    await settle();
    expect(q('loop-audit')?.textContent).toContain('no second model re-checked');
  });

  it('opens a node to its votes and member crops, announced with aria-expanded', async () => {
    await render(
      loopDeps({
        nodes: async () => ({
          nodes: [
            loopNode({
              distribution: [
                { class_name: 'widget_a', votes: 30 },
                { class_name: 'widget_b', votes: 3 },
              ],
            }),
          ],
          next_after: null,
        }),
      }),
    );
    await settle();
    const toggle = q('loop-node-toggle') as HTMLButtonElement;
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    toggle.click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('loop-node-detail')).not.toBeNull();
      expect(q('loop-members-count')?.textContent).toContain('2 member ids');
    });
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(q('loop-node-detail')?.textContent).toContain('widget_a 30, widget_b 3');
    expect(target.querySelectorAll('[data-testid="loop-node-detail"] img')).toHaveLength(
      2,
    );
  });

  it('rolls back in two steps: the served dry run, then a confirmation', async () => {
    const rollback = vi.fn(async (_id: string, b: { dry_run: boolean }) => ({
      dry_run: b.dry_run,
      matched: 267,
      restored: 260,
      kept_changed: 7,
      conflicts: 0,
    }));
    await render(loopDeps({ rollback }));
    await settle();
    click('loop-rollback-all');
    await vi.waitFor(() => {
      flushSync();
      expect(q('loop-rollback-plan')).not.toBeNull();
    });
    expect(rollback).toHaveBeenCalledTimes(1);
    const plan = q('loop-rollback-plan')?.textContent?.replace(/\s+/g, ' ') ?? '';
    expect(plan).toContain('Restore 260 of 267');
    expect(plan).toContain('7 were changed since');
    click('loop-rollback-confirm');
    await vi.waitFor(() => {
      flushSync();
      expect(q('loop-rollback-result')?.textContent).toContain('260 restored');
    });
    expect(rollback).toHaveBeenLastCalledWith('loop_abc123', {
      node_paths: null,
      dry_run: false,
    });
  });

  it('is built from named, non-nested controls', async () => {
    await render(loopDeps({ list: async () => ({ loops: [loopState('completed')] }) }));
    await settle();
    click('loop-node-toggle');
    await settle();
    const interactive = 'button, a[href], input, select, textarea, [role="button"]';
    for (const el of target.querySelectorAll(interactive)) {
      expect(el.parentElement?.closest('button, a[href], [role="button"]')).toBeNull();
      const name =
        el.getAttribute('aria-label') ||
        el.textContent?.trim() ||
        (el.id && target.querySelector(`label[for="${el.id}"]`)?.textContent?.trim()) ||
        el.closest('label')?.textContent?.trim();
      expect(name, el.outerHTML.slice(0, 120)).toBeTruthy();
    }
    for (const bar of target.querySelectorAll('[role="progressbar"]')) {
      expect(bar.getAttribute('aria-label')).toBeTruthy();
    }
    expect(q('loop-panel-toggle')?.getAttribute('aria-controls')).toBe('loop-panel-body');
    expect(target.querySelector('#loop-panel-body')).not.toBeNull();
  });
});
