import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { ApiError } from '$lib/api_errors';
import {
  AssistRunController,
  type AssistRunDeps,
} from '$lib/assistRun/assistRunController.svelte';
import { assistRun, proposal } from '$lib/test/fixtures/assist';
import { EMPTY_METHODS, parseMethodsResponse } from '$lib/strategies';
import { strategiesStore } from '$stores/strategies.svelte';
import { MIN_AGREEMENT_RANGE, type AssistRunRequest } from '$lib/types_assist';

vi.mock('$lib/api_assist', () => ({
  applyAssistRun: vi.fn(async () => ({})),
  createAssistRun: vi.fn(),
  getAssistRun: vi.fn(),
  cancelAssistRun: vi.fn(),
}));

import AssistRunDialog from '$components/assistRun/AssistRunDialog.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  strategiesStore.loaded = true;
  strategiesStore.methods = parseMethodsResponse({
    strategies: [
      { id: 'local_vlm', axis: 'vlm', label: 'Local VLM', status: 'stable' },
      { id: 'cars', axis: 'prompt_pack', label: 'Cars', status: 'stable' },
    ],
  });
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
  strategiesStore.methods = EMPTY_METHODS;
  strategiesStore.loaded = false;
});

function setup(over: Partial<AssistRunDeps> = {}) {
  const deps: AssistRunDeps = {
    create: vi.fn(async (b: AssistRunRequest) =>
      assistRun(b.dry_run === false ? 'queued' : 'dry_run'),
    ),
    get: vi.fn(async () =>
      assistRun('completed', {
        result: { proposal: proposal(), classify: { predicted: 41, updated: 14 } },
      }),
    ),
    cancel: vi.fn(async () => assistRun('cancelled')),
    debounceMs: 0,
    pollMs: 0,
    ...over,
  };
  const ctl = new AssistRunController(deps);
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(AssistRunDialog, { target, props: { ctl } });
  return { ctl, deps };
}

const q = (id: string) =>
  document.querySelector(`[data-testid="${id}"]`) as HTMLElement | null;
const open = (ctl: AssistRunController) => {
  ctl.open({
    title: 'Sub-cluster 10000b',
    selected: {
      kind: 'crops',
      crop_ids: ['c1', 'c2', 'c3'],
      context: { cluster_id: 10000, cluster_subid: '10000b' },
    },
    whole: { kind: 'cluster', cluster_id: 10000, cluster_subid: '10000b' },
  });
  flushSync();
};

describe('AssistRunDialog', () => {
  it('renders nothing until a controller opens it', () => {
    setup();
    expect(q('assist-dialog')).toBeNull();
  });

  it('offers the three sample choices and the four actions, and shows the served dry run', async () => {
    const { ctl } = setup();
    open(ctl);
    expect(q('assist-choice-selected')?.textContent).toContain('3');
    expect(q('assist-choice-all')).not.toBeNull();
    expect(q('assist-choice-representative')).not.toBeNull();
    for (const a of ['classify', 'attributes', 'verify', 'segment']) {
      expect(q(`assist-action-${a}`)).not.toBeNull();
    }
    await vi.waitFor(() => {
      flushSync();
      expect(q('assist-plan-sampled')?.textContent).toBe('41');
    });
    expect(q('assist-plan-calls')?.textContent).toBe('6');
  });

  it('shows the server default for min_agreement and re-plans with the typed value', async () => {
    const { ctl, deps } = setup();
    open(ctl);
    ctl.setChoice('all');
    await vi.waitFor(() => expect(ctl.preview).not.toBeNull());
    flushSync();
    const field = q('assist-min-agreement') as HTMLInputElement;
    expect(field.placeholder).toBe(String(ctl.preview!.request.min_agreement));
    expect(field.min).toBe(String(MIN_AGREEMENT_RANGE.min));
    expect(field.max).toBe(String(MIN_AGREEMENT_RANGE.max));
    field.value = '0.8';
    field.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    await vi.waitFor(() => {
      const last = vi.mocked(deps.create).mock.calls.at(-1)![0];
      expect(last.min_agreement).toBe(0.8);
    });
  });

  it('re-plans with the representative sample, seed and size the user typed', async () => {
    const { ctl, deps } = setup();
    open(ctl);
    (
      q('assist-choice-representative')!.querySelector('input') as HTMLInputElement
    ).click();
    flushSync();
    const size = q('assist-size') as HTMLInputElement;
    size.value = '12';
    size.dispatchEvent(new Event('input', { bubbles: true }));
    const seed = q('assist-seed') as HTMLInputElement;
    seed.value = '3';
    seed.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    await vi.waitFor(() => {
      const last = vi.mocked(deps.create).mock.calls.at(-1)![0];
      expect(last.sample).toMatchObject({ mode: 'representative', size: 12, seed: 3 });
    });
    expect(vi.mocked(deps.create).mock.calls.every(([b]) => b.dry_run === true)).toBe(
      true,
    );
  });

  it('shows the served refusal and keeps Start disabled', async () => {
    const { ctl } = setup({
      create: vi.fn(async () => {
        throw new ApiError(409, 'u', {
          detail: { error: 'vlm_not_configured', message: 'this project has no VLM' },
        });
      }),
    });
    open(ctl);
    await vi.waitFor(() => {
      flushSync();
      expect(q('assist-plan-error')?.textContent).toContain('this project has no VLM');
    });
    expect((q('assist-start') as HTMLButtonElement).disabled).toBe(true);
  });

  it('starts the run, shows its result and the proposal card', async () => {
    const { ctl, deps } = setup();
    open(ctl);
    await vi.waitFor(() => {
      flushSync();
      expect((q('assist-start') as HTMLButtonElement).disabled).toBe(false);
    });
    (q('assist-start') as HTMLButtonElement).click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('assist-progress')?.getAttribute('data-status')).toBe('completed');
    });
    expect(vi.mocked(deps.create).mock.calls.at(-1)![0].dry_run).toBe(false);
    expect(q('assist-result')?.textContent).toContain('41 answered');
    expect(q('assist-proposal-card')).not.toBeNull();
  });

  it('closes from the Close button', async () => {
    const { ctl } = setup();
    open(ctl);
    (q('assist-close') as HTMLButtonElement).click();
    flushSync();
    expect(ctl.isOpen).toBe(false);
  });
});
