import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';

vi.mock('$lib/api_select', async () => {
  const actual =
    await vi.importActual<typeof import('$lib/api_select')>('$lib/api_select');
  return {
    ...actual,
    bulkLabelSelection: vi.fn(),
    batchExcludeSelection: vi.fn(),
    batchUnexcludeSelection: vi.fn(),
    moveSelectionToCluster: vi.fn(),
  };
});

vi.mock('$lib/api_jobs', () => ({ getJob: vi.fn() }));

import { getJob } from '$lib/api_jobs';
import { ApiError } from '$lib/api_errors';
import { batchExcludeSelection, bulkLabelSelection } from '$lib/api_select';
import SelectionActionDialog from './SelectionActionDialog.svelte';
import { createSelectionActionController } from '$lib/itemFilter/selectionActionController.svelte';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';
import type { RegistryClass } from '$lib/types_classes';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;
const prevClasses = classesStore.classes;

const tick = () => new Promise((r) => setTimeout(r, 0));
const q = (sel: string) => target.querySelector<HTMLElement>(sel);
const apply = () =>
  [...document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button')].at(-1)!;

function mountDialog(onapplied = vi.fn()) {
  const controller = createSelectionActionController(() => ({
    filter: { class_names: ['widget'] },
  }));
  instance = mount(SelectionActionDialog, { target, props: { controller, onapplied } });
  return { controller, onapplied };
}

beforeEach(() => {
  classesStore.classes = [{ id: 5, name: 'widget', deprecated: false } as RegistryClass];
  undoStore.clear();
  target = document.createElement('div');
  document.body.appendChild(target);
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target.remove();
  classesStore.classes = prevClasses;
  vi.clearAllMocks();
});

describe('SelectionActionDialog', () => {
  it('renders nothing until an action is opened', () => {
    mountDialog();
    flushSync();
    expect(q('[role="dialog"]')).toBeNull();
  });

  it("states the server's dry-run count and applies only after Apply", async () => {
    vi.mocked(batchExcludeSelection)
      .mockResolvedValueOnce({ dry_run: true, selected: 12 })
      .mockResolvedValueOnce({ excluded: 12, updated_ids: ['a'], errors: 0 });
    const { controller, onapplied } = mountDialog();
    await controller.open('exclude');
    flushSync();
    expect(q('[data-testid="selection-count"]')!.textContent).toContain('12');
    expect(batchExcludeSelection).toHaveBeenCalledTimes(1);
    apply().click();
    await tick();
    flushSync();
    expect(batchExcludeSelection).toHaveBeenCalledTimes(2);
    expect(onapplied).toHaveBeenCalled();
    expect(q('[role="dialog"]')).toBeNull();
    expect(undoStore.stack).toHaveLength(1);
  });

  it('Apply is disabled for a zero-item dry run', async () => {
    vi.mocked(batchExcludeSelection).mockResolvedValue({ dry_run: true, selected: 0 });
    const { controller } = mountDialog();
    await controller.open('exclude');
    flushSync();
    expect(apply().disabled).toBe(true);
  });

  it('label asks for a class before any request, then dry-runs with it', async () => {
    vi.mocked(bulkLabelSelection).mockResolvedValue({ dry_run: true, selected: 3 });
    const { controller } = mountDialog();
    await controller.open('label');
    flushSync();
    expect(bulkLabelSelection).not.toHaveBeenCalled();
    expect(apply().disabled).toBe(true);
    const sel = q('[data-testid="selection-class"]') as HTMLSelectElement;
    sel.value = '5';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    await tick();
    flushSync();
    expect(bulkLabelSelection).toHaveBeenCalledWith(
      { filter: { class_names: ['widget'] } },
      5,
      true,
      expect.anything(),
    );
    expect(q('[data-testid="selection-count"]')!.textContent).toContain('3');
  });

  it('the seed control exists only for a random sample, and a change re-runs the dry run', async () => {
    vi.mocked(batchExcludeSelection).mockResolvedValue({ dry_run: true, selected: 4 });
    const { controller } = mountDialog();
    await controller.open('exclude');
    flushSync();
    expect(q('[data-testid="selection-seed"]')).toBeNull();
    const sample = q('[data-testid="selection-sample"]') as HTMLSelectElement;
    sample.value = 'random';
    sample.dispatchEvent(new Event('change', { bubbles: true }));
    await tick();
    flushSync();
    expect(q('[data-testid="selection-seed"]')).not.toBeNull();
    expect(batchExcludeSelection).toHaveBeenCalledTimes(2);
    expect(vi.mocked(batchExcludeSelection).mock.calls[1]![0]).toMatchObject({
      sample: 'random',
    });
  });

  it('shows a served refusal verbatim', async () => {
    vi.mocked(batchExcludeSelection).mockRejectedValue(
      new ApiError(422, '/x', { detail: 'selection matches more than 20000 items' }),
    );
    const { controller } = mountDialog();
    await controller.open('exclude');
    flushSync();
    expect(q('[data-testid="selection-error"]')!.textContent).toContain(
      'selection matches more than 20000 items',
    );
    expect(apply().disabled).toBe(true);
  });

  it('a queued label shows the served state and the jobs ahead, never a bare "Working…"', async () => {
    vi.mocked(bulkLabelSelection)
      .mockResolvedValueOnce({ dry_run: true, selected: 24 })
      .mockResolvedValueOnce({ op_id: 'op1', job_id: 'job1' } as never);
    vi.mocked(getJob).mockResolvedValue({
      id: 'job1',
      kind: 'label_op',
      status: 'queued',
      queue: 'interactive',
      queue_ahead: 2,
      progress: {},
    } as never);
    const { controller } = mountDialog();
    controller.classId = 5;
    await controller.open('label');
    flushSync();
    apply().click();
    await tick();
    await tick();
    flushSync();
    expect(q('[data-testid="selection-job"]')!.textContent).toContain('Queued, 2 ahead');
    expect(apply().textContent).toContain('Queued, 2 ahead');
    expect(apply().textContent).not.toContain('Working');
    expect(q('[data-testid="selection-background"]')).not.toBeNull();
    controller.cancel();
  });

  it('Run in background closes the dialog, keeps the job followed and toasts the end with Undo', async () => {
    vi.mocked(bulkLabelSelection)
      .mockResolvedValueOnce({ dry_run: true, selected: 24 })
      .mockResolvedValueOnce({ op_id: 'op1', job_id: 'job1' } as never);
    vi.mocked(getJob)
      .mockResolvedValueOnce({
        id: 'job1',
        status: 'queued',
        queue_ahead: 1,
        progress: {},
      } as never)
      .mockResolvedValue({
        id: 'job1',
        status: 'completed',
        progress: {},
        result: { op_id: 'op1', updated: 24, conflicts: 0 },
      } as never);
    const { controller, onapplied } = mountDialog();
    controller.classId = 5;
    await controller.open('label');
    flushSync();
    apply().click();
    await tick();
    await tick();
    flushSync();
    q('[data-testid="selection-background"]')!.click();
    flushSync();
    expect(q('[role="dialog"]')).toBeNull();
    await new Promise((r) => setTimeout(r, 900));
    flushSync();
    expect(onapplied).toHaveBeenCalled();
    const toast = toastStore.toasts.find((t) => t.text.startsWith('Labeled 24'));
    expect(toast?.action?.label).toBe('Undo');
    expect(undoStore.stack).toHaveLength(1);
  });
});
