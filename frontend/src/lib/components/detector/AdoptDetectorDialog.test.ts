import { flushSync } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import { AdoptDetectorController } from '$lib/detector/adoptDetectorController.svelte';
import AdoptDetectorDialog from './AdoptDetectorDialog.svelte';

const dry = {
  dry_run: true,
  items: 40,
  classes_to_create: ['bus', 'car'],
  blocked: ['person'],
  blocked_items: 3,
  remaining: 0,
  op_id: null,
  job_id: null,
};
const real = { ...dry, dry_run: false, remaining: 5, op_id: 'op-9', job_id: 'j' };
const q = (id: string) => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);

describe('AdoptDetectorDialog', () => {
  it('shows the dry run, then writes on Confirm and links to the Label history', async () => {
    const adopt = vi.fn(async (d: boolean) => (d ? dry : real));
    const run = vi.fn(async () => ({
      opId: 'op-9',
      updated: 37,
      conflicts: 0,
      complete: true,
      status: 'completed' as const,
    }));
    const ondone = vi.fn();
    mountUi(AdoptDetectorDialog, {
      ctl: new AdoptDetectorController(adopt, run),
      onclose: () => {},
      ondone,
    });
    await vi.waitFor(() => expect(q('adopt-items')).not.toBeNull());
    expect(q('adopt-items')!.textContent).toBe('40');
    expect(q('adopt-create')!.textContent).toContain('bus, car');
    expect(q('adopt-blocked')!.textContent).toContain('person');
    expect(adopt).toHaveBeenCalledTimes(1);
    const confirm = [...document.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('Adopt 40 items'),
    )!;
    confirm.click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('adopt-result')).not.toBeNull();
    });
    expect(q('adopt-result')!.textContent).toContain('5 more are left');
    expect(q('adopt-undo-link')!.getAttribute('href')).toMatch(/quality\/history$/);
    expect(ondone).toHaveBeenCalledOnce();
  });
});
