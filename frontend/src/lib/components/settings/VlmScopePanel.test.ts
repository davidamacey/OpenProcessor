import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { ApiError } from '$lib/api';
import { VlmPolicyEditor } from '$lib/labelConfirmation/vlmPolicyController.svelte';
import type { VlmPolicy, VlmPolicyUpdate } from '$lib/types_labelConfirmation';
import VlmScopePanel from './VlmScopePanel.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const SERVED: VlmPolicy = {
  scope: 'all',
  conf_max: 0.8,
  per_cluster: 5,
  max_crops_per_day: 0,
  sample_frac: 1,
  revision: 3,
};

async function render(
  opts: {
    policy?: VlmPolicy;
    put?: (req: VlmPolicyUpdate) => Promise<VlmPolicy>;
    get?: () => Promise<VlmPolicy>;
  } = {},
) {
  const putPolicy = vi.fn(
    opts.put ?? (async (r: VlmPolicyUpdate) => ({ ...r, revision: 4 })),
  );
  const editor = new VlmPolicyEditor({
    getPolicy: opts.get ?? (async () => opts.policy ?? SERVED),
    putPolicy,
  });
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(VlmScopePanel, { target, props: { editor } });
  await vi.waitFor(() => {
    flushSync();
    expect(editor.loading).toBe(false);
  });
  flushSync();
  return { editor, putPolicy };
}

const q = (sel: string) => target.querySelector(sel) as HTMLElement | null;
const knob = (k: string) => q(`[data-testid="vlm-knob-${k}"]`) as HTMLInputElement | null;

describe('VlmScopePanel', () => {
  it('offers the four served scopes and checks the served one', async () => {
    await render({ policy: { ...SERVED, scope: 'representatives' } });
    for (const s of ['all', 'uncertain', 'representatives', 'off']) {
      expect(q(`[data-testid="vlm-scope-${s}"]`)).not.toBeNull();
    }
    expect(
      (q('[data-testid="vlm-scope-representatives"]') as HTMLInputElement).checked,
    ).toBe(true);
  });

  it('shows only the knobs the chosen scope reads', async () => {
    await render({ policy: { ...SERVED, scope: 'uncertain' } });
    expect(knob('conf_max')).not.toBeNull();
    expect(knob('per_cluster')).toBeNull();
    expect(knob('sample_frac')).not.toBeNull();
    expect(knob('max_crops_per_day')).not.toBeNull();

    (q('[data-testid="vlm-scope-representatives"]') as HTMLInputElement).click();
    flushSync();
    expect(knob('conf_max')).toBeNull();
    expect(knob('per_cluster')?.value).toBe('5');

    (q('[data-testid="vlm-scope-off"]') as HTMLInputElement).click();
    flushSync();
    expect(q('[data-testid="vlm-scope-knobs"]')).toBeNull();
  });

  it('saves the edited policy with the revision it read, then confirms', async () => {
    const { putPolicy } = await render();
    expect((q('[data-testid="vlm-scope-save"]') as HTMLButtonElement).disabled).toBe(
      true,
    );

    (q('[data-testid="vlm-scope-uncertain"]') as HTMLInputElement).click();
    flushSync();
    const conf = knob('conf_max')!;
    conf.value = '0.6';
    conf.dispatchEvent(new Event('input', { bubbles: true }));
    const budget = knob('max_crops_per_day')!;
    budget.value = '500';
    budget.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    (q('[data-testid="vlm-scope-save"]') as HTMLButtonElement).click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('[data-testid="vlm-scope-saved"]')).not.toBeNull();
    });
    expect(putPolicy.mock.calls[0]![0]).toMatchObject({
      scope: 'uncertain',
      conf_max: 0.6,
      max_crops_per_day: 500,
      expected_revision: 3,
    });
  });

  it('on a stale revision shows the served message with Reload and Keep my edits', async () => {
    await render({
      put: async () => {
        throw new ApiError(409, 'u', {
          detail: { error: 'revision_conflict', message: 'vlm policy changed: 3 != 7' },
        });
      },
    });
    (q('[data-testid="vlm-scope-off"]') as HTMLInputElement).click();
    flushSync();
    (q('[data-testid="vlm-scope-save"]') as HTMLButtonElement).click();
    await vi.waitFor(() => {
      flushSync();
      expect(q('[data-testid="vlm-scope-save-error"]')).not.toBeNull();
    });
    const err = q('[data-testid="vlm-scope-save-error"]')!;
    expect(err.textContent).toContain('vlm policy changed: 3 != 7');
    expect(err.textContent).toContain('Reload');
    expect(err.textContent).toContain('Keep my edits');
  });

  it('shows the served error when the policy cannot be read', async () => {
    await render({
      get: async () => {
        throw new ApiError(404, 'u', { detail: 'Not Found' });
      },
    });
    expect(q('[data-testid="vlm-scope-load-error"]')?.textContent).toContain('Not Found');
    expect(q('[data-testid="vlm-scope-save"]')).toBeNull();
  });

  it('says a VLM label stays a suggestion whatever the scope', async () => {
    await render();
    expect(target.textContent).toContain('suggestion until a human validates it');
  });
});
