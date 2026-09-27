/**
 * Mount-based behavior test for `/settings#keyboard`'s `KeymapCard`
 * (K2, docs/design/configurable-keyboard-shortcuts-plan-2026-09-26.md
 * §5.4/§5.6): rows per context, a locked action is read-only, a served
 * 422 validation report renders on the offending row, and a 409
 * `class_hotkey_conflict` opens the unbind dialog.
 *
 * Errors are constructed from the REAL `ApiError` class (not a fake
 * subclass) because `keymapValidationFailedDetail`/
 * `keymapClassConflictDetail` (api.ts) do their own `instanceof ApiError`
 * check against the actual, un-mocked class — a `FakeApiError` extending
 * plain `Error` would silently fail that check and never match.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import { FALLBACK_KEYMAP, type KeymapDocument } from '$lib/keymapFallback';

const putKeymap = vi.fn();
const validateKeymap = vi.fn();
const resetKeymap = vi.fn();

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    putKeymap: (...args: unknown[]) => putKeymap(...args),
    validateKeymap: (...args: unknown[]) => validateKeymap(...args),
    resetKeymap: (...args: unknown[]) => resetKeymap(...args),
  };
});

const { default: KeymapCard } = await import('./KeymapCard.svelte');
const { keymapStore } = await import('$stores/keymap.svelte');
const { ApiError } = await import('$lib/api');

function servedDoc(overrides: Partial<KeymapDocument> = {}): KeymapDocument {
  return { ...FALLBACK_KEYMAP, revision: 3, is_default: true, ...overrides };
}

function okReport() {
  return { ok: true, errors: [], warnings: [], force_allowed: false };
}

let target: HTMLDivElement;
let instance: unknown;

beforeEach(() => {
  validateKeymap.mockResolvedValue(okReport());
  keymapStore.setDocument(servedDoc(), 'served');
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  target.remove();
  keymapStore.resetToFallback();
  vi.clearAllMocks();
});

function changeButtonForRow(labelText: string): HTMLElement {
  const row = [...target.querySelectorAll('tr')].find((tr) =>
    tr.textContent?.includes(labelText),
  );
  if (!row) throw new Error(`no row for ${labelText}`);
  const btn = [...row.querySelectorAll('button')].find(
    (b) => b.textContent?.trim() === 'Change',
  );
  if (!btn) throw new Error(`no Change button for ${labelText}`);
  return btn as HTMLElement;
}

function saveButton(): HTMLElement {
  return [...target.querySelectorAll('button')].find(
    (b) => b.textContent?.trim() === 'Save',
  ) as HTMLElement;
}

function confirmDialogSaveButton(): HTMLElement {
  return [...target.querySelectorAll('button')].filter(
    (b) => b.textContent?.trim() === 'Save',
  )[1] as HTMLElement;
}

/** Change the Discard row's key to 'x' and open the confirm dialog. */
async function rebindDiscardToXAndOpenConfirm(): Promise<void> {
  changeButtonForRow('Discard').dispatchEvent(new MouseEvent('click', { bubbles: true }));
  flushSync();
  const captureButton = target.querySelector('[data-capture]') as HTMLElement;
  expect(captureButton).toBeTruthy();
  captureButton.dispatchEvent(
    new KeyboardEvent('keydown', { key: 'x', bubbles: true, cancelable: true }),
  );
  flushSync();
  saveButton().click();
  flushSync();
}

function text(): string {
  return target.textContent ?? '';
}

describe('KeymapCard', () => {
  it('renders a row per context, grouped by the served labels', () => {
    instance = mount(KeymapCard, { target });
    flushSync();
    expect(text()).toContain('Everywhere');
    expect(text()).toContain('Review queue');
    expect(text()).toContain('Discard');
  });

  it('renders a locked action read-only, with no Change button', () => {
    instance = mount(KeymapCard, { target });
    flushSync();
    const closeRow = [...target.querySelectorAll('tr')].find((tr) =>
      tr.textContent?.includes('Close this panel'),
    );
    expect(closeRow).toBeTruthy();
    expect(closeRow?.querySelector('button')).toBeNull();
  });

  it('shows the "custom keys" badge when the served doc is not default', () => {
    keymapStore.setDocument(servedDoc({ is_default: false }), 'served');
    instance = mount(KeymapCard, { target });
    flushSync();
    expect(text()).toContain('custom keys');
  });

  it('renders a served 422 validation_failed report on save', async () => {
    putKeymap.mockRejectedValue(
      new ApiError(422, '/curation/keymap', {
        detail: {
          error: 'validation_failed',
          message: 'The keymap has 1 error.',
          current_revision: 3,
          report: {
            ok: false,
            force_allowed: false,
            errors: [
              {
                code: 'keymap_context_collision',
                severity: 'error',
                field: 'overrides.review.queue.discard',
                message: "'x' is already Discard selected on Cluster.",
                bypassable: false,
              },
            ],
            warnings: [],
          },
        },
      }),
    );
    instance = mount(KeymapCard, { target });
    flushSync();

    await rebindDiscardToXAndOpenConfirm();
    confirmDialogSaveButton().click();
    await new Promise((r) => setTimeout(r, 0));
    flushSync();

    expect(text()).toContain("'x' is already Discard selected on Cluster.");
  });

  it('a 409 revision_conflict shows the reload banner', async () => {
    putKeymap.mockRejectedValue(
      new ApiError(409, '/curation/keymap', {
        detail: {
          error: 'revision_conflict',
          message: 'The keymap changed since you loaded it.',
          current_revision: 5,
        },
      }),
    );
    instance = mount(KeymapCard, { target });
    flushSync();

    await rebindDiscardToXAndOpenConfirm();
    confirmDialogSaveButton().click();
    await new Promise((r) => setTimeout(r, 0));
    flushSync();

    expect(text()).toContain('changed elsewhere');
    expect(text()).toContain('Reload');
  });

  it('a 409 class_hotkey_conflict opens the unbind dialog and resends with the flag', async () => {
    putKeymap.mockRejectedValueOnce(
      new ApiError(409, '/curation/keymap', {
        detail: {
          error: 'class_hotkey_conflict',
          message: "'x' is bound to a class.",
          current_revision: 3,
          report: okReport(),
          class_conflicts: [
            {
              project: 'default',
              class_id: 9,
              class_name: 'widget',
              combo: 'x',
              action_id: 'cluster.discard',
            },
          ],
        },
      }),
    );
    putKeymap.mockResolvedValueOnce(servedDoc({ revision: 4 }));

    instance = mount(KeymapCard, { target });
    flushSync();

    await rebindDiscardToXAndOpenConfirm();
    confirmDialogSaveButton().click();
    await new Promise((r) => setTimeout(r, 0));
    flushSync();

    expect(target.querySelector('[data-testid="class-conflict-dialog"]')).toBeTruthy();
    expect(text()).toContain('widget');

    const unbindButton = [...target.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('Unbind these class keys and save'),
    ) as HTMLElement;
    unbindButton.click();
    await new Promise((r) => setTimeout(r, 0));
    flushSync();

    expect(putKeymap).toHaveBeenLastCalledWith(
      expect.objectContaining({ unbind_conflicting_class_hotkeys: true }),
    );
  });

  // --- K2b: per-context overrides -----------------------------------

  function memberRow(actionId: string): HTMLElement {
    const row = target.querySelector(`[data-testid="member-${actionId}"]`);
    if (!row) throw new Error(`no member row for ${actionId}`);
    return row as HTMLElement;
  }

  function memberChangeButton(actionId: string): HTMLElement {
    const btn = [...memberRow(actionId).querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'Change',
    );
    if (!btn) throw new Error(`no Change button for ${actionId}`);
    return btn as HTMLElement;
  }

  function pressKeyOnCaptureIn(container: HTMLElement, key: string): void {
    const captureButton = container.querySelector('[data-capture]') as HTMLElement;
    expect(captureButton).toBeTruthy();
    captureButton.dispatchEvent(
      new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }),
    );
    flushSync();
  }

  it('a group row edits every member action id at once', () => {
    instance = mount(KeymapCard, { target });
    flushSync();

    // 'undo' spans review / cluster / clusters_search / region_gallery.
    const groupSummary = [...target.querySelectorAll('summary')].find((s) =>
      s.textContent?.includes('Undo'),
    );
    expect(groupSummary).toBeTruthy();
    const groupRow = groupSummary!.closest('details') as HTMLElement;
    const groupChangeBtn = [...groupRow.querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'Change',
    ) as HTMLElement;
    groupChangeBtn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    flushSync();
    pressKeyOnCaptureIn(groupRow, 'y');

    saveButton().click();
    flushSync();
    confirmDialogSaveButton().click();
    flushSync();

    expect(putKeymap).toHaveBeenCalledWith(
      expect.objectContaining({
        overrides: expect.objectContaining({
          'review.undo': expect.arrayContaining(['y']),
          'cluster.undo': expect.arrayContaining(['y']),
          'clusters_search.undo': expect.arrayContaining(['y']),
          'region_gallery.undo': expect.arrayContaining(['y']),
        }),
      }),
    );
  });

  it('a per-context edit writes only that action id and shows the detached marker', () => {
    instance = mount(KeymapCard, { target });
    flushSync();

    // Expand "Customize per page" for the undo group.
    const customizeSummary = [...target.querySelectorAll('summary')].find(
      (s) => s.textContent?.trim() === 'Customize per page',
    );
    expect(customizeSummary).toBeTruthy();

    memberChangeButton('cluster.undo').dispatchEvent(
      new MouseEvent('click', { bubbles: true }),
    );
    flushSync();
    pressKeyOnCaptureIn(memberRow('cluster.undo'), 'y');

    // Detached marker on the edited member, not on its siblings.
    expect(target.querySelector('[data-testid="detached-cluster.undo"]')).toBeTruthy();
    expect(target.querySelector('[data-testid="detached-review.undo"]')).toBeNull();

    saveButton().click();
    flushSync();
    confirmDialogSaveButton().click();
    flushSync();

    const call = putKeymap.mock.calls.at(-1)?.[0];
    expect(call.overrides['cluster.undo']).toEqual(expect.arrayContaining(['y']));
    expect(call.overrides['review.undo']).toBeUndefined();
  });

  it('"reset to group" clears a detached member back to the shared value', () => {
    instance = mount(KeymapCard, { target });
    flushSync();

    memberChangeButton('cluster.undo').dispatchEvent(
      new MouseEvent('click', { bubbles: true }),
    );
    flushSync();
    pressKeyOnCaptureIn(memberRow('cluster.undo'), 'y');
    expect(target.querySelector('[data-testid="detached-cluster.undo"]')).toBeTruthy();

    const resetBtn = [...memberRow('cluster.undo').querySelectorAll('button')].find(
      (b) => b.textContent?.trim() === 'reset to group',
    ) as HTMLElement;
    resetBtn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    flushSync();

    expect(target.querySelector('[data-testid="detached-cluster.undo"]')).toBeNull();
  });
});
