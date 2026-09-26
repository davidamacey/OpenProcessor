import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { reservedHotkeyLetters, setClassHotkey } from './classHotkey';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';
import { ApiError } from '$lib/api';
import type { RegistryClass } from '$lib/types';
import {
  installServedRegionProfile,
  resetDeploymentSlots,
} from '$lib/annotations/registeredSlots';
import { WIDGET_TAG_PROFILE } from '$lib/test/fixtures/regionSlot';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, renameClass: vi.fn() };
});

import { renameClass } from '$lib/api';

function cls(over: Partial<RegistryClass> & { id: number; name: string }): RegistryClass {
  return {
    group: null,
    count: 0,
    validated_count: 0,
    cluster_size: 0,
    added_at: '2026-01-01',
    hotkey_letter: null,
    deprecated: false,
    ...over,
  };
}

describe('reservedHotkeyLetters — server-served base, registry union on top', () => {
  const prevReserved = classesStore.reservedHotkeys;
  const prevClasses = classesStore.classes;

  afterEach(() => {
    classesStore.reservedHotkeys = prevReserved;
    classesStore.classes = prevClasses;
  });

  it('reflects the live server set (2026-09-24: /abdefgmnuxz) as the base — not a hardcoded constant', () => {
    classesStore.reservedHotkeys = [
      '/',
      'a',
      'b',
      'd',
      'e',
      'f',
      'g',
      'm',
      'n',
      'u',
      'x',
      'z',
    ];
    const reserved = reservedHotkeyLetters();
    for (const letter of ['/', 'a', 'b', 'd', 'e', 'f', 'g', 'm', 'n', 'u', 'x', 'z']) {
      expect(reserved.has(letter)).toBe(true);
    }
    expect(reserved.has('q')).toBe(false);
  });

  it('a letter absent from the served set is NOT reserved — this module never invents its own base', () => {
    classesStore.reservedHotkeys = ['g']; // deliberately narrow
    expect(reservedHotkeyLetters().has('n')).toBe(false);
  });

  it('no region profile: exactly the served set (no region keymap letters)', () => {
    classesStore.reservedHotkeys = ['g'];
    installServedRegionProfile(null);
    try {
      expect([...reservedHotkeyLetters()]).toEqual(['g']);
    } finally {
      resetDeploymentSlots();
    }
  });

  it('a served region profile adds its review keymap letters (d/f/e/b) to the served set', () => {
    classesStore.reservedHotkeys = ['g'];
    installServedRegionProfile(WIDGET_TAG_PROFILE);
    try {
      expect([...reservedHotkeyLetters()].sort()).toEqual(['b', 'd', 'e', 'f', 'g']);
    } finally {
      resetDeploymentSlots();
    }
  });
});

describe('setClassHotkey — server 400/409/422 detail surfaces verbatim in the toast', () => {
  const prevClasses = classesStore.classes;
  const prevReserved = classesStore.reservedHotkeys;

  beforeEach(() => {
    classesStore.classes = [];
    classesStore.reservedHotkeys = [];
    toastStore.toasts = [];
    vi.mocked(renameClass).mockReset();
  });

  afterEach(() => {
    classesStore.classes = prevClasses;
    classesStore.reservedHotkeys = prevReserved;
  });

  it('shows the server ApiError.detail, not a generic "API 409 ..." string', async () => {
    vi.mocked(renameClass).mockRejectedValue(
      new ApiError(409, '/curation/classes/4', {
        detail: "hotkey 'q' is already bound to 'widget_b' (class_id=4)",
      }),
    );
    const target = cls({ id: 8, name: 'widget_a' });
    await setClassHotkey(target, 'q');
    const last = toastStore.toasts.at(-1);
    expect(last?.kind).toBe('error');
    expect(last?.text).toBe(
      "Hotkey set failed: hotkey 'q' is already bound to 'widget_b' (class_id=4)",
    );
  });

  it('rejects a server-reserved letter client-side before any API call', async () => {
    classesStore.reservedHotkeys = ['b'];
    const target = cls({ id: 4, name: 'widget_b' });
    await setClassHotkey(target, 'b');
    expect(renameClass).not.toHaveBeenCalled();
    const last = toastStore.toasts.at(-1);
    expect(last?.text).toContain('reserved for a labeling action');
  });

  // K2 (plan §4.5): a server-side reserved-hotkey race (the client-side
  // check above missed it, e.g. a keymap rebind landed between page load
  // and this write) names which action(s) actually own the key, rather
  // than a generic string.
  it('names the owning actions for a structured 422 hotkey_reserved detail', async () => {
    classesStore.reservedHotkeys = []; // client-side check must not catch this first
    vi.mocked(renameClass).mockRejectedValue(
      new ApiError(422, '/curation/classes/8', {
        detail: {
          error: 'hotkey_reserved',
          message: "'q' is Discard on Review and Cluster.",
          actions: [
            {
              action_id: 'review.queue.discard',
              context: 'review.queue',
              label: 'Discard',
            },
            {
              action_id: 'cluster.discard',
              context: 'cluster',
              label: 'Discard selected',
            },
          ],
        },
      }),
    );
    const target = cls({ id: 8, name: 'widget_a' });
    await setClassHotkey(target, 'q');
    const last = toastStore.toasts.at(-1);
    expect(last?.kind).toBe('error');
    expect(last?.text).toContain('Discard');
    expect(last?.text).toContain('Discard selected');
  });

  it('names the owning class for a structured 409 hotkey_taken detail', async () => {
    vi.mocked(renameClass).mockRejectedValue(
      new ApiError(409, '/curation/classes/8', {
        detail: {
          error: 'hotkey_taken',
          message: "'k' is already bound to 'kart'.",
          class_id: 9,
          class_name: 'kart',
        },
      }),
    );
    const target = cls({ id: 8, name: 'widget_a' });
    await setClassHotkey(target, 'k');
    const last = toastStore.toasts.at(-1);
    expect(last?.kind).toBe('error');
    expect(last?.text).toContain("bound to 'kart'");
  });
});
