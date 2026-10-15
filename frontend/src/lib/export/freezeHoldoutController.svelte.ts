/**
 * /export's test-holdout freeze: the confirm modal's state and the
 * one-shot `POST {API_PREFIX}/test_holdout/freeze`.
 */
import { apiErrorText } from '$lib/api_errors';
import { freezeTestHoldout } from '$lib/api_export';
import { toastStore } from '$stores/toast.svelte';

/** `loadAll` re-reads the page's data after a successful freeze. */
export function createFreezeHoldoutController(loadAll: () => Promise<void>) {
  let freezeOpen = $state<boolean>(false);
  let freezePercent = $state<number>(10);
  let freezeBusy = $state<boolean>(false);

  function openFreeze(): void {
    freezePercent = 10;
    freezeOpen = true;
  }

  function closeFreeze(): void {
    if (freezeBusy) return;
    freezeOpen = false;
  }

  async function submitFreeze(): Promise<void> {
    const ok = window.confirm(
      `Freeze ${freezePercent}% of validated crops as the test set? This is ` +
        'one-shot per dataset version — re-running requires ?force=true ' +
        'and is recorded in the manifest. Selection is deterministic ' +
        '(SHA1 of each crop id, per class) — no seed to pick.',
    );
    if (!ok) return;
    freezeBusy = true;
    try {
      const res = await freezeTestHoldout({ percent: freezePercent });
      toastStore.success(
        `Frozen: ${res.n_frozen} crops across ${res.n_classes_covered} classes via ` +
          `${res.selection} (min ${res.min_per_class}/class).`,
      );
      freezeOpen = false;
      await loadAll();
    } catch (e) {
      toastStore.error(`Freeze failed: ${apiErrorText(e)}`);
    } finally {
      freezeBusy = false;
    }
  }

  return {
    get freezeOpen() {
      return freezeOpen;
    },
    get freezePercent() {
      return freezePercent;
    },
    set freezePercent(v: number) {
      freezePercent = v;
    },
    get freezeBusy() {
      return freezeBusy;
    },
    openFreeze,
    closeFreeze,
    submitFreeze,
  };
}

export type FreezeHoldoutController = ReturnType<typeof createFreezeHoldoutController>;
