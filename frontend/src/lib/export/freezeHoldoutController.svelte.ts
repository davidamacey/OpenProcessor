/**
 * /export's test-holdout freeze: the confirm modal's state and
 * `POST {API_PREFIX}/test_holdout/freeze`. The modal is the one confirmation; a refreeze
 * (a holdout already exists) sends `?force=true`.
 */
import { apiErrorText } from '$lib/api_errors';
import { freezeTestHoldout } from '$lib/api_export';
import { nearDupStaleLevels } from '$lib/nearDup/staleRefusal';
import { createNearDupRefreshController } from '$lib/nearDup/nearDupRefreshController.svelte';
import { toastStore } from '$stores/toast.svelte';

/** `loadAll` re-reads the page's data after a successful freeze. */
export function createFreezeHoldoutController(loadAll: () => Promise<void>) {
  let freezeOpen = $state<boolean>(false);
  let freezePercent = $state<number>(10);
  let freezeBusy = $state<boolean>(false);
  let freezeForce = $state<boolean>(false);
  const nearDup = createNearDupRefreshController();

  function openFreeze(opts: { refreeze?: boolean } = {}): void {
    freezePercent = 10;
    freezeForce = opts.refreeze === true;
    nearDup.clear();
    freezeOpen = true;
  }

  function closeFreeze(): void {
    if (freezeBusy) return;
    nearDup.dispose();
    freezeOpen = false;
  }

  async function submitFreeze(): Promise<void> {
    freezeBusy = true;
    try {
      const res = await freezeTestHoldout({
        percent: freezePercent,
        force: freezeForce || undefined,
      });
      toastStore.success(
        `Frozen: ${res.n_frozen} crops across ${res.n_classes_covered} classes via ` +
          `${res.selection} (min ${res.min_per_class}/class)` +
          (res.moved_by_near_dup
            ? `, ${res.moved_by_near_dup} pulled in as near-duplicates.`
            : '.'),
      );
      freezeOpen = false;
      await loadAll();
    } catch (e) {
      const levels = nearDupStaleLevels(e);
      if (levels) nearDup.show(levels);
      else toastStore.error(`Freeze failed: ${apiErrorText(e)}`);
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
    get freezeForce() {
      return freezeForce;
    },
    get freezeBusy() {
      return freezeBusy;
    },
    nearDup,
    openFreeze,
    closeFreeze,
    submitFreeze,
  };
}

export type FreezeHoldoutController = ReturnType<typeof createFreezeHoldoutController>;
