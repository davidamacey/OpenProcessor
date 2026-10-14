/**
 * `/review` class picker (P1-4): a fuzzy-search combobox over every
 * assignable class, opened with the class-picker key or the "search all
 * classes" button, assigning the picked class to the current item.
 *
 * Moved verbatim out of `review/+page.svelte`.
 */

import { itemClassTargets, searchClasses } from '$lib/classPicker';
import { isSlotTab } from '$lib/reviewTabs';
import type { ReviewItemActions } from '$lib/review/reviewItemActions.svelte';
import type { ReviewQueueLoader } from '$lib/review/reviewQueueLoader.svelte';
import type { ReviewTabsController } from '$lib/review/reviewTabsController.svelte';
import type { RegistryClass } from '$lib/types';
import { classesStore } from '$stores/classes.svelte';

export interface ReviewClassPickerOptions {
  tabs: ReviewTabsController;
  loader: ReviewQueueLoader;
  actions: ReviewItemActions;
}

export function createReviewClassPicker(opts: ReviewClassPickerOptions) {
  const { assign } = opts.actions;
  const tab = $derived(opts.tabs.tab);
  const current = $derived(opts.loader.current);
  const currentHintIds = $derived(opts.actions.currentHintIds);

  const pickerClasses = $derived(itemClassTargets(classesStore.classes));

  // -- class picker (P1-4) ---------------------------------------------
  // Fuzzy-search combobox over *every* non-deprecated class, opened with
  // '/'. topClasses caps quick-assign at the 10 most-validated
  // classes; 73 of 84 need a round-trip to /classes without this. Pure
  // filter/ranking logic lives in $lib/classPicker.ts (searchClasses) so
  // it's unit-testable without @testing-library/svelte.
  let pickerOpen = $state(false);
  let pickerQuery = $state('');
  let pickerIndex = $state(0);
  let pickerInputEl = $state<HTMLInputElement | null>(null);

  const pickerResults = $derived(
    searchClasses(pickerClasses, pickerQuery, 50, currentHintIds),
  );

  // Re-center the highlighted row on the best match whenever the query
  // (re-ranks the list) changes. Doesn't fire on arrow-key navigation,
  // which only touches pickerIndex.
  $effect(() => {
    void pickerQuery;
    pickerIndex = 0;
  });

  function openPicker(): void {
    if (isSlotTab(tab) || !current) return;
    pickerOpen = true;
    pickerQuery = '';
    pickerIndex = 0;
    // Input isn't in the DOM until this render commits.
    requestAnimationFrame(() => pickerInputEl?.focus());
  }

  function closePicker(): void {
    pickerOpen = false;
    pickerQuery = '';
    pickerIndex = 0;
  }

  async function pickClass(cls: RegistryClass): Promise<void> {
    closePicker();
    await assign(cls.id);
  }

  // Element-scoped handler on the picker's own <input> — not a second
  // window keydown listener (keyboard.svelte.test.ts's guard). Browser
  // focus already keeps this from colliding with keyboardStore/the
  // layout's per-class dispatcher: both treat a focused <input> as a
  // typing target and skip it entirely.
  function onPickerKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      e.preventDefault();
      closePicker();
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      pickerIndex = Math.min(pickerResults.length - 1, pickerIndex + 1);
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      pickerIndex = Math.max(0, pickerIndex - 1);
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      const cls = pickerResults[pickerIndex];
      if (cls) void pickClass(cls);
    }
  }

  return {
    get pickerClasses() {
      return pickerClasses;
    },
    get pickerOpen() {
      return pickerOpen;
    },
    get pickerQuery() {
      return pickerQuery;
    },
    set pickerQuery(v: string) {
      pickerQuery = v;
    },
    get pickerIndex() {
      return pickerIndex;
    },
    set pickerIndex(v: number) {
      pickerIndex = v;
    },
    get pickerInputEl() {
      return pickerInputEl;
    },
    set pickerInputEl(v: HTMLInputElement | null) {
      pickerInputEl = v;
    },
    get pickerResults() {
      return pickerResults;
    },
    openPicker,
    closePicker,
    pickClass,
    onPickerKeydown,
  };
}

export type ReviewClassPicker = ReturnType<typeof createReviewClassPicker>;
