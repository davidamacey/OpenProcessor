/**
 * The class picker's state: a combobox over the bounded, served class
 * registry. Typed text only filters that list (`searchClasses`); creating a
 * class goes to `POST /classes`, which stores the typed name as its slug and
 * answers that `class_name`. The element-scoped keydown handler is not a
 * window listener.
 */
import { addClass } from '$lib/api_classes';
import { apiErrorText } from '$lib/api_errors';
import { itemClassTargets, searchClasses } from '$lib/classPicker';
import type { RegistryClass } from '$lib/types_classes';
import { classesStore } from '$stores/classes.svelte';

const MAX_ROWS = 50;

export interface ClassPickerOptions {
  /** `create` offers only the create row (the rail's "+ New class"). */
  mode: () => 'assign' | 'create';
  /** Class names to rank first (the ones used last). */
  recent: () => string[];
  /** Called with the picked, or just created, class. */
  onpick: (cls: RegistryClass) => void;
  onclose: () => void;
}

export function createClassPicker(opts: ClassPickerOptions) {
  let query = $state('');
  let index = $state(0);
  let error = $state<string | null>(null);
  let busy = $state(false);

  const targets = $derived(itemClassTargets(classesStore.classes));
  const recentIds = $derived(
    opts
      .recent()
      .map((n) => targets.find((c) => c.name === n)?.id)
      .filter((id): id is number => id != null),
  );
  const results = $derived(
    opts.mode() === 'create' ? [] : searchClasses(targets, query, MAX_ROWS, recentIds),
  );
  const typed = $derived(query.trim());
  /** The name the last row would create, or null when nothing is typed or it exists. */
  const createName = $derived.by(() => {
    if (!typed) return null;
    return targets.some((c) => c.name === typed) ? null : typed;
  });
  const rowCount = $derived(results.length + (createName ? 1 : 0));

  function reset(): void {
    query = '';
    index = 0;
    error = null;
  }

  function setQuery(v: string): void {
    query = v;
    index = 0;
    error = null;
  }

  async function create(): Promise<void> {
    const name = createName;
    if (!name || busy) return;
    busy = true;
    error = null;
    try {
      const { class_name } = await addClass({ name });
      await classesStore.clearAndRefetch();
      const created = classesStore.classes.find((c) => c.name === class_name);
      if (created) opts.onpick(created);
      opts.onclose();
    } catch (e) {
      error = apiErrorText(e);
    } finally {
      busy = false;
    }
  }

  function pick(cls: RegistryClass): void {
    opts.onpick(cls);
    opts.onclose();
  }

  function activate(): void {
    const cls = results[index];
    if (cls) pick(cls);
    else void create();
  }

  function onKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      e.preventDefault();
      opts.onclose();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      index = Math.min(rowCount - 1, index + 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      index = Math.max(0, index - 1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      activate();
    }
  }

  return {
    get query() {
      return query;
    },
    set query(v: string) {
      setQuery(v);
    },
    get index() {
      return index;
    },
    set index(v: number) {
      index = v;
    },
    get total() {
      return targets.length;
    },
    get results() {
      return results;
    },
    get createName() {
      return createName;
    },
    get error() {
      return error;
    },
    get busy() {
      return busy;
    },
    reset,
    pick,
    create,
    onKeydown,
  };
}

export type ClassPickerController = ReturnType<typeof createClassPicker>;
