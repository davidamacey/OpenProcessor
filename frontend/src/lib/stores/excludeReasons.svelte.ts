/**
 * The served ignore reasons (`GET {P}/exclude_reasons`), in display order.
 * A failed read leaves the list empty and the menu offers only the one-click
 * default; the browser never carries its own list.
 */
import { getExcludeReasons } from '$lib/api_cropWrites';
import { onProjectChange } from '$lib/projectChange';
import type { ExcludeReasonEntry } from '$lib/types_itemFilter';

class ExcludeReasonsStore {
  reasons = $state<ExcludeReasonEntry[]>([]);
  loaded = $state(false);
  #inflight: Promise<void> | null = null;
  #gen = 0;

  init(): Promise<void> {
    if (this.loaded) return Promise.resolve();
    if (this.#inflight) return this.#inflight;
    const gen = this.#gen;
    const run = (async () => {
      try {
        const reasons = await getExcludeReasons();
        if (gen === this.#gen) this.reasons = reasons;
      } catch {
        if (gen === this.#gen) this.reasons = [];
      } finally {
        if (gen === this.#gen) {
          this.loaded = true;
          this.#inflight = null;
        }
      }
    })();
    this.#inflight = run;
    return run;
  }

  reset(): void {
    this.#gen += 1;
    this.#inflight = null;
    this.reasons = [];
    this.loaded = false;
  }
}

export const excludeReasonsStore = new ExcludeReasonsStore();
onProjectChange(() => excludeReasonsStore.reset());
