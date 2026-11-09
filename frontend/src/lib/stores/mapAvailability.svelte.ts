/**
 * Whether the active project has a map, as `GET /map` serves it (a stored photo position and a
 * location policy that allows serving it). The nav link shows only when it does; a failed read
 * hides it rather than guess.
 */
import { SvelteMap } from 'svelte/reactivity';
import { getMapMeta } from '$lib/api_map';

class MapAvailabilityStore {
  #available = new SvelteMap<string, boolean>();
  #seq = new Map<string, number>();

  availableFor(slug: string | null | undefined): boolean {
    return slug ? (this.#available.get(slug) ?? false) : false;
  }

  /** Reads `GET /map` for the project the scoped prefix points at. Never throws. */
  async load(slug: string): Promise<void> {
    const mine = (this.#seq.get(slug) ?? 0) + 1;
    this.#seq.set(slug, mine);
    try {
      const meta = await getMapMeta();
      if (this.#seq.get(slug) === mine) this.#available.set(slug, meta.available);
    } catch {
      if (this.#seq.get(slug) === mine) this.#available.delete(slug);
    }
  }
}

export const mapAvailabilityStore = new MapAvailabilityStore();
