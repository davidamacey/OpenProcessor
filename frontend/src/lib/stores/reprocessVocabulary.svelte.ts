/**
 * The Reprocess vocabulary (`reprocess` block of `GET /config/vocabulary`,
 * OpenProcessor 5441975e): the served label and description of every
 * scope, filter field, job status and lock reason. Read once per project.
 * An id the vocabulary does not list, or any id before it loads, prints as
 * served, never as a guessed label.
 */
import { getConfigVocabulary } from '$lib/api';
import { onProjectChange } from '$lib/projectChange';
import type { ReprocessVocabulary } from '$lib/types_profiles';

export type ReprocessVocabularyKind = keyof ReprocessVocabulary;

class ReprocessVocabularyStore {
  vocabulary = $state<ReprocessVocabulary | null>(null);
  loaded = $state(false);
  #inflight: Promise<void> | null = null;
  #gen = 0;

  async init(): Promise<void> {
    if (this.loaded) return;
    if (this.#inflight) return this.#inflight;
    const gen = this.#gen;
    this.#inflight = (async () => {
      try {
        const v = await getConfigVocabulary(false);
        if (gen === this.#gen) this.vocabulary = v.reprocess ?? null;
      } catch {
        if (gen === this.#gen) this.vocabulary = null;
      } finally {
        if (gen === this.#gen) {
          this.loaded = true;
          this.#inflight = null;
        }
      }
    })();
    return this.#inflight;
  }

  #entry(kind: ReprocessVocabularyKind, id: string) {
    return this.vocabulary?.[kind].find((e) => e.id === id);
  }

  label(kind: ReprocessVocabularyKind, id: string): string {
    return this.#entry(kind, id)?.label ?? id;
  }

  description(kind: ReprocessVocabularyKind, id: string): string | null {
    return this.#entry(kind, id)?.description ?? null;
  }

  /** Tooltip for a lock badge: the item serves only a locked flag, so this
   *  lists the served lock rule rather than naming one reason. */
  lockText(): string {
    const reasons = this.vocabulary?.lock_reasons ?? [];
    if (reasons.length === 0) return 'Locked';
    return `Locked. Locked when: ${reasons.map((r) => `${r.label}: ${r.description}`).join('; ')}`;
  }

  resetForProjectChange(): void {
    this.#gen += 1;
    this.#inflight = null;
    this.vocabulary = null;
    this.loaded = false;
  }
}

export const reprocessVocabularyStore = new ReprocessVocabularyStore();
onProjectChange(() => reprocessVocabularyStore.resetForProjectChange());
