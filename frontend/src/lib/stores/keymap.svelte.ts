/**
 * keymapStore — the action-id keymap (docs/design/configurable-keyboard-
 * shortcuts-plan-2026-09-26.md §5.1, step K1).
 *
 * Every rebindable shortcut is an action id (`review.queue.discard`,
 * `cluster.undo`, ...). Pages register handlers by id through
 * `keyboardStore.registerAction`, and every hint, toast and overlay row
 * prints its key through `glyph(id)` — so a binding lives in exactly one
 * place: the current document.
 *
 * K1: the document is always `FALLBACK_KEYMAP` (today's exact keys). K2
 * adds a `loadKeymap()` that reads the scoped `GET {prefix}/keymap` and
 * hands the served document to `setDocument(doc, 'served')`; nothing
 * below changes shape for that.
 *
 * Invariant enforced on every document (plan §0 decision 3): an action
 * never loses one of its `locked_keys`, and no action takes a key in
 * `grammar.locked_keys` unless that key is one of its own locked keys.
 */

import { ApiError, getKeymap } from '$lib/api';
import { onProjectChange } from '$lib/projectChange';
import { formatCompactKey, formatShortcutKey } from '$lib/keyboardDisplay';
import {
  FALLBACK_KEYMAP,
  type KeymapAction,
  type KeymapContext,
  type KeymapDocument,
  type KeymapValidationIssue,
} from '$lib/keymapFallback';

export type KeymapSource = 'fallback' | 'served';

export interface LabelVars {
  /** The region noun, e.g. a slot's `label.singular`. */
  region?: string;
}

const KNOWN_IDS = new Set(FALLBACK_KEYMAP.actions.map((a) => a.id));
const FALLBACK_BY_ID = new Map(FALLBACK_KEYMAP.actions.map((a) => [a.id, a]));

/** An action's effective keys under the document's locked-key rules. */
export function effectiveKeys(action: KeymapAction, doc: KeymapDocument): string[] {
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local lookup set consumed synchronously within this function, never stored in reactive state
  const lockedGrammar = new Set(doc.grammar.locked_keys);
  const own = action.locked_keys ?? [];
  const requested = (action.modifiable ? action.keys : action.default).map((k) =>
    k.toLowerCase(),
  );
  let keys = requested.filter(
    (k, i) => requested.indexOf(k) === i && (!lockedGrammar.has(k) || own.includes(k)),
  );
  keys = [...own.filter((k) => !keys.includes(k)), ...keys];
  const max = doc.grammar.max_combos_per_action;
  if (keys.length > max) {
    const others = keys.filter((k) => !own.includes(k)).slice(0, max - own.length);
    keys = keys.filter((k) => own.includes(k) || others.includes(k));
  }
  return keys;
}

class KeymapStore {
  #doc = $state.raw<KeymapDocument>(FALLBACK_KEYMAP);
  source = $state<KeymapSource>('fallback');

  #byId = $derived.by(() => {
    const doc = this.#doc;
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- rebuilt from scratch on every $derived recompute and returned as an immutable value; reactivity comes from the surrounding $derived.by, not per-key mutation
    const map = new Map<string, KeymapAction & { keys: string[] }>();
    for (const a of doc.actions) {
      // A served id this build doesn't know is ignored: nothing here
      // would ever register a handler for it.
      if (!KNOWN_IDS.has(a.id)) continue;
      map.set(a.id, { ...a, keys: effectiveKeys(a, doc) });
    }
    for (const [id, fb] of FALLBACK_BY_ID) {
      if (map.has(id)) continue;
      if (this.source === 'served') {
        console.warn(`[keymap] served keymap omits "${id}"; using the built-in default`);
      }
      map.set(id, { ...fb, keys: effectiveKeys(fb, FALLBACK_KEYMAP) });
    }
    return map;
  });

  #contextsById = $derived(new Map(this.#doc.contexts.map((c) => [c.id, c])));

  /** Replace the active document. K2's loader calls this with the served doc. */
  setDocument(doc: KeymapDocument, source: KeymapSource): void {
    this.source = source;
    this.#doc = doc;
  }

  /** Back to the built-in defaults. */
  resetToFallback(): void {
    this.setDocument(FALLBACK_KEYMAP, 'fallback');
  }

  get contexts(): KeymapContext[] {
    return this.#doc.contexts;
  }

  /**
   * The served class-hotkey reserved set, or `null` when the keymap isn't
   * served. On `null` callers keep reading `GET {API_PREFIX}/classes`'s own
   * `reserved_hotkeys` (plan §5.1), which stays authoritative.
   */
  get reserved(): string[] | null {
    if (this.source !== 'served') return null;
    return this.#doc.reserved_hotkeys ?? null;
  }

  /** The served document verbatim — the editor's read model. `null` on
   *  fallback (the editor is absent then; see `keymapAvailability`). */
  get document(): KeymapDocument {
    return this.#doc;
  }

  get revision(): number | null {
    return this.source === 'served' ? (this.#doc.revision ?? null) : null;
  }

  get isDefault(): boolean {
    return this.#doc.is_default ?? true;
  }

  get issues(): KeymapValidationIssue[] {
    return this.#doc.issues ?? [];
  }

  action(id: string): KeymapAction | undefined {
    return this.#byId.get(id);
  }

  isAvailable(id: string): boolean {
    return this.#byId.get(id)?.available ?? false;
  }

  /** Effective combos for an action; `[]` when unknown or unbound. */
  keysFor(id: string): string[] {
    const a = this.#byId.get(id);
    if (!a) {
      console.warn(`[keymap] unknown action id "${id}"`);
      return [];
    }
    return a.keys;
  }

  /** Display label; `{region}` becomes `vars.region` (default "region"). */
  label(id: string, vars: LabelVars = {}): string {
    const a = this.#byId.get(id);
    if (!a) return id;
    return a.label.replaceAll('{region}', vars.region ?? 'region');
  }

  /** The first combo, formatted for a hint or toast; `''` when unbound. */
  glyph(id: string): string {
    const k = this.keysFor(id)[0];
    return k ? formatShortcutKey(k) : '';
  }

  /** Every combo, formatted. */
  glyphs(id: string): string[] {
    return this.keysFor(id).map(formatShortcutKey);
  }

  /** The first combo in symbol form (`⇧N`, `↵`, `⌫`); `''` when unbound. */
  compactGlyph(id: string): string {
    const k = this.keysFor(id)[0];
    return k ? formatCompactKey(k) : '';
  }

  /** `context` plus the transitive closure of its `includes`. */
  activeSet(context: string): string[] {
    const out: string[] = [];
    const walk = (id: string) => {
      if (out.includes(id)) return;
      out.push(id);
      for (const inc of this.#contextsById.get(id)?.includes ?? []) walk(inc);
    };
    walk(context);
    return out;
  }

  /** Available actions declared in exactly this context, in document order. */
  actionsForContext(context: string): KeymapAction[] {
    return [...this.#byId.values()].filter((a) => a.context === context && a.available);
  }

  /**
   * The available action bound to `combo` in `context`'s active set, or
   * `null`. The context's own actions win over inherited ones.
   */
  actionFor(context: string, combo: string): string | null {
    const c = combo.toLowerCase();
    for (const ctx of this.activeSet(context)) {
      for (const a of this.actionsForContext(ctx)) {
        if (a.keys.includes(c)) return a.id;
      }
    }
    return null;
  }
}

export const keymapStore = new KeymapStore();

/**
 * `keymapAvailability` — provisional capability gate for the `/settings`
 * Keyboard section (K2, plan §5.1).
 *
 * A pre-W2b backend 404s/501s `GET {prefix}/keymap`: `available` becomes
 * `false`, `keymapStore` stays on `FALLBACK_KEYMAP`, and the editor is
 * ABSENT, not disabled. Any other failure (network, 5xx) leaves
 * `available` at its current, optimistic value — a transient outage must
 * not hide a route that actually exists — and `keymapStore` also stays on
 * the fallback until a load succeeds.
 */
class KeymapAvailabilityStore {
  available = $state<boolean | null>(null);
  #loaded = false;
  #inflight: Promise<void> | null = null;

  async init(): Promise<void> {
    if (this.#loaded) return;
    if (this.#inflight) return this.#inflight;
    this.#inflight = loadKeymap().finally(() => {
      this.#loaded = true;
      this.#inflight = null;
    });
    return this.#inflight;
  }

  /** For a test/dev reset only. */
  reset(): void {
    this.available = null;
    this.#loaded = false;
    this.#inflight = null;
  }

  setAvailable(v: boolean | null): void {
    this.available = v;
  }
}

export const keymapAvailability = new KeymapAvailabilityStore();

/**
 * Bounded 2s x 3 tries, matching `loadRegionProfile()`'s boot pattern
 * (plan §5.1). Called once from the root layout's `load()`; also called
 * by the `config.changed axis=keymap` SSE handler to refetch (unbounded
 * there — a single retry is enough for a live refetch).
 */
const KEYMAP_LOAD_TIMEOUT_MS = 2000;
const KEYMAP_RETRY_DELAYS_MS = [250, 750];

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('keymap load timed out')), ms);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Reads the scoped `GET {prefix}/keymap` and hands the result to
 * `keymapStore`. Never throws.
 *
 * - 404/501 -> `keymapAvailability.available = false`, stays on the
 *   fallback silently (this is expected on every deployment until
 *   OpenProcessor W2b lands).
 * - Any other failure on every try -> availability left unchanged, stays
 *   on the fallback.
 * - Success -> `keymapAvailability.available = true`,
 *   `keymapStore.setDocument(doc, 'served')`.
 */
export async function loadKeymap(): Promise<void> {
  for (let attempt = 0; attempt < 1 + KEYMAP_RETRY_DELAYS_MS.length; attempt++) {
    try {
      const doc = await withTimeout(getKeymap(), KEYMAP_LOAD_TIMEOUT_MS);
      keymapAvailability.setAvailable(true);
      keymapStore.setDocument(doc, 'served');
      return;
    } catch (e) {
      if (e instanceof ApiError && (e.status === 404 || e.status === 501)) {
        keymapAvailability.setAvailable(false);
        return;
      }
      if (attempt < KEYMAP_RETRY_DELAYS_MS.length) {
        await sleep(KEYMAP_RETRY_DELAYS_MS[attempt]);
      }
    }
  }
}

// The keymap is a per-project axis: a switch drops back to the fallback
// document and re-probes availability; the /p/[project] layout then
// loads the new project's served keymap.
onProjectChange(() => {
  keymapStore.resetToFallback();
  keymapAvailability.reset();
});
