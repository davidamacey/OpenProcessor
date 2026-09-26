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

import { formatCompactKey, formatShortcutKey } from '$lib/keyboardDisplay';
import {
  FALLBACK_KEYMAP,
  type KeymapAction,
  type KeymapContext,
  type KeymapDocument,
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
