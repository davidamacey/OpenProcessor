/**
 * KeyboardStore — global keyboard shortcut registry.
 *
 * Handlers register themselves with a key combo and a scope name. Scopes are
 * stack-managed: a page that sets its scope to "cluster" means handlers
 * registered with scope: 'cluster' fire, plus all scope: 'global' handlers.
 *
 * Handlers receive the raw KeyboardEvent so they can `preventDefault` if the
 * action consumed the keystroke.
 *
 * Modifier-key handling: keys are normalized as
 *   `ctrl+shift+enter`, `shift+enter`, `~`, `1`, `arrowleft`, etc.
 *
 * We never fire when the active element is an input, textarea, contenteditable,
 * or select.
 *
 * Action ids (docs/design/configurable-keyboard-shortcuts-plan-2026-09-26.md
 * §5.2): `registerAction('review.queue.discard', fn, scope)` stores the id,
 * not a combo — dispatch resolves the id's keys through `keymapStore` at
 * keypress time, so a rebind applies with no re-registration. The legacy
 * `register(combo, ...)` form stays for combo-level tests and callers.
 */

import type { KeyboardShortcut } from '$lib/types';
import { keymapStore, type LabelVars } from './keymap.svelte';

export type ShortcutHandler = (
  e: KeyboardEvent,
) => void | boolean | Promise<void | boolean>;

interface Registration {
  /** Set for a legacy combo registration. */
  combo?: string;
  /** Set for an action registration. */
  actionId?: string;
  /** Explicit keys for an action (a slot-declared keymap), else the store's. */
  keys?: string[];
  scope: string;
  description: string;
  handler: ShortcutHandler;
}

export interface RegisterActionOptions {
  /** Pin this registration to these combos instead of the keymap's keys
   *  (a tier-2 slot's own declared `queue.keymap`). */
  keys?: string[];
  /** Overlay text; defaults to the keymap label. */
  description?: string;
  /** Substitutions for the keymap label (`{region}`). */
  labelVars?: LabelVars;
}

// Before configurable keys the overlay toggle matched every backtick
// spelling a layout/browser produces, including the physical key. Kept
// only while the toggle is still bound to backtick, so a rebind away from
// it isn't shadowed (plan §2.2 drops these in K2).
const OVERLAY_LAYOUT_ALIASES = ['`', '~', 'shift+`', 'shift+~'];

function regKeys(r: Registration): string[] {
  if (r.combo !== undefined) return [r.combo];
  return r.keys ?? keymapStore.keysFor(r.actionId!);
}

export function normalize(e: KeyboardEvent): string {
  const parts: string[] = [];
  if (e.ctrlKey) parts.push('ctrl');
  if (e.metaKey) parts.push('meta');
  if (e.altKey) parts.push('alt');
  if (e.shiftKey) parts.push('shift');
  parts.push(e.key.toLowerCase());
  return parts.join('+');
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (target.isContentEditable) return true;
  return false;
}

class KeyboardStore {
  scope = $state<string>('global');
  overlayOpen = $state<boolean>(false);

  #regs: Registration[] = [];
  #listenerInstalled = false;
  #suspended = false;

  constructor() {
    // m11 (2026-09-24 interactive pass): the listener used to install lazily
    // on the first register() call, so `~`/`` ` `` did nothing on pages that
    // register no shortcuts of their own (/clusters, /classes, /dashboard).
    // The overlay toggle is a global affordance, so install at boot.
    this.#install();
  }

  /** Set the current page-scope. Pages call this in $effect on mount. */
  setScope(scope: string): void {
    this.scope = scope;
  }

  register(
    combo: string,
    handler: ShortcutHandler,
    scope = 'global',
    description = '',
  ): () => void {
    const reg: Registration = {
      combo: combo.toLowerCase(),
      scope,
      description,
      handler,
    };
    this.#regs.push(reg);
    this.#install();
    return () => {
      const i = this.#regs.indexOf(reg);
      if (i >= 0) this.#regs.splice(i, 1);
    };
  }

  /**
   * Register a handler for a keymap action id. Its keys are resolved at
   * keypress time (or pinned via `opts.keys`); an action the keymap marks
   * unavailable is never registered.
   */
  registerAction(
    actionId: string,
    handler: ShortcutHandler,
    scope = 'global',
    opts: RegisterActionOptions = {},
  ): () => void {
    if (!keymapStore.isAvailable(actionId)) return () => {};
    const reg: Registration = {
      actionId,
      keys: opts.keys?.map((k) => k.toLowerCase()),
      scope,
      description: opts.description ?? keymapStore.label(actionId, opts.labelVars),
      handler,
    };
    this.#regs.push(reg);
    this.#install();
    return () => {
      const i = this.#regs.indexOf(reg);
      if (i >= 0) this.#regs.splice(i, 1);
    };
  }

  /**
   * One row per ACTION (registration), not per key. A multi-key action
   * (e.g. "Step back" on `arrowleft`/`b`) used to print once per key —
   * K2 fix (plan §5, item 3): every combo it owns lists together on one
   * row, sorted by the first (lowest) combo for a stable order.
   */
  shortcutsForCurrentScope(): KeyboardShortcut[] {
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local dedup set consumed synchronously within this call, never stored in reactive state
    const seen = new Set<string>();
    const out: KeyboardShortcut[] = [];
    for (const r of this.#regs) {
      if (r.scope !== 'global' && r.scope !== this.scope) continue;
      if (!r.description) continue;
      const keys = regKeys(r);
      if (keys.length === 0) continue;
      const dedupeKey = `${r.scope}:${r.actionId ?? r.description}`;
      if (seen.has(dedupeKey)) continue;
      seen.add(dedupeKey);
      out.push({ keys, scope: r.scope, description: r.description });
    }
    return out.sort((a, b) => a.keys[0].localeCompare(b.keys[0]));
  }

  /**
   * Belt-and-braces guard for the layout's class-hotkey listener (plan
   * §5.3): a registered action shortcut beats a class hotkey bound to
   * the same key, in a context where `class_hotkeys_live`. Returns
   * `true` when some CURRENTLY-ACTIVE registration (global scope or the
   * current page scope) owns `combo`.
   */
  hasActiveBinding(combo: string): boolean {
    const c = combo.toLowerCase();
    for (const r of this.#regs) {
      if (r.scope !== 'global' && r.scope !== this.scope) continue;
      if (regKeys(r).includes(c)) return true;
    }
    return false;
  }

  /**
   * Suspends dispatch while a key-capture widget (the `/settings`
   * editor) is recording a new combo, so a captured key never also fires
   * whatever it's currently bound to. `resume()` undoes it; both are
   * idempotent.
   */
  suspend(): void {
    this.#suspended = true;
  }

  resume(): void {
    this.#suspended = false;
  }

  get suspended(): boolean {
    return this.#suspended;
  }

  toggleOverlay(): void {
    this.overlayOpen = !this.overlayOpen;
  }

  closeOverlay(): void {
    this.overlayOpen = false;
  }

  #install(): void {
    if (this.#listenerInstalled) return;
    if (typeof window === 'undefined') return;
    this.#listenerInstalled = true;
    window.addEventListener('keydown', (e: KeyboardEvent) => this.#dispatch(e));
  }

  #dispatch(e: KeyboardEvent): void {
    if (this.#suspended) return;
    if (isTypingTarget(e.target)) return;
    const combo = normalize(e);
    // m11 (2026-09-24 interactive pass): on a US layout, Shift+` reports
    // e.key === '~', which normalize() renders as "shift+~"; some
    // browsers/layouts still report e.key === '`' while shiftKey is true,
    // hence the physical-key (e.code) match too.
    const overlayKeys = keymapStore.keysFor('global.shortcuts_overlay');
    const layoutAlias =
      overlayKeys.includes('`') &&
      (OVERLAY_LAYOUT_ALIASES.includes(combo) || e.code === 'Backquote');
    if (overlayKeys.includes(combo) || layoutAlias) {
      e.preventDefault();
      this.toggleOverlay();
      return;
    }
    if (this.overlayOpen && keymapStore.keysFor('global.close_overlay').includes(combo)) {
      e.preventDefault();
      this.closeOverlay();
      return;
    }
    for (const r of this.#regs) {
      if (!regKeys(r).includes(combo)) continue;
      if (r.scope !== 'global' && r.scope !== this.scope) continue;
      const result = r.handler(e);
      // Treat undefined as "consumed".
      if (result !== false) {
        e.preventDefault();
      }
      return;
    }
  }
}

export const keyboardStore = new KeyboardStore();
