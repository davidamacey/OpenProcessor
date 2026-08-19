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
 */

import type { KeyboardShortcut } from '$lib/types';

export type ShortcutHandler = (
  e: KeyboardEvent,
) => void | boolean | Promise<void | boolean>;

interface Registration {
  combo: string;
  scope: string;
  description: string;
  handler: ShortcutHandler;
}

function normalize(e: KeyboardEvent): string {
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

  shortcutsForCurrentScope(): KeyboardShortcut[] {
    const seen = new Set<string>();
    const out: KeyboardShortcut[] = [];
    for (const r of this.#regs) {
      if (r.scope !== 'global' && r.scope !== this.scope) continue;
      const k = `${r.scope}:${r.combo}`;
      if (seen.has(k)) continue;
      seen.add(k);
      if (!r.description) continue;
      out.push({ key: r.combo, scope: r.scope, description: r.description });
    }
    return out.sort((a, b) => a.key.localeCompare(b.key));
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
    if (isTypingTarget(e.target)) return;
    const combo = normalize(e);
    // Built-in: ~ toggles overlay, Esc closes overlay.
    if (combo === '`' || combo === '~' || combo === 'shift+`') {
      e.preventDefault();
      this.toggleOverlay();
      return;
    }
    if (combo === 'escape' && this.overlayOpen) {
      e.preventDefault();
      this.closeOverlay();
      return;
    }
    for (const r of this.#regs) {
      if (r.combo !== combo) continue;
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
