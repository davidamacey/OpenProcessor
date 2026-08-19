/**
 * Cross-component DnD bus for "drop crops onto a class".
 *
 * The layout-level ClassSidebar lives outside the per-page tree, so the
 * cluster page can't pass an ondrop handler directly. This store lets
 * the active page register a handler on mount and the sidebar invoke it
 * on drop. Only one handler at a time — pages must register on mount
 * and unregister on destroy.
 *
 * Pattern: the store IS the contract, so there's no risk of the wrong
 * page receiving a drop (it's last-registered-wins, and components that
 * don't register simply don't accept drops).
 */

import type { OpClass } from '$lib/types';

type DropHandler = (cls: OpClass, droppedIds: string[]) => void | Promise<void>;

class DropOnClassStore {
  /** Active drop handler. ``null`` means the sidebar's drop targets are inert. */
  handler = $state<DropHandler | null>(null);

  /** Register a page-scoped handler. Returns the unregister function. */
  register(fn: DropHandler): () => void {
    this.handler = fn;
    return () => {
      if (this.handler === fn) this.handler = null;
    };
  }

  /**
   * Called by the sidebar when a drop happens on a class row, and by the
   * layout's class-letter keydown listener.
   *
   * Two callers, two contracts:
   *
   * - **Drop**: ``droppedIds`` carries the crops the sidebar saw on
   *   finalize. The handler must prefer its own drag capture when it has
   *   one (the sidebar only ever sees the single shadow item, so a
   *   multi-drag arrives here as 1 id), and must NOT fall back to
   *   page-level selection — the user can drag an un-selected card.
   * - **Hotkey**: ``droppedIds`` is empty and there is no drag context at
   *   all. Only then may the handler fall back to its ``selected`` set;
   *   without that fallback the keyboard-first labeling path does nothing.
   */
  async dispatch(cls: OpClass, droppedIds: string[]): Promise<void> {
    if (this.handler) await this.handler(cls, droppedIds);
  }
}

export const dropOnClassStore = new DropOnClassStore();
