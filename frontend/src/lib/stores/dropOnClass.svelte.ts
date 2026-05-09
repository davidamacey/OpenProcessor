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

class DropOnClassStore {
  /** Active drop handler. ``null`` means the sidebar's drop targets are inert. */
  handler = $state<((cls: OpClass) => void | Promise<void>) | null>(null);

  /** Register a page-scoped handler. Returns the unregister function. */
  register(fn: (cls: OpClass) => void | Promise<void>): () => void {
    this.handler = fn;
    return () => {
      if (this.handler === fn) this.handler = null;
    };
  }

  /** Called by the sidebar when a drop happens on a class row. */
  async dispatch(cls: OpClass): Promise<void> {
    if (this.handler) await this.handler(cls);
  }
}

export const dropOnClassStore = new DropOnClassStore();
