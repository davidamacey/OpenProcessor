/**
 * The one `box_edit` key handler (docs/design/configurable-keyboard-
 * shortcuts-plan-2026-09-26.md §1.1 row 24, §5.2). `/review`'s edit mode
 * (via `BboxCanvas.handleKey`) and the `SlotBboxEditor` modal used to each
 * carry their own hardcoded switch over the same keys; both now resolve a
 * keypress to a `box_edit.*` action id through the keymap and run it
 * against whichever handlers they supply.
 */

import { normalize } from '$stores/keyboard.svelte';
import { keymapStore } from '$stores/keymap.svelte';

export interface BoxEditHandlers {
  /** `box_edit.save` — omitted where another registration owns it. */
  save?: () => void | Promise<void>;
  /** `box_edit.cancel` — omitted where another registration owns it. */
  cancel?: () => void;
  nudge: (dx: number, dy: number) => void;
  nudgeRightEdge: (dx: number) => void;
  deleteBox: () => void;
}

/**
 * The `box_edit` action a keypress triggers, or `null`. The bare key is
 * matched too: the original switch keyed on `e.key` alone, so Shift or
 * Ctrl held down never stopped a nudge.
 */
export function boxEditActionFor(e: KeyboardEvent): string | null {
  return (
    keymapStore.actionFor('box_edit', normalize(e)) ??
    keymapStore.actionFor('box_edit', e.key.toLowerCase())
  );
}

/**
 * Run the `box_edit` action for `e`. Returns true when a supplied handler
 * consumed the key (the caller then calls `preventDefault`).
 */
export function runBoxEditKey(
  e: KeyboardEvent,
  h: BoxEditHandlers,
  step: number,
): boolean {
  switch (boxEditActionFor(e)) {
    case 'box_edit.save':
      if (!h.save) return false;
      void h.save();
      return true;
    case 'box_edit.cancel':
      if (!h.cancel) return false;
      h.cancel();
      return true;
    case 'box_edit.delete_box':
      h.deleteBox();
      return true;
    case 'box_edit.nudge_up':
      h.nudge(0, -step);
      return true;
    case 'box_edit.nudge_down':
      h.nudge(0, step);
      return true;
    case 'box_edit.nudge_left':
      h.nudge(-step, 0);
      return true;
    case 'box_edit.nudge_right':
      h.nudge(step, 0);
      return true;
    case 'box_edit.shrink_right':
      h.nudgeRightEdge(-step);
      return true;
    case 'box_edit.grow_right':
      h.nudgeRightEdge(step);
      return true;
    default:
      return false;
  }
}
