/**
 * Binds the editor's keymap actions to the controller. Each `editor.*` id is registered with the
 * keyboard store, which resolves its keys at keypress time, so a rebind in Settings applies with
 * no re-registration. A class hotkey sets the selected object's class through the drop bus the
 * layout's letter listener already feeds. Nothing here listens on `window`.
 */
import { keyboardStore } from '$stores/keyboard.svelte';
import { dropOnClassStore } from '$stores/dropOnClass.svelte';
import type { EditorController } from './editorController.svelte';

/** The scope the editor's actions live in. */
export const EDITOR_SCOPE = 'editor';

type Direction = readonly [dx: number, dy: number];
const DIRECTIONS: Record<string, Direction> = {
  up: [0, -1],
  down: [0, 1],
  left: [-1, 0],
  right: [1, 0],
};

/** The action id -> handler table. A handler returning `false` leaves the key to the browser. */
function handlers(c: EditorController): Record<string, () => unknown> {
  const out: Record<string, () => unknown> = {
    'editor.tool_select': () => c.setTool('select'),
    'editor.tool_draw': () => c.setTool('draw'),
    'editor.tool_draw_sticky': () => c.setTool('draw', true),
    'editor.tool_pan': () => c.setTool('pan'),
    'editor.zoom_in': c.zoomIn,
    'editor.zoom_out': c.zoomOut,
    'editor.zoom_fit': c.zoomFit,
    'editor.zoom_actual': c.zoomActual,
    'editor.delete': () => (c.selected ? c.deleteSelected() : false),
    'editor.next_box': () => (c.objects.length ? c.selectNext() : false),
    'editor.prev_box': () => (c.objects.length ? c.selectPrev() : false),
    'editor.regions': c.editRegions,
    'editor.labels': c.toggleLabels,
    'editor.undo': c.undo,
    'editor.redo': c.redo,
    'editor.class_picker': c.openClassPicker,
    'editor.prev_image': c.prevImage,
    'editor.next_image': c.nextImage,
    'editor.cancel': c.escape,
  };
  for (const [name, [dx, dy]] of Object.entries(DIRECTIONS)) {
    out[`editor.nudge_${name}`] = () => (c.selected ? c.nudge(dx, dy) : false);
    out[`editor.nudge_${name}_far`] = () => (c.selected ? c.nudgeFar(dx, dy) : false);
    out[`editor.resize_${name}`] = () => (c.selected ? c.resizeCorner(dx, dy) : false);
  }
  return out;
}

/** Registers every editor action and the class hotkey; returns the function that undoes it. */
export function registerEditorKeys(c: EditorController): () => void {
  const offs = Object.entries(handlers(c)).map(([id, run]) =>
    keyboardStore.registerAction(
      id,
      () => {
        const result = run();
        return result === false ? false : undefined;
      },
      EDITOR_SCOPE,
    ),
  );
  const offHotkey = dropOnClassStore.register((cls) => c.setClass(cls));
  const previous = keyboardStore.scope;
  keyboardStore.setScope(EDITOR_SCOPE);
  return () => {
    for (const off of offs) off();
    offHotkey();
    keyboardStore.setScope(previous);
  };
}
