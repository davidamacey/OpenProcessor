<script lang="ts">
  /**
   * The editor's tools, zoom and undo as buttons. Each one is also a keymap action; the key it shows
   * is read from the keymap, so a rebind changes the hint. Presentation only: it calls back.
   */
  import { keymapStore } from '$stores/keymap.svelte';
  import type { EditorTool } from '$components/editor/EditorCanvas.svelte';

  interface Props {
    tool: EditorTool;
    sticky: boolean;
    labels: boolean;
    zoomPercent: number;
    canUndo: boolean;
    canRedo: boolean;
    undoLabel: string | null;
    redoLabel: string | null;
    ontool: (tool: EditorTool, keepDrawing?: boolean) => void;
    onzoomin: () => void;
    onzoomout: () => void;
    onfit: () => void;
    onactual: () => void;
    onundo: () => void;
    onredo: () => void;
    onlabels: () => void;
  }

  let {
    tool,
    sticky,
    labels,
    zoomPercent,
    canUndo,
    canRedo,
    undoLabel,
    redoLabel,
    ontool,
    onzoomin,
    onzoomout,
    onfit,
    onactual,
    onundo,
    onredo,
    onlabels,
  }: Props = $props();

  const BTN =
    'flex h-9 min-w-9 items-center justify-center rounded-md border border-zinc-700 px-2 text-sm text-zinc-200 hover:bg-zinc-800 focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:outline-none disabled:opacity-40 aria-pressed:border-sky-400 aria-pressed:bg-sky-500/20';

  function hint(id: string, text: string): string {
    const key = keymapStore.glyph(id);
    return key ? `${text} (${key})` : text;
  }
</script>

{#snippet button(
  id: string,
  text: string,
  glyph: string,
  onclick: () => void,
  pressed?: boolean,
  disabled?: boolean,
  title?: string,
)}
  <button
    type="button"
    class={BTN}
    aria-label={text}
    aria-pressed={pressed}
    {disabled}
    title={title ?? hint(id, text)}
    data-action={id}
    {onclick}
  >
    <span aria-hidden="true">{glyph}</span>
  </button>
{/snippet}

<div
  class="flex flex-row flex-wrap items-center gap-1 lg:flex-col lg:items-stretch"
  role="toolbar"
  aria-label="Editor tools"
  data-testid="tool-strip"
>
  {@render button(
    'editor.tool_select',
    'Select and move',
    '↖',
    () => ontool('select'),
    tool === 'select',
  )}
  {@render button(
    'editor.tool_draw',
    'Draw a box',
    '▭',
    () => ontool('draw'),
    tool === 'draw' && !sticky,
  )}
  {@render button(
    'editor.tool_draw_sticky',
    'Draw boxes, stay on',
    '▭▭',
    () => ontool('draw', true),
    tool === 'draw' && sticky,
  )}
  {@render button('editor.tool_pan', 'Pan', '✋', () => ontool('pan'), tool === 'pan')}
  <span class="mx-1 hidden h-px bg-zinc-800 lg:block" role="separator"></span>
  {@render button('editor.zoom_in', 'Zoom in', '+', onzoomin)}
  {@render button('editor.zoom_out', 'Zoom out', '−', onzoomout)}
  {@render button('editor.zoom_fit', 'Fit the picture', 'Fit', onfit)}
  {@render button(
    'editor.zoom_actual',
    'Actual size',
    `${Math.round(zoomPercent)}%`,
    onactual,
  )}
  <span class="mx-1 hidden h-px bg-zinc-800 lg:block" role="separator"></span>
  {@render button(
    'editor.undo',
    'Undo',
    '↶',
    onundo,
    undefined,
    !canUndo,
    hint('editor.undo', undoLabel ? `Undo: ${undoLabel}` : 'Undo'),
  )}
  {@render button(
    'editor.redo',
    'Redo',
    '↷',
    onredo,
    undefined,
    !canRedo,
    hint('editor.redo', redoLabel ? `Redo: ${redoLabel}` : 'Redo'),
  )}
  {@render button('editor.labels', 'Box labels', 'Aa', onlabels, labels)}
</div>
