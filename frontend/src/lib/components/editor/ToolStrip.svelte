<script lang="ts">
  /**
   * The editor's tools, zoom and undo as segmented groups of buttons (a vertical toolbar from lg up, 36px buttons, the active tool in the accent). Each one is also a keymap action; the key it shows
   * is read from the keymap, so a rebind changes the hint. Presentation only: it calls back.
   */
  import { keymapStore } from '$stores/keymap.svelte';
  import type { EditorTool } from '$components/editor/EditorCanvas.svelte';
  import Icon from '$components/ui/Icon.svelte';
  import type { IconName } from '$components/ui/iconRegistry';

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

  const GROUP =
    'inline-flex flex-row gap-0.5 rounded-control border border-line bg-surface-1 p-0.5 shadow-e1 lg:flex-col';
  const BTN =
    'flex size-9 items-center justify-center rounded-chip text-control text-fg-2 hover:bg-surface-2 hover:text-fg-1 disabled:opacity-40 disabled:hover:bg-transparent aria-pressed:bg-accent-soft aria-pressed:text-accent aria-pressed:shadow-[inset_0_0_0_1px_var(--color-accent-line)]';

  function hint(id: string, text: string): string {
    const key = keymapStore.glyph(id);
    return key ? `${text} (${key})` : text;
  }
</script>

{#snippet button(
  id: string,
  text: string,
  icon: IconName | null,
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
    {#if icon}
      <Icon name={icon} size={18} />
    {:else}
      <span class="text-meta font-medium tabular-nums">{Math.round(zoomPercent)}%</span>
    {/if}
  </button>
{/snippet}

<div
  class="flex flex-row flex-wrap items-center gap-2 lg:flex-col lg:items-stretch"
  role="toolbar"
  aria-label="Editor tools"
  data-testid="tool-strip"
>
  <div class={GROUP} role="group" aria-label="Tool">
    {@render button(
      'editor.tool_select',
      'Select and move',
      'mouse-pointer-2',
      () => ontool('select'),
      tool === 'select',
    )}
    {@render button(
      'editor.tool_draw',
      'Draw a box',
      'square',
      () => ontool('draw'),
      tool === 'draw' && !sticky,
    )}
    {@render button(
      'editor.tool_draw_sticky',
      'Draw boxes, stay on',
      'group',
      () => ontool('draw', true),
      tool === 'draw' && sticky,
    )}
    {@render button(
      'editor.tool_pan',
      'Pan',
      'hand',
      () => ontool('pan'),
      tool === 'pan',
    )}
  </div>
  <div class={GROUP} role="group" aria-label="Zoom">
    {@render button('editor.zoom_in', 'Zoom in', 'zoom-in', onzoomin)}
    {@render button('editor.zoom_out', 'Zoom out', 'zoom-out', onzoomout)}
    {@render button('editor.zoom_fit', 'Fit the picture', 'scan', onfit)}
    {@render button(
      'editor.zoom_actual',
      `Actual size, ${Math.round(zoomPercent)}%`,
      null,
      onactual,
    )}
  </div>
  <div class={GROUP} role="group" aria-label="History and labels">
    {@render button(
      'editor.undo',
      'Undo',
      'undo-2',
      onundo,
      undefined,
      !canUndo,
      hint('editor.undo', undoLabel ? `Undo: ${undoLabel}` : 'Undo'),
    )}
    {@render button(
      'editor.redo',
      'Redo',
      'redo-2',
      onredo,
      undefined,
      !canRedo,
      hint('editor.redo', redoLabel ? `Redo: ${redoLabel}` : 'Redo'),
    )}
    {@render button('editor.labels', 'Box labels', 'type', onlabels, labels)}
  </div>
</div>
