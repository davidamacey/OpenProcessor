/**
 * What the VLM menu's Assist covers: the selection when there is one, else
 * everything the filter, queue and scope match. A crop selection goes as
 * crops; any other selection goes as the filter that `ItemSelection` carries.
 */
import type { AssistOpener } from '$lib/assistRun/assistRunController.svelte';
import type { ItemFilter, ItemSelection } from '$lib/types_itemFilter';
import type { WorkspaceSelection } from './selection';

export function vlmAssistOpener(
  selection: WorkspaceSelection,
  itemSelection: ItemSelection | null,
  scopeFilter: ItemFilter,
): AssistOpener {
  const whole = {
    kind: 'filter' as const,
    filter: structuredClone($state.snapshot(scopeFilter)),
  };
  if (selection.kind === 'items') {
    return {
      title: 'the selection',
      selected: { kind: 'crops', crop_ids: [...selection.cropIds] },
      whole,
    };
  }
  if (itemSelection?.filter) {
    return {
      title: 'the selection',
      selected: {
        kind: 'filter',
        filter: structuredClone($state.snapshot(itemSelection.filter)),
      },
      whole,
    };
  }
  return { title: 'the scope', whole };
}
