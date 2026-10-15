/**
 * Read-only views over the served keymap document for the "Keyboard
 * shortcuts" card: which actions a context table shows, the verb groups and
 * their members, and display labels.
 */

import { keymapStore } from '$stores/keymap.svelte';
import type { KeymapAction } from '$lib/keymapFallback';

function actionsFor(contextId: string): KeymapAction[] {
  return keymapStore.document.actions.filter((a) => a.context === contextId);
}

/**
 * Actions still rendered inline in a context's own table: ungrouped
 * actions, plus every locked/non-modifiable action regardless of
 * group (a grouped-and-modifiable action moves to the Verb groups
 * section below instead).
 */
export function contextRowActions(contextId: string): KeymapAction[] {
  return actionsFor(contextId).filter((a) => a.group === null || !a.modifiable);
}

export function contextLabel(contextId: string): string {
  return (
    keymapStore.document.contexts.find((c) => c.id === contextId)?.label ?? contextId
  );
}

/** Group ids shared by 2+... actually any modifiable actions, in first-seen order. */
export function namedGroups(): string[] {
  const out: string[] = [];
  for (const a of keymapStore.document.actions) {
    if (a.group && a.modifiable && !out.includes(a.group)) out.push(a.group);
  }
  return out;
}

export function groupMembers(groupId: string): KeymapAction[] {
  return keymapStore.document.actions.filter((a) => a.group === groupId && a.modifiable);
}

export function groupLabel(groupId: string): string {
  const words = groupId.replace(/_/g, ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
}
