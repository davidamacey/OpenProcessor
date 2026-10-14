/**
 * The one glyph reader for every `/review` file: every key the page prints
 * comes from the keymap (never a literal), so a rebinding shows everywhere.
 */

import { keymapStore } from '$stores/keymap.svelte';

export const kg = (actionId: string) => keymapStore.glyph(actionId);
