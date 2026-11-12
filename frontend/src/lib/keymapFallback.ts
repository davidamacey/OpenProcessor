/**
 * FALLBACK_KEYMAP — every keyboard action Cropwright binds, with today's
 * exact default keys and labels (docs/design/configurable-keyboard-
 * shortcuts-plan-2026-09-26.md §1.1/§2, step K1).
 *
 * This is DATA, in the shape the backend's `GET {prefix}/keymap` will
 * serve (plan §4.2). K1 uses it verbatim; K2 swaps in the served document
 * through `keymapStore.setDocument()` and keeps this only for a backend
 * that predates the route.
 *
 * Labels are the exact strings the shortcut overlay has always printed.
 * `{region}` is the region noun (the slot's `label.singular`); the served
 * document will render it server-side, the fallback substitutes it
 * client-side (`keymapStore.label(id, { region })`).
 *
 * Locked keys (plan §0 decision 3): Esc, Enter and the four arrows. An
 * action whose default includes one keeps it (`locked_keys`); no other
 * action may take one. The store enforces this on every document.
 *
 * The three per-box actions (`review.region.accept_box`/`reject_box`,
 * `box_edit.next_box`) are `available: true` — registered by the review
 * page's multi-box editing surface.
 */

export interface KeymapGrammar {
  /** Keys only a locked action may use, and never lose. */
  locked_keys: string[];
  max_combos_per_action: number;
}

export interface KeymapContext {
  id: string;
  label: string;
  description: string;
  /** Contexts active at the same time as this one (transitively). */
  includes: string[];
  /** Whether a per-class hotkey letter also fires in this context. */
  class_hotkeys_live: boolean;
}

export interface KeymapAction {
  id: string;
  context: string;
  group: string | null;
  label: string;
  description: string;
  default: string[];
  /** Effective combos, in `keyboard.svelte.ts`'s `normalize()` grammar. */
  keys: string[];
  modifiable: boolean;
  available: boolean;
  /** Default keys this action can never lose (subset of `grammar.locked_keys`). */
  locked_keys?: string[];
}

export interface KeymapValidationIssue {
  code: string;
  severity: 'error' | 'warning';
  field: string;
  message: string;
  detail?: Record<string, unknown>;
  bypassable?: boolean;
}

export interface KeymapDocument {
  grammar: KeymapGrammar;
  contexts: KeymapContext[];
  actions: KeymapAction[];
  /** Served only (K2): the derived class-hotkey reserved set. */
  reserved_hotkeys?: string[];
  /** Served only (K2, plan §0.1): per-project scope. */
  scope?: 'project' | 'deployment';
  project?: string;
  revision?: number;
  etag?: string;
  /** Whether the project has any saved override — the "custom keys" badge. */
  is_default?: boolean;
  updated_at?: string | null;
  /** The raw action-id -> combos override map that produced `actions[].keys`. */
  overrides?: Record<string, string[]>;
  /** `keymap_class_hotkey_shadowed` warnings and similar (plan §4.2). */
  issues?: KeymapValidationIssue[];
}

const LOCKED_KEYS = [
  'escape',
  'enter',
  'arrowleft',
  'arrowright',
  'arrowup',
  'arrowdown',
];

type ActionSeed = [
  id: string,
  context: string,
  group: string | null,
  label: string,
  keys: string[],
  extra?: { modifiable?: boolean; available?: boolean; description?: string },
];

const SEEDS: ActionSeed[] = [
  // `~` first so the printed glyph stays "~" (the overlay's "Always" row
  // and the ShortcutsButton title have always shown it).
  [
    'global.shortcuts_overlay',
    'global',
    null,
    'Toggle this panel',
    ['~', '`', 'shift+~'],
  ],
  [
    'global.close_overlay',
    'global',
    'cancel',
    'Close this panel / cancel',
    ['escape'],
    { modifiable: false },
  ],

  ['review.skip', 'review', 'skip', 'Skip', ['n']],
  ['review.undo', 'review', 'undo', 'Undo last', ['z']],

  [
    'review.queue.confirm',
    'review.queue',
    'confirm',
    'Confirm proposed & advance (or search classes if blank)',
    ['enter'],
  ],
  ['review.queue.discard', 'review.queue', 'discard', 'Discard', ['d']],
  ['review.queue.class_picker', 'review.queue', null, 'Search all classes…', ['/']],
  ['review.queue.prev', 'review.queue', 'prev', 'Previous item', ['arrowleft']],
  ['review.queue.next', 'review.queue', 'next', 'Next item', ['arrowright']],

  [
    'review.region.confirm',
    'review.region',
    'confirm',
    'Confirm {region} & advance',
    ['enter'],
  ],
  [
    'review.region.reject',
    'review.region',
    'reject',
    'Reject (no {region} visible)',
    ['d'],
  ],
  [
    'review.region.false_positive',
    'review.region',
    null,
    'False positive (keep box)',
    ['f'],
  ],
  ['review.region.edit_box', 'review.region', null, 'Edit {region} box', ['e']],
  [
    'review.region.back',
    'review.region',
    'prev',
    'Step back to last confirmed {region}',
    ['arrowleft', 'b'],
  ],
  ['review.region.next', 'review.region', 'next', 'Next item', ['arrowright']],
  [
    'review.region.accept_box',
    'review.region',
    null,
    'Accept selected {region} box',
    ['y'],
    { available: true },
  ],
  [
    'review.region.reject_box',
    'review.region',
    null,
    'Reject selected {region} box',
    ['r'],
    { available: true },
  ],

  ['box_edit.save', 'box_edit', 'confirm', 'Save {region} & exit edit', ['enter']],
  [
    'box_edit.cancel',
    'box_edit',
    'cancel',
    'Cancel edit',
    ['escape'],
    { modifiable: false },
  ],
  ['box_edit.nudge_up', 'box_edit', 'nudge', 'Move box up', ['arrowup']],
  ['box_edit.nudge_down', 'box_edit', 'nudge', 'Move box down', ['arrowdown']],
  ['box_edit.nudge_left', 'box_edit', 'nudge', 'Move box left', ['arrowleft']],
  ['box_edit.nudge_right', 'box_edit', 'nudge', 'Move box right', ['arrowright']],
  ['box_edit.shrink_right', 'box_edit', null, 'Move right edge left', ['[']],
  ['box_edit.grow_right', 'box_edit', null, 'Move right edge right', [']']],
  ['box_edit.delete_box', 'box_edit', null, 'Clear the box', ['backspace']],
  [
    'box_edit.next_box',
    'box_edit',
    null,
    'Select the next box',
    ['tab'],
    { available: true },
  ],

  ['workspace.view_prev', 'workspace', null, 'Previous view', ['[']],
  ['workspace.view_next', 'workspace', null, 'Next view', [']']],
  ['workspace.scope_prev', 'workspace', null, 'Previous cluster', [',']],
  ['workspace.scope_next', 'workspace', null, 'Next cluster', ['.']],
  ['workspace.toggle_details', 'workspace', null, 'Show or hide details', ['\\']],
  ['workspace.open_editor', 'workspace', null, 'Open in the image editor', ['shift+o']],
  ['workspace.class_picker', 'workspace.items', null, 'Search all classes…', ['/']],
  [
    'workspace.escape',
    'workspace',
    'cancel',
    'Back to the grid / clear selection / leave scope / close panel',
    ['escape'],
    { modifiable: false },
  ],
  [
    'workspace.select_toggle',
    'workspace.items',
    null,
    'Select or deselect the focused card or row',
    ['space'],
  ],
  [
    'workspace.open_review',
    'workspace.items',
    null,
    'Open the focused card or row in Review',
    ['enter'],
    { modifiable: false },
  ],

  ['table.prev_row', 'table', null, 'Previous row', ['arrowup'], { modifiable: false }],
  ['table.next_row', 'table', null, 'Next row', ['arrowdown'], { modifiable: false }],
  ['table.extend_prev', 'table', null, 'Extend selection up', ['shift+arrowup']],
  ['table.extend_next', 'table', null, 'Extend selection down', ['shift+arrowdown']],

  [
    'cluster.confirm',
    'cluster',
    'confirm',
    'Confirm selected & advance',
    ['shift+enter'],
  ],
  [
    'cluster.accept_all_vlm',
    'cluster',
    null,
    'Confirm all VLM suggestions on page',
    ['shift+g'],
  ],
  ['cluster.accept_vlm', 'cluster', null, 'Accept VLM suggestion for selected', ['g']],
  ['cluster.skip', 'cluster', 'skip', 'Skip selected', ['n']],
  [
    'cluster.flag_new_class',
    'cluster',
    null,
    'Flag selected as needing new class (curator review)',
    ['shift+n'],
  ],
  ['cluster.discard', 'cluster', 'discard', 'Discard selected', ['d']],
  ['cluster.undo', 'cluster', 'undo', 'Undo last action', ['z', 'ctrl+z', 'meta+z']],
  [
    'cluster.ignore',
    'cluster',
    'ignore',
    'Ignore selected (exclude from training)',
    ['x'],
  ],
  ['cluster.unignore', 'cluster', null, 'Undo last ignore', ['u']],
  [
    'cluster.select_all',
    'cluster',
    'select_all',
    'Select all on page',
    ['a', 'ctrl+a', 'meta+a'],
  ],
  ['cluster.prev', 'cluster', 'prev', 'Previous crop', ['arrowleft']],
  ['cluster.next', 'cluster', 'next', 'Next crop', ['arrowright']],
  ['cluster.move', 'cluster', null, 'Move selected to cluster…', ['m']],

  [
    'clusters_search.select_all',
    'clusters_search',
    'select_all',
    'Select all results',
    ['a'],
  ],
  [
    'clusters_search.cancel',
    'clusters_search',
    'cancel',
    'Clear selection',
    ['escape'],
    { modifiable: false },
  ],
  [
    'clusters_search.ignore',
    'clusters_search',
    'ignore',
    'Ignore selected (exclude from training)',
    ['x'],
  ],
  ['clusters_search.undo', 'clusters_search', 'undo', 'Undo last action', ['z']],

  ['region_gallery.undo', 'region_gallery', 'undo', 'Undo last {region} action', ['z']],
];

function seedToAction([
  id,
  context,
  group,
  label,
  keys,
  extra,
]: ActionSeed): KeymapAction {
  const locked = keys.filter((k) => LOCKED_KEYS.includes(k));
  return {
    id,
    context,
    group,
    label,
    description: extra?.description ?? '',
    default: [...keys],
    keys: [...keys],
    modifiable: extra?.modifiable ?? true,
    available: extra?.available ?? true,
    ...(locked.length > 0 ? { locked_keys: locked } : {}),
  };
}

export const FALLBACK_KEYMAP: KeymapDocument = {
  grammar: { locked_keys: LOCKED_KEYS, max_combos_per_action: 3 },
  contexts: [
    {
      id: 'global',
      label: 'Everywhere',
      description: 'Active on every page.',
      includes: [],
      class_hotkeys_live: true,
    },
    {
      id: 'review',
      label: 'Review',
      description: 'Scan mode in the Review view, with the Workspace keys.',
      includes: ['global', 'workspace'],
      class_hotkeys_live: true,
    },
    {
      id: 'review.queue',
      label: 'Review queue',
      description: 'A plain Review queue.',
      includes: ['review'],
      class_hotkeys_live: true,
    },
    {
      id: 'review.region',
      label: 'Region review',
      description: 'The region queue, scan mode.',
      includes: ['review'],
      class_hotkeys_live: true,
    },
    {
      id: 'box_edit',
      label: 'Box editing',
      description: 'Region edit mode and the box editor.',
      includes: ['global'],
      class_hotkeys_live: false,
    },
    {
      id: 'workspace',
      label: 'Workspace',
      description: 'Keys that work in every Workspace view.',
      includes: ['global'],
      class_hotkeys_live: true,
    },
    {
      id: 'workspace.items',
      label: 'Workspace items',
      description:
        'Keys of the views that list items to open and select: Enter opens Review, Space selects, / assigns a class.',
      includes: ['workspace'],
      class_hotkeys_live: true,
    },
    {
      id: 'table',
      label: 'Workspace table',
      description: 'The Workspace Table view: row focus and selection.',
      includes: ['workspace.items'],
      class_hotkeys_live: true,
    },
    {
      id: 'cluster',
      label: 'Workspace grid',
      description:
        "The Workspace card grids: clusters, detections and a cluster's crops.",
      includes: ['workspace.items'],
      class_hotkeys_live: true,
    },
    {
      id: 'map',
      label: 'Map',
      description:
        'The Workspace Map view; no class letters, so a key cannot label an area.',
      includes: ['workspace'],
      class_hotkeys_live: false,
    },
    {
      id: 'clusters_search',
      label: 'Cluster search',
      description: 'Semantic-search results in the Workspace Clusters view.',
      includes: ['global'],
      class_hotkeys_live: true,
    },
    {
      id: 'region_gallery',
      label: 'Region gallery',
      description: 'The region gallery in the Workspace Clusters view.',
      includes: ['global'],
      class_hotkeys_live: false,
    },
  ],
  actions: SEEDS.map(seedToAction),
};
