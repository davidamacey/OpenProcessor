/**
 * Canonical legacy_sorter group → hotkey-letter mapping.
 *
 * Mirrors the desktop sorter app's `default_required_hotkeys()` in
 * src-tauri/src/core/hotkeys.rs so labelers don't need to learn two
 * keyboard layouts. Source of truth for both repos lives upstream in
 * legacy_sorter; this is the labeler's local copy — keep in sync if
 * legacy_sorter ever bumps the defaults.
 *
 * Excludes 'd' (delete in sorter; collides with the labeler's 'd' for
 * Discard which is identical semantics) and 'f'/'h' (sorter-only
 * destinations: friends + highlights folders, not class assignments).
 */

export interface SorterGroupHotkey {
  /** Single-character keyboard binding. Lowercase. */
  key: string;
  /** Human-readable label. */
  label: string;
  /** Registry `group` value to filter / assign by. */
  group: string;
}

/**
 * Group hotkeys that map cleanly to legacy class registry groups.
 * The labeler uses these for coarse navigation (filter the cluster view
 * to a single group, or jump to the group's dominant class). Class-level
 * assignment continues to use number hotkeys 1-9, 0.
 */
export const SORTER_GROUP_HOTKEYS: readonly SorterGroupHotkey[] = [
  { key: 'c', label: 'cars', group: 'cars' },
  { key: 'r', label: 'class_a', group: 'class_a' },
  { key: 's', label: 'class_bs', group: 'class_bs' },
  { key: 'x', label: 'sportycars', group: 'sportycars' },
  { key: 't', label: 'touring-adv', group: 'class_g' },
  {
    key: 'o',
    label: 'oddshots',
    group: 'oddshots-commercial-atvs-utvs-rvs-leos-ems',
  },
  {
    key: ' ',
    label: 'trikes-class_fs',
    group: 'trikes-class_fs-motards-scooters-bicycles',
  },
  { key: 'e', label: 'exotics', group: 'exotics' },
] as const;

/** Lookup: registry group → hotkey letter. */
export const GROUP_TO_KEY: Readonly<Record<string, string>> = Object.fromEntries(
  SORTER_GROUP_HOTKEYS.map((g) => [g.group, g.key]),
);

/** Lookup: hotkey letter → registry group. */
export const KEY_TO_GROUP: Readonly<Record<string, string>> = Object.fromEntries(
  SORTER_GROUP_HOTKEYS.map((g) => [g.key, g.group]),
);

/** Display key — turn ' ' into 'space' for the legend UI. */
export function displayKey(key: string): string {
  return key === ' ' ? 'space' : key;
}
