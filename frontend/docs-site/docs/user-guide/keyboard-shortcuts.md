---
sidebar_position: 11
title: Keyboard shortcuts
---

# Keyboard shortcuts

Class assignment is per-class (`hotkey_letter`, set on `/p/<project>/classes`
or in the `~` overlay). Reserved single-character action keys can never be
bound to a class — the reserved set is served by the backend
(`GET {prefix}/classes`'s `reserved_hotkeys`) unioned with every registered
slot's own keymap.

Every other shortcut is a named **action** (e.g. "confirm", "discard",
"undo") bound to a key, not a hardcoded key with no name — this is what
makes the shortcuts below customizable per project rather than fixed.

## Global

| Key | Action |
| --- | --- |
| `` ` `` / `~` | Toggle the keyboard shortcut overlay |
| `Esc` | Close the overlay |
| _class letter_ | Assign that class (selection / current item) |

## `/p/<project>/clusters/[id]`

| Key | Action |
| --- | --- |
| `Enter` | Confirm selected to the chosen class + advance |
| `Shift+Enter` | Accept all VLM suggestions on the page |
| `G` | Accept the VLM suggestion for selected |
| `N` | Skip + advance |
| `Shift+N` | Flag selected as needing a new class |
| `D` | Discard selected |
| `Z` | Undo last action (bulk label, move, discard, VLM-accept-all, dismiss) |
| `X` | Ignore selected (exclude from training + clustering) |
| `U` | Undo last ignore |
| `A` | Select all on page |
| `←` / `→` | Move the selection by one crop |
| `M` | Move selected to another cluster |
| `Esc` | Clear drag capture / close picker / clear selection |

## `/p/<project>/review`

| Key | Action |
| --- | --- |
| `Enter` | Confirm proposed + advance (opens the class picker when there's no proposal) |
| `/` | Open the fuzzy-search class picker |
| `D` | Discard — dismiss from every review queue (reversible, not via `Z`) |
| `N` | Skip |
| `Z` | Undo last action |
| `←` / `→` | Previous / next item |

### Region tab additions

| Key | Action |
| --- | --- |
| `F` | Mark false positive (box kept) |
| `E` | Enter box edit mode |
| `Y` / `R` | Accept / reject the selected box (saves immediately) |
| `Enter` | Confirm every box still proposed (see [Multi-box regions](./multi-box-regions.md)) |
| `Tab` (edit mode) | Select the next box |
| `Enter` / `Esc` (edit mode) | Save boxes / cancel edit |
| Arrow keys, `[` `]`, `Backspace` (edit mode) | Nudge the selected box, move its right edge, clear it |

`Esc` cannot cancel an in-progress pointer drag (only keyboard/aria drags) —
it clears the captured multi-drag set and restores the grid layout instead.

## Customizing shortcuts

The tables above are the defaults every deployment starts with. When the
backend serves a keymap, `/p/<project>/settings` gets a **Keyboard
shortcuts** card that lets you rebind them for that project — it's
per-project, so two projects on the same deployment can bind the same
action to different keys. On a backend that doesn't serve a keymap yet, the
card doesn't appear at all and every shortcut runs on its built-in default.

- **Verb groups.** Actions that mean the same thing across several pages —
  undo, confirm, discard, skip, previous/next, select all, ignore, nudge —
  are edited together as one group by default: rebinding "undo" applies
  everywhere undo appears. A "Customize per page" section under each group
  lets you give one specific page its own key for that verb instead,
  without touching the others.
- **Up to three keys per action.** Each action can carry more than one key
  combo (so an old habit and a new one can both work at once), capped at
  three per action.
- **Locked keys.** `Esc`, `Enter`, and the four arrow keys keep their
  built-in meaning and can't be reassigned away from it — an action that
  already uses one of them keeps it, and no other action can take it.
  You can still add extra key combos to actions that don't already use a
  locked key.
- **Class-hotkey clashes.** If a key you're binding to an action is already
  used as a class hotkey, the save is refused with an offer to unbind the
  class key first and save again — it never silently overrides one binding
  with the other.
- **A project's own keys.** Because the keymap is per project, the project
  switcher shows a "custom keys" badge on a project that has been rebound.
  Copying settings between projects can carry a keymap; any clash with the
  destination's class hotkeys is reported afterwards.
- **Server messages are shown as served.** While you edit, the backend checks
  the draft and its own errors and warnings appear verbatim. A save can be
  refused because someone else changed the keymap (reload and reapply), or
  because a key is a class hotkey (offered: unbind those class keys and save
  again).
- **Reset.** Every action (or the whole keymap) can be reset back to its
  default binding.

<Screenshot name="settings-keymap-1600.png" alt="Cropwright keyboard shortcuts editor" caption="Settings — the keyboard shortcuts editor, verb groups and per-page overrides" />
