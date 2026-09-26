---
sidebar_position: 11
title: Keyboard shortcuts
---

# Keyboard shortcuts

Class assignment is per-class (`hotkey_letter`, set on `/classes` or in the
`~` overlay). Reserved single-character action keys can never be bound to a
class — the reserved set is served by the backend (`GET {API_PREFIX}/classes`'s
`reserved_hotkeys`) unioned with every registered slot's own keymap.

## Global

| Key | Action |
| --- | --- |
| `` ` `` / `~` | Toggle the keyboard shortcut overlay |
| `Esc` | Close the overlay |
| _class letter_ | Assign that class (selection / current item) |

## `/clusters/[id]`

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

## `/review`

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
| `E` | Enter bbox edit mode |
| `Enter` / `Esc` (edit mode) | Save bbox / cancel edit |

`Esc` cannot cancel an in-progress pointer drag (only keyboard/aria drags) —
it clears the captured multi-drag set and restores the grid layout instead.
