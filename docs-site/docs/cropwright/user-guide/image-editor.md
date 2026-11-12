---
sidebar_position: 4
title: Image editor
---

# Image editor (`/p/<project>/images/<image_id>`)

The editor shows one picture with all its objects and lets you fix it fast: draw, move,
resize and delete boxes, change or create a class without leaving the page, and undo and
redo. Open it from the Workspace with **Open in the image editor** (on a detection's details)
or the `Shift+O` key on a focused image or detection.

The editor keeps the Workspace's filter, queue, scope and sort in its URL. The strip under
the picture shows the image before this one and the next few, in the order the Workspace
showed them; `,` and `.` walk it, and `Esc` (with nothing selected) goes back to the same
Workspace view.

## Tools

| Tool | Key | What it does |
| --- | --- | --- |
| Select and move | `V` | Click a box to select it; drag it to move, drag a handle to resize |
| Draw one box | `B` | Drag, or click two corners; then back to Select |
| Draw boxes | `Shift+B` | Like Draw, and stays on |
| Pan | `H` | Drag the zoomed picture. Holding `Space` pans for a moment |

After you draw a box the **class picker** opens with the class you used last on its first
row, so `Enter` saves the box with it. Type to search the registry, or type a new name and
choose **Create**; if the server refuses the name it offers a valid one.

The wheel zooms at the cursor. `=` and `-` zoom, `0` fits the picture, `1` shows it at
100 % (a sharper copy loads when it is drawn larger than the one on screen).

## Objects

The list on the right shows each object with the label the server decided. **Locked** marks
a box a person set: a re-detect never moves it, and the editor does not move it either.
**Deleted** objects stay listed (struck through) until you reload. `Tab` and `Shift+Tab`
walk the live objects. A class hotkey sets the selected object's class. `/` opens the
picker for it.

On a narrow screen the list is a drawer under the picture.

## Saving and undo

Every finished gesture is saved at once; nothing is sent while you drag. The top right shows
**Saved**, **Saving…** or **Changed by someone else, reloaded**; a failed save also shows the
server's own message. If someone else changed a box first, the editor reloads and shows their
box.

`Ctrl+Z` undoes the last change and `Ctrl+Shift+Z` redoes it, for the current image in this
session (Cmd on a Mac). Delete is a soft delete: undo brings the box back. Moving a box so
that region boxes fall outside it asks before deleting those too.

A **re-detect** of the image never changes a box you drew, moved or deleted.

`Shift+R` edits the selected object's region boxes in the region editor when the
deployment has a region profile; that edit can be undone but not redone.

## Keys

Every key above is a keymap action in the `editor` context and can be changed in
`/p/<project>/settings`; see [Keyboard shortcuts](./keyboard-shortcuts.md). A class hotkey
cannot use a key the editor binds.
