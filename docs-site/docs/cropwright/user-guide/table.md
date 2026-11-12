---
sidebar_position: 12
title: Table
---

# Table

The **Table** view lists detections or images as rows, with the columns you choose. Every
cell is text the backend already formatted, so what you read is what the server holds. Use
it to sort by anything and act on the rows.

<Screenshot name="cropwright/table-1600.png" alt="The Table view with two rows selected: thumbnail, class and confidence columns, sorted by confidence" caption="The Table view: the arrow marks the sort the server applied; two rows are selected" />

## Rows and columns

- **Detections** shows one row per item; **images** shows one row per image.
- **Columns** opens the picker. Your choice is saved in the page address (`columns=`), so a
  link or a reload keeps it. Without a choice you get the default set the server names.
- Numbers are right-aligned; a thumbnail column shows a small preview.

## Sorting

Click a column header to sort by it; click again for the other direction. The sort is done
by the server over the whole list, not over the rows on screen, and the arrow in the header
shows the order the server applied. Columns without an arrow cannot be sorted.

The table follows the same [item filter](./item-filter.md) as the other views. Rows load
as you scroll, by the server's cursor.

## Selecting and acting

- `Space` selects the focused row; `Shift` with the arrow keys extends the selection.
- The selection feeds the bulk bar, the same as in the grid.
- `Enter` opens the row in Review; `Shift+O` opens the image in the editor.

`GET /table` and `GET /table/columns` serve the table; an unknown column answers
`unknown_column`.
