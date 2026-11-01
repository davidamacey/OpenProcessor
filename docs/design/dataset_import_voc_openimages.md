# Dataset import: Pascal VOC and Open Images (#370, part 1)

Status: design for the first two native formats of #370. The dataset-import
service already reads YOLO, COCO and OpenProcessor exports into one
format-neutral `DatasetScan`; this adds readers for **Pascal VOC XML** and
**Open Images CSV** (boxes CSV plus class descriptions, MID to name). The
class mapping, regions, lock rule, undo and report code are unchanged.

Out of scope: CVAT, Label Studio, YOLO variants, KITTI, CreateML, Open Images
image-level labels, and segmentation masks.

## Principles

- **Class identity is the name.** The VOC `<name>` text and the Open Images
  `DisplayName` are the `dataset_class`. An Open Images MID is a lookup key
  into the descriptions file and never leaves the reader. Neither format has
  an integer class index, so `DatasetScan.class_ids` stays empty.
- Every path a dataset names resolves through `paths.resolve_ref` and the
  path guard, as in the COCO and YOLO readers.
- Every text file is size-capped before it is read.
- Malformed input becomes an issue code, never an exception out of the scan.

## Wire

The source format ids are `voc` and `openimages`. Served labels are
"Pascal VOC (XML per image)" and "Open Images (boxes CSV + class
descriptions)". The `DatasetSource.format` and issue-code enums in the
generated OpenAPI contract grow accordingly; there are no new request
fields.

## Detection

`detect_format` becomes an ordered rule table; the first match wins.

| # | Rule | Result |
|---|---|---|
| 1 | OpenProcessor export marker | `openprocessor_export` |
| 2 | the root is a `.yaml`/`.yml` file | `yolo` |
| 3 | `data.yaml`, `data.yml` or `dataset.yaml` in the root | `yolo` |
| 4 | `annotations/*.json` and no `labels/` directory | `coco` |
| 5 | `Annotations/` (or `annotations/`) holds `*.xml` and no `*.json` | `voc` |
| 6 | `*annotations-bbox*.csv` and `*class-descriptions*.csv` in the root | `openimages` |
| 7 | an `images/` directory | `yolo` |
| 8 | a child directory has `images/` | `yolo` |
| 9 | `annotations/*.json` | `coco` |

Ambiguities: a `data.yaml` always wins, so a YOLO dataset that also carries
VOC or Open Images files is YOLO. An `annotations/` directory with both JSON
and XML is COCO. VOC and Open Images markers in one root resolve to VOC.
`images/` plus `Annotations/*.xml` used to resolve to YOLO and then fail for
lack of a `data.yaml`; it is now VOC, so no dataset that imported before
changes format. The user can always force a format with `source.format`.

## Issue codes

| Code | Severity | Blocking |
|---|---|---|
| `voc_xml_invalid` | warning | no |
| `voc_size_missing` | info | no |
| `voc_difficult_skipped` | info | no |
| `openimages_csv_invalid` | error | yes |
| `openimages_class_unknown` | warning | no |
| `openimages_class_duplicate_name` | error | yes |
| `openimages_group_skipped` | info | no |
| `openimages_image_split_conflict` | warning | no |
| `openimages_image_unreferenced` | info | no |
| `image_file_missing` | warning | no |

The label of `image_size_mismatch` becomes format-neutral ("Declared image
size does not match the decoded image"); the code is unchanged.

## Pascal VOC

Layout (the root is the `VOC2012`-style directory, not `VOCdevkit`):
`Annotations/*.xml`, images in `JPEGImages/` (else `images/`, else the
root), optional `ImageSets/Main/{train,val,test}.txt` for splits.

- XML is parsed by a stdlib parser after every DOCTYPE and ENTITY
  declaration is refused, so no entity can be declared or expanded (no XXE,
  no entity expansion). `defusedxml` is not a dependency. A file over 1 MiB is
  refused unread.
- The image is `<filename>` resolved in the images directory, falling back to
  the XML stem plus an image extension. `<folder>` and `<path>` are ignored:
  `<path>` is an absolute path on the annotator's machine.
- Size comes from `<size>`; it is checked against the decoded header for the
  first images and carried as `declared_size` so the job checks the rest. A
  missing or unusable size is `voc_size_missing` and the decoded size is used.
- Objects are direct `<object>` children only (`<part>` sub-boxes are never
  boxes). Boxes are 1-based inclusive pixels, so `(xmin - 1, ymin - 1, xmax,
  ymax)` is divided by the size. A one pixel tolerance absorbs 0-based
  writers without an issue.
- `difficult=1` objects are skipped (the devkit evaluation convention) and
  count as present. `truncated`, `occluded` and `pose` are ignored.
- An annotation with no object is `negative`; with objects but no kept box,
  `unlabeled`; otherwise `labeled`.
- A split comes from the first of test, val, train whose list names the
  stem. An image without an XML file is not part of the dataset.

## Open Images

Layout: `class-descriptions-boxable.csv` and one or more
`*annotations-bbox*.csv` files directly in the root; images in
`images/<split>/`, `<split>/`, `images/` or the root, named `<ImageID>.<ext>`.

- Descriptions are `MID,DisplayName`, with or without the v7 header row. A
  MID listed twice with different names, or several descriptions files,
  block the import.
- Box columns are ordered `XMin, XMax, YMin, YMax` and are already
  normalized. Extra columns are optional.
- `IsGroupOf=1` boxes are skipped (like COCO crowd) and count as present.
  `IsDepiction` boxes are kept; occluded, truncated and inside flags are
  ignored.
- The split comes from the boxes file name; the first file that names an
  ImageID owns it, and a row from another split is
  `openimages_image_split_conflict`.
- An ImageID must match `[A-Za-z0-9_-]{1,128}`; it is a dictionary key and is
  never joined into a path.
- An Open Images image is never `negative`: absent boxes do not mean a
  verified empty image. With no kept box it is `unlabeled`.
- Two referenced MIDs with one display name block the import; collisions
  among unused MIDs do not. Images with no box row are reported and not
  imported.
- A boxes file over 512 MiB is refused (the full train file is). The
  distinct-image count is bounded by `OP_DATASET_PREVIEW_MAX_FILES`.
