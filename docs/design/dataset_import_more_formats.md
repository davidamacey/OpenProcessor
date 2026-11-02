# Dataset import: CVAT, Label Studio, KITTI, CreateML, LabelMe, YOLO names files (#370, part 2)

Status: implemented. Follows `dataset_import_voc_openimages.md`: every format is
a reader that emits the same format-neutral `DatasetScan`; the class mapping,
regions, lock rule, undo and report code are unchanged.

Out of scope: Label Studio `JSON_MIN`, CVAT video (`<track>`) exports, rotated
boxes, polygons, masks and keypoints as boxes, COCO segmentation-only files.

## Principles (unchanged)

- **Class identity is the name.** The label text is `dataset_class`;
  `DatasetScan.class_ids` stays empty (YOLO keeps its index map for reporting
  only). A YOLO dataset without a names source is refused.
- Every referenced image resolves through `paths.resolve_ref` and the path
  guard; a remote URL is never fetched.
- Every file is size-capped before it is read; XML refuses every DTD.
- Malformed input becomes an issue code, never an exception out of the scan.

## Formats

| Format id | Detection (after VOC and Open Images) | Box source | No-object image |
|---|---|---|---|
| `cvat` | `annotations.xml` whose head opens `<annotations` | `<box>` pixels, 0-based edges | negative |
| `labelstudio` | a root `*.json`, head is a list with `"data":` and `"annotations":` | `rectanglelabels` percent | negative if one annotation has no box; unlabeled with none |
| `createml` | a `*.json` (root, or `train`/`valid`/`test` child), head is a list with `"image":` and `"annotations":` | centre `x,y,width,height` pixels | negative |
| `labelme` | a `*.json` (same places), head is an object with `"shapes":` | `rectangle` two corners, pixels | negative |
| `kitti` | `label_2/*.txt` at the root or under `training/` | fields 5-8 pixels | negative (empty file) |
| `yolo` (names file) | `images/` + `classes.txt`/`obj.names`/`names.txt` | normalized rows | negative (empty label file) |

Order of the detection table: OpenProcessor export, YOLO `data.yaml`, COCO,
VOC, Open Images, CVAT, Label Studio, CreateML, LabelMe, KITTI, YOLO `images/`,
COCO. A `data.yaml` still wins everywhere. Label Studio is tested before
CreateML because both are JSON lists with an `annotations` key.

## Skipped shapes

Only axis-aligned boxes map. Rotated boxes (CVAT `rotation`, Label Studio
`rotation`), polygons, polylines, points, masks, ellipses, cuboids, LabelMe
non-rectangle shapes and KITTI `DontCare` are skipped with the info issue
`shape_not_box_skipped` and count as present, so an image whose only shapes
were skipped is `unlabeled`, not `negative`.

## Issue codes

| Code | Severity | Blocking |
|---|---|---|
| `cvat_xml_invalid` | error | yes |
| `labelstudio_json_invalid` | error | yes |
| `labelstudio_task_skipped` | warning | no |
| `createml_json_invalid` | error | yes |
| `labelme_json_invalid` | warning | no |
| `yolo_names_file_invalid` | error | yes |
| `shape_not_box_skipped` | info | no |

## Choices

- Label Studio with several non-cancelled annotations on a task is refused for
  that task rather than guessing which annotator is right.
- LabelMe `imagePath` often holds the annotator's relative path (`..\images\a.jpg`);
  when it resolves to nothing inside the dataset the JSON's own stem is tried
  next to the JSON and in `images/` there.
- Pixel formats decode every image header for the size (CVAT, LabelMe and Label
  Studio carry a declared size, which is checked for the first
  `PREVIEW_SIZE_CHECKS` images and by the job after that).
- JSON auto-detection reads only the first 64 KiB; `source.format` parses the
  whole file.
