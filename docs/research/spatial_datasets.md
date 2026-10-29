# Public multi-view datasets for the spatial-memory work

Status: research snapshot (2026-10-29), tracked in #348. The datasets back the plan in
[OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md](OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md):
discover same-place photo groups, verify overlap geometrically, reconstruct, and place photos from
different years in one scene. Everything here is public. Images are not committed; the fetcher
[`scripts/datasets/fetch_spatial_sets.py`](../../scripts/datasets/fetch_spatial_sets.py) rebuilds the set.

Default root `data/spatial/<dataset>/`, not committed (158 GB, see the summary for per-set sizes). Licences differ and several are
non-commercial research only. They are fine for evaluating here; do not redistribute the data or
publish derived images from the non-commercial sets without checking the terms.

## Summary

| Dataset | On disk | Images | Pose / geometry ground truth | Main use in the plan |
|---|---|---|---|---|
| 7-Scenes | 7 scenes, 22 GB | 43,000 RGB-D frames | per-frame camera pose (Kinect fusion) | indoor pose-recovery sanity check |
| Cambridge Landmarks | 5 of 6 scenes, 25 GB | 7,295 | per-image pose, SfM model (`.nvm`) | outdoor landmark relocalisation, same-place groups |
| Aachen Day-Night v1.1 | 17 GB | 7,890 | SfM model, query poses via server only | day vs night matching |
| RobotCar Seasons v2 | 6 GB (images) | 32,796 | train poses in `robotcar_v2_train.txt` | seasons and weather, temporal drift |
| IMC PhotoTourism | train 4 + test 7 scenes, 20 GB | 5,816 | COLMAP sparse model and depth maps per scene | unordered tourist photos, closest to a real library |
| Google Landmarks v2 subset | 6 shards, 7 GB | 49,596 | landmark id only | place-retrieval distractors and positives |
| Tanks and Temples | training 7, intermediate 8, 11 GB | 5,440 | laser scan point cloud (training) | reconstruction quality, no pose files |
| ETH3D multi-view | 13 DSLR + 5 rig scenes, 22 GB | 454 DSLR, 4,796 rig | COLMAP poses + laser scan | reconstruction accuracy |
| Mip-NeRF 360 | 9 scenes, 32 GB | 1,940 | COLMAP poses (`poses_bounds.npy`, `sparse/`) | splat and NeRF training/rendering |
| COCO val2017 | 1 GB | 5,000 | none (object boxes only) | non-spatial negatives, shared with the detection benchmarks |

## Per dataset

### 7-Scenes
- Source: <https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/>; archives at
  `download.microsoft.com/.../28564B23-0828-408F-8631-23B1EFF1DAC8/<scene>.zip`.
- Licence: Microsoft Research non-commercial research licence (see the project page).
- Size and count: chess, fire, heads, office, pumpkin, redkitchen, stairs; 640x480 RGB plus depth,
  500 to 1000 frames per sequence, 26,000 train and 17,000 test frames in all.
- Ground truth: camera-to-world pose per frame, plus the dense fused model the poses came from.
- Use: small controlled indoor scenes to check that pose recovery and overlap scoring do not fail
  on textureless surfaces (stairs, heads). Too small and too uniform for discovery tests.

### Cambridge Landmarks
- Source: University of Cambridge repository, handles `1810/251342` (KingsCollege), `251340`
  (OldHospital), `251336` (ShopFacade), `251294` (StMarysChurch), `251291` (GreatCourt); project page
  <http://mi.eng.cam.ac.uk/projects/relocalisation/>. Street (`251292`) was not fetched.
- Licence: none shipped in the archives; the READMEs ask for citation of Kendall et al., ICCV 2015.
  Treat as research use and confirm on the repository record before redistributing.
- Size and count: 7,295 PNG frames across the five scenes (for example King's College 1,220 train and
  343 test), plus videos and an SfM reconstruction per scene.
- Ground truth: 6-DoF pose per image (`dataset_train.txt`, `dataset_test.txt`) and `reconstruction.nvm`.
- Use: outdoor same-landmark grouping with known poses, so recovered relative poses can be scored.

### Aachen Day-Night v1.1
- Source: <https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/>
  (`aachen_v1_1.zip`, `images/database_and_query_images.zip`), described at <https://www.visuallocalization.net/>.
- Licence: CC BY-NC-SA 4.0, non-commercial academic use (README).
- Size and count: extracted copy has 4,479 database, 1,040 query (day and night) and 2,371 sequence
  frames. Faces and plates were blurred by the authors.
- Ground truth: COLMAP and NVM reference model for the database images. Query poses are withheld and
  scored by the evaluation service.
- Use: day against night matching for the geometric verifier; a failure-mode set for the
  appearance-change gate.

### RobotCar Seasons v2
- Source: <https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/>.
- Licence: CC BY-NC-SA 4.0, non-commercial academic use.
- Size and count: 20,862 overcast-reference images plus 11,934 across dawn, dusk, night, night-rain,
  overcast-summer, overcast-winter, rain, snow and sun (left, rear and right cameras).
- Ground truth: `robotcar_v2_train.txt` holds poses for the reference images and about half of the
  other conditions; `robotcar_v2_test.txt` lists images without poses. Intrinsics and extrinsics are
  included. The reference SfM models (`3D-models/all-merged`, `overcast-reference.db`) were not fetched;
  only `individual/queries_per_location.zip` is present.
- Use: the closest public analogue of "the same street across seasons"; tests the temporal axis.

### IMC PhotoTourism (Image Matching Challenge 2020)
- Source: `https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/<scene>.tar.gz` and
  `.../TestData/<scene>.tar.gz`; challenge page <https://www.cs.ubc.ca/research/image-matching-challenge/>.
- Licence: crowd-sourced Flickr photos under their individual licences, released for research use.
  No single licence applies; do not redistribute.
- Size and count: 5,816 images. Test scenes: british_museum 660, florence_cathedral_side 108,
  milan_cathedral 124, mount_rushmore 138, piazza_san_marco 249, sagrada_familia 401,
  st_pauls_cathedral 615 (2,295). Train scenes: brandenburg_gate 1,363, reichstag 75, sacre_coeur 1,179,
  temple_nara_japan 904 (3,521).
- Ground truth: every tarball carries a COLMAP sparse model (`dense/sparse`) and per-image depth maps.
- Use: unordered, unlabelled, mixed-camera tourist photos with years of spread, which is the shape of
  a personal library. The main benchmark for scene discovery and verification.

### Google Landmarks v2 (subset)
- Source: `https://s3.amazonaws.com/google-landmark/` (`train/images_00[0-5].tar`,
  `metadata/train_clean.csv`, `md5sum/train/*.txt`); <https://github.com/cvdfoundation/google-landmark>.
- Licence: images are CC BY or public domain; `metadata/train_attribution.csv` lists the author and
  licence for every image. Annotations are CC BY 4.0.
- Size and count: 6 of 500 shards, 8,266 images each, 49,596 in all. The attribution CSV (1 GB) was
  also fetched.
- Ground truth: landmark id only, no poses.
- Use: large pool of place-retrieval positives and hard distractors for the SALAD/CLIP candidate stage.

### Tanks and Temples
- Source: <https://www.tanksandtemples.org/download/>. The archives sit behind a manual form, so the
  fetcher does not cover them.
- Licence: research use under the site terms (no licence file shipped); confirm before redistributing.
- Size and count: all 7 training scenes (Barn 410, Caterpillar 383, Church 507, Courthouse 1,106,
  Ignatius 263, Meetingroom 371, Truck 251 images) and all 8 intermediate scenes (2,149 images).
- Ground truth: training scenes have a laser-scan point cloud. The local `*_gt.zip` files are
  binary PLY files with a misleading extension, not zip archives. Intermediate scenes have no public
  ground truth. Camera pose files from the site were not downloaded, so poses must be estimated.
- Use: end-to-end reconstruction quality (completeness and accuracy against the scan) for the
  engine comparison.

### ETH3D multi-view
- Source: <https://www.eth3d.net/datasets>, `https://www.eth3d.net/data/multi_view_training_*.7z`.
- Licence: stated on the site; verify before redistribution (not shipped in the archives).
- Size and count: 13 DSLR scenes (454 undistorted images) and 5 rig scenes (4,796 images), with
  scan-evaluation archives for both.
- Ground truth: COLMAP-format `cameras.txt`/`images.txt`/`points3D.txt` and a laser scan with
  occlusion masks.
- Use: scoring recovered poses and meshes with a trusted metric; includes low-texture scenes.

### Mip-NeRF 360
- Source: `https://storage.googleapis.com/gresearch/refraw360/360_v2.zip` and `360_extra_scenes.zip`;
  <https://jonbarron.info/mipnerf360/>.
- Licence: no licence file shipped; research release. `flowers.txt` and `treehill.txt` in the extra
  scenes say those two scenes may not be shared publicly, so keep them local.
- Size and count: bicycle 194, bonsai 292, counter 240, flowers 173, garden 185, kitchen 279, room 311,
  stump 125, treehill 141 full-resolution images (plus `images_2/4/8` downscales).
- Ground truth: COLMAP sparse models and `poses_bounds.npy`.
- Use: Gaussian splat trainer and viewer tests on well-covered object-centric captures.

### COCO val2017
- Source: <http://images.cocodataset.org/zips/val2017.zip> and `annotations_trainval2017.zip`.
- Licence: annotations CC BY 4.0; images carry their Flickr licences.
- Size and count: 5,000 images, no camera data.
- Use: non-place negatives for the discovery stage. Already used by the detection benchmarks.

## Integrity status (2026-10-29)

Checks run: server size comparison (Range request, exact byte match) for every fetched file, then
`unzip -t`, `7z t`, `gzip -t` plus `tar -t`, and the published md5 for the Google Landmarks shards.

| Dataset | Result |
|---|---|
| 7-Scenes | complete. `office`, `pumpkin`, `redkitchen` were missing and were fetched in this pass |
| Cambridge Landmarks | complete for the 5 scenes held; Street (`251292`) was never fetched |
| Aachen Day-Night v1.1 | complete; extracted tree matches the image zip |
| RobotCar Seasons v2 | image archives complete; the reference SfM models (`all-merged`, `overcast-reference.db`, `query.db`) were never fetched |
| IMC PhotoTourism | complete. `brandenburg_gate` (0.7 of 4.3 GB), `sacre_coeur` (0.3 of 4.1 GB) and `temple_nara_japan` (6 MB of 3.1 GB) were truncated downloads and were resumed to full size in this pass |
| Google Landmarks v2 subset | complete; all 6 shard md5 sums match the published values |
| Tanks and Temples | all 15 image zips pass `unzip -t`. The 7 `*_gt.zip` files fail as zips because they are PLY point clouds; their header vertex counts match the byte sizes exactly. No server to size-check against (manual download) |
| ETH3D | complete; all four 7z archives pass `7z t`; extracted trees present for all 18 scenes |
| Mip-NeRF 360 | complete; both zips pass; 9 extracted scene trees present |
| COCO val2017 | complete |

Every other file matches the server size byte for byte and passes its archive test (67 archives
tested, 0 failures after the refetch). Not fetched and not a defect: Cambridge Street, the RobotCar
reference models, the withheld Tanks and Temples advanced scenes, and the other ~490 Google Landmarks shards.

Reproduce with:

```bash
.venv/bin/python scripts/datasets/fetch_spatial_sets.py verify --root data/spatial --test-archives
.venv/bin/python scripts/datasets/fetch_spatial_sets.py fetch  --root data/spatial --only imc-phototourism
```
