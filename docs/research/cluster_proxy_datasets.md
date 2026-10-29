# Public proxy datasets for cluster evaluation

Fetched 2026-10-29 (wave W1 of `docs/design/cluster_search_makemodel_plan.md`), at the commit
that adds `scripts/datasets/fetch_face_proxy.py` and `fetch_vehicle_proxy.py`. These sets give
the clustering work real, independently labelled data: identities for faces, make and model for
vehicles. Nothing here is a measurement of the product; it is the data the later waves score on.

Data lives on the NAS under `<datasets root>/clustering/<dataset>/`, one directory per
dataset, each with `images/`, `manifest.json` (every file: bytes and SHA-256, source, licence),
`labels.jsonl` (`path`, `class_name`, `taxonomy_path`, `view`, `identity`, plus per-source
extras) and, where the licence needs it, `ATTRIBUTION.csv`. Verify any set with
`python -I scripts/datasets/fetch_face_proxy.py verify <dir>` (or the vehicle script; same code).

## What was fetched

| Set | Directory | Images | Size | Licence | Committed |
|---|---|---|---|---|---|
| LFW | `lfw` | 13,233 images, 5,749 identities, plus the 3 pair lists | 369 MB (with the 180 MB archive) | none published; local only | pin only (`manifests/lfw.pin.json`) |
| Open Images V7 faces | `openimages_faces` | 1,500 validation images with a qualifying "Human face" box | 421 MB | images CC BY 2.0, annotations CC BY 4.0 | pin only |
| Open Images V7 vehicles | `openimages_vehicles` | 746 validation images: Car 200, Truck 200, Van 133, Motorcycle 117, Bus 67, Taxi 20, Ambulance 9 | 248 MB | images CC BY 2.0, annotations CC BY 4.0 | pin only |
| Commons set C | `commons_set_c` | 2,943 images, 74 models of 26 makes, 40 per model (a model with fewer than 15 usable files is dropped) | 893 MB | per file, see below | full manifest with per-file author and licence (`manifests/commons_set_c.json`) |

The "pin only" files hold the manifest digest, count, size and licence (`manifest_sha256`
covers every file row). A re-run with the same seed picks the same images; a changed file shows
as a size or hash mismatch in `verify`.

### LFW

Source: the figshare copy that scikit-learn uses (`ndownloader.figshare.com/files/5976018` and
the pair lists); the original `vis-www.cs.umass.edu` host did not resolve from this machine. The
archive and pair-list SHA-256 values are pinned in `fetch_face_proxy.py`, taken on first
download, so they prove the copy has not changed since, not that it equals the original
(trust on first use). Counts match the published 13,233 images and 5,749 identities. The
archive is extracted member by member; any member outside `lfw/<Person>/<Person>_NNNN.jpg` aborts
the extraction. LFW publishes no licence: the images are web-crawled news photographs. Plan O2
says local use yes, publish aggregates after review.

### Open Images

Selection (`proxy_openimages.py`): V7 validation split; images whose list licence is CC BY 2.0;
boxes that are not groups, not depictions and cover at least 0.2 % (faces) or 5 % (vehicles) of
the image. Vehicles take the largest qualifying box as the image class. The vehicle classes are
Car, Truck, Bus, Motorcycle, Van, Taxi, Ambulance (names checked against
`oidv7-class-descriptions-boxable.csv`). Only 746 validation images qualify, so the vehicle
set is small and unbalanced (Ambulance 9); the test split would add more and is one flag away
(`--split test`) into a separate directory. Labels carry the boxes. Faces have no identities.

### Commons set C

Categories are listed in `manifests/commons_set_c_categories.json` (78 requested). Per model:
files in the generation category plus one level of subcategories (interior, engine, detail and
similar subcategories excluded), JPEG, long side at least 1,000 px, licence CC0, public domain,
CC BY or CC BY-SA. Files that sit in two listed models are dropped from both. Images are the
1,280 px thumbnails Commons serves. 74 models passed the 15-file minimum; dropped for too few
files: BMW 5 Series (F10) 0, BMW X5 (E70) 0, Toyota Camry (XV50) 14, Toyota Camry (XV70) 13.

Licences in the set (files): CC BY-SA 4.0 1,318; CC BY 2.0 373; CC BY-SA 3.0 294; CC0 242;
CC BY-SA 2.0 233; public domain 199; CC BY 4.0 120; CC BY-SA 3.0 de 76; CC BY 3.0 69; 19
files in 12 further country-ported or older CC BY and CC BY-SA variants. One file has no author recorded. A photo
shown anywhere must carry the author and licence from the manifest; share-alike applies to
adaptations. Labels come from the category (make, model, generation) and are noisy: a seeded 10 %
sample (294 files) is in `review_sample.csv` for a person to mark `label_correct`. That check
has **not** been done; until it is, treat the label error rate as unknown.

## What was not fetched

| Set | Why | What the script offers |
|---|---|---|
| CPLFW | Only distribution is a Google Drive or Baidu link on the authors' page; no licence stated | none; fetch by hand if wanted and index it |
| CompCars | Requires a signed release agreement (non-commercial research, no redistribution) | `fetch_vehicle_proxy.py compcars --root <licensed copy> --make-names ... --model-names ...` indexes an existing copy in place |
| Stanford Cars | Original host gone, torchvision download broken ([vision #7545](https://github.com/pytorch/vision/issues/7545)); mirrors need a login or state no licence | `stanford --root <copy>` indexes a copy laid out as `<class name>/*.jpg` |
| VMMRdb | No licence found, and the dataset host did not answer | none |

The CompCars and Stanford indexers are tested on hand-built trees and the label parsers on the
documented naming (Stanford class string to make, model, body, year; CompCars ids to paths and
viewpoint), but they have not run on real copies, so the name-list file format for CompCars is
untested against the real distribution.

## Reproduce

```bash
PY=.venv/bin/python; C=<datasets root>/clustering; M=scripts/datasets/manifests
nice -n 10 $PY -I scripts/datasets/fetch_face_proxy.py lfw --out $C/lfw --pin $M/lfw.pin.json
nice -n 10 $PY -I scripts/datasets/fetch_face_proxy.py openimages --out $C/openimages_faces \
    --cache-dir $C/_cache --pin $M/openimages_faces.pin.json
nice -n 10 $PY -I scripts/datasets/fetch_vehicle_proxy.py openimages --out $C/openimages_vehicles \
    --cache-dir $C/_cache --pin $M/openimages_vehicles.pin.json
nice -n 10 $PY -I scripts/datasets/fetch_vehicle_proxy.py commons --out $C/commons_set_c \
    --categories $M/commons_set_c_categories.json --manifest $M/commons_set_c.json
```

`--dry-run` prints counts per class and licence without downloading images. With an existing
`--manifest`, `commons` downloads exactly the pinned files and verifies every hash. The first
Commons run took 50 minutes (listing at one request per second, two download threads).
Raw run logs: `artifacts_local/w1_*.log` (gitignored). Tests:
`tests/test_proxy_dataset_manifest.py`.
