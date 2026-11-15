# 13b: crop vectors from one encoder pass over the photo

| Field | Value |
|---|---|
| Status | revision 2 (2026-11-15), by the coordinator: the pooled design of revisions 0 and 1 is REJECTED by its own high-resolution gate (section 5); sections 2 to 4 are kept as the record of what was tried; the decision in force is section 6 |
| Refines | [13_backend_speed_plan.md](13_backend_speed_plan.md) track T3 (encoder cost) and item E3; [14_roadmap.md](14_roadmap.md) phase 1b |
| Not in scope | faces, text reading, the vector index (plan 15) |

## 0. The user's job

A person labels objects cluster by cluster. The clusters are computed from one vector per object
crop. Today that vector costs one encoder call per crop, and after the engine's host cost was cut
the GPU is the bound at about 20 ms per 20-megapixel photo, most of it those calls. The crop vector
must get cheaper without making the clusters worse.

## 1. What was measured (prototype, public sets; GPU times are arithmetic from encoder rates)

| Option | Crop retrieval R@1, COCO / Cars | Cluster NMI, COCO / Cars | Text to crop mAP, COCO / Cars | GPU ms per photo at 1 / 3 / 10 boxes |
|---|---|---|---|---|
| Today: PE-L per crop, centre crop | 0.673 / 0.870 | 0.544 / 0.897 | not scored | 8.3 / 16.7 / 45.8 |
| PE-L per crop, letterbox (the crop rule fixed) | 0.712 / 0.913 | 0.596 / 0.921 | 0.806 / 0.906 | same |
| PE-B per crop, letterbox | 0.730 / 0.888 | 0.591 / 0.881 | 0.741 / 0.861 | 4.7 / 5.6 / 9.1 (centre variant) |
| Pooled from the frame pass at 336 | 0.704 / 0.923 | 0.622 / 0.940 | 0.779 / 0.899 | 4.2, any number of boxes |
| Pooled from a frame pass at 448 | 0.701 / 0.925 | 0.631 / 0.935 | 0.795 / 0.907 | 8.4, any number of boxes |

By box size on COCO (photos about 640 px; small under 32 px, medium 32 to 96, large above), R@1 and
NMI: per crop letterbox 0.698 / 0.408, 0.790 / 0.599, 0.660 / 0.605; pooled at 336 0.711 / 0.519,
0.750 / 0.617, 0.648 / 0.618; pooled at 448 0.750 / 0.571, 0.770 / 0.638, 0.650 / 0.617.

Not measured: the pooling step's own time; the real time through the model server; small objects
in large photos, where a per-crop call sees the original pixels and the pooled vector does not.

## 2. Decision

The crop vector is pooled from the frame pass at 336. Reasons: the best cluster quality on both
sets, which is what labelling uses; no loss on small boxes at COCO's scale; text search over crops
still works with the same text tower; the cost no longer grows with the number of boxes; one model
instead of two calls. It is a new vector space; the data is re-ingested for the storage change
anyway (plan 15), so nothing is migrated.

The decision is conditional on section 4's gates measured in the product. If the high-resolution
gate fails, the fallbacks are, in this order: the frame pass at 448; a second pass over tiles for
photos whose boxes are small against the encoder's patches (as the face branch does). Two crop
vector spaces at once are not an option: clusters must compare every crop with every other.

## 2a. The frame vector moves to the same pass (revision 1)

The prototype pooled from a pass over the whole photo letterboxed to the encoder's input, so every
box has tokens. Today's frame vector is made from the centre square, which drops the sides of a
wide photo; the two differ (median cosine 0.962), and the letterbox one scored 0.5575 against 0.5477
on COCO text-to-image R@1. One pass cannot give both. Decision: the letterbox pass is the only
pass, and its frame vector is the frame vector. The frame space is therefore new as well. The
near-duplicate thresholds were calibrated on today's space and are re-calibrated in lane P2.

The model's own attention-pooling head is applied to a box's tokens. It runs as an exported model,
not as a hand-written copy of the layer, unless that costs more than about 1 ms per photo; either
way it is tested against the reference head.

## 3. Lanes

| Lane | Work |
|---|---|
| P1 (engine) | The encoder exported with its patch tokens as an output beside the frame vector (same weights; the frame vector must not change). A pooled-crop branch in the frame engine: tokens and boxes in, one vector per box out, computed on the GPU, the pooling rule taken from the prototype and named in the code. The plan vocabulary of protocol version 1 gains the branch; nothing else on the wire changes. Gates of section 4 measured engine-alone. |
| P2 (API and store) | The new space registered as the default crop space; ingest asks for the pooled branch; the drawn-object, geometry-edit, reprocess and re-index paths ask the engine for one pass per photo; the per-crop encoder branch, the centre-crop rule and the conditional crop-encoder rule are deleted in the same lane. Starts when P1's gates pass. |

## 4. Gates (ground truth, through the engine)

| Gate | Bar |
|---|---|
| Export check | the new model fed today's centre-square input gives today's frame vector: cosine at least 0.9999 on 200 COCO and 100 set P photos |
| Frame vector (letterbox pass) | COCO text-to-image R@1 on the 5,000 captions and image-to-image retrieval not below today's; the cosine distributions of near-duplicate pairs and of random pairs reported for both frame vectors |
| Quality, public | COCO crop R@1 at least 0.70 and NMI at least 0.61; Cars R@1 at least 0.92 and NMI at least 0.93; text to crop mAP at least 0.77 on COCO headline boxes |
| Quality, high resolution | on labelled photos of 12 megapixels or more (set P's labelled vehicle classes as numbers only, and any public large-photo set on the host), by box size: pooled at 336 against per crop letterbox and pooled at 448. Pooled at 336 passes if NMI is within 2 points on every size bucket; otherwise the lane stops and reports, and the coordinator picks the fallback |
| GPU cost | at most 6 ms of GPU per photo for the frame and every crop vector at 1, 3, 10 and 50 boxes, measured from the model server's counters and the engine's stage timers |
| Rate, engine alone, 32 in flight | COCO at least 60 photos per second (25.9 today); the 20 MP set at least 60 (46 to 48 today); the bound that remains is named with its evidence |
| Memory | device memory per photo in flight stated; the 12 GB profile still fits the full plan |

## 5. What the engine lane measured (revisions 0 and 1 rejected)

The pooled branch was built in the engine on a lane branch and measured through it.

| Gate | Result |
|---|---|
| Export check | pass: minimum cosine 0.999995 against today's frame vector |
| Public quality | pass: COCO crop R@1 0.704, NMI 0.623; Cars 0.922, 0.938; text to crop mAP 0.780 |
| High resolution: objects of 32 to 512 px placed in 12 to 22 megapixel photos, cluster NMI, pooled against a call per crop | MISS by far: COCO classes 0.214 / 0.233 / 0.276 / 0.361 / 0.612 against 0.736 / 0.760 / 0.760 / 0.760 / 0.766; car models 0.584 / 0.578 / 0.573 / 0.582 / 0.641 against 0.848 / 0.915 / 0.929 / 0.924 / 0.924. A pass at 448 does not close it |
| Letterbox frame vector | text to image R@1 0.558 (bar 0.548: pass); copy detection R@1 0.865 against 0.882 (miss) |
| GPU cost | 6.3 to 6.6 ms per photo from the model server's counters (bar 6: miss by 5 to 10%) |
| Rate, engine alone, 32 in flight | COCO 68.8 to 70.3 photos per second (pass); the 20 MP set 46.6 to 48.2 (miss); the bound is the engine's network loop thread and the interpreter lock (held in 0.78 of samples), GPU 55 to 60% busy |

Why it fails: an object that is small against the photo covers one or two encoder patches, so its
pooled vector is the scene around it. That helps to tell COCO classes apart and carries nothing to
tell two car models apart. A pass zoomed so that the object spans at least 12 patches per side
restores the quality of a call per crop (and no less than 12 does), but then costs about as much;
and vectors of one object taken at different zooms are no closer to each other (cosine 0.82 to
0.84) than two different cars of one class, so zoom levels cannot be mixed in one cluster space.

The lane branch is kept unmerged as a record. Nothing of it is in the product.

## 6. Decision in force (revision 2)

What the labelling loop needs from a crop vector is to tell objects of ONE detector class apart
(clusters are computed within a detector class), at every object size. So:

1. A call per crop stays. The crop is letterboxed to the encoder's input (the centre-crop rule,
   which keeps about half of a wide box, is deleted): measured +4 points of retrieval on COCO and on
   Cars, and 0.85 to 0.93 cluster NMI on car models from 32 to 512 px objects in large photos.
2. The crop encoder becomes a named model of its own in the engine's plan: its own input size, its
   own preprocessing, its own vector space, independent of the frame encoder. First filling, no new
   weights: today's encoder at a 224 input for crops, measured at 2.1 times the images per second
   with equal crop quality (COCO crop R@1 0.663 against 0.674, NMI equal; Cars 0.955 against 0.954).
3. Whether a smaller dedicated crop encoder replaces it is decided by a measurement on fine-grained
   ground truth with the letterbox rule, at controlled object sizes, on at least three public
   fine-grained sets and on set P's labelled vehicle classes (numbers only): today's encoder at 336
   and 224, PE-Core-B/16, SigLIP 2 B/16 (2,217 images per second, the best car-model clusters on
   whole photos so far, its own text tower). The bar: fine-grained cluster NMI not below today's
   encoder with the letterbox rule by more than 1 point on any set or size; text-to-crop search
   reported. The engine change of point 2 makes the swap a model and a setting, not a rewrite.
4. The frame vector does not change (the centre square stays): the letterbox frame vector gained
   on text search and lost on copy detection.
5. After the crop cost falls, the engine's own front is the next bound (about 70 photos per second
   on small photos, about 47 on 20 megapixel ones, with the GPU half idle). That is the input to
   the engine's native rewrite (plan 14 section 4), not to this plan.

| Lane | Work | Gates, through the engine |
|---|---|---|
| C1 (engine and API) | the crop encoder as its own named model with input size and letterbox preprocessing; the 224 build of today's encoder through the documented export path; the centre-crop rule deleted; the crop space registered as new | COCO crop R@1 and NMI not below the 336 letterbox numbers by more than 1.5 points; Cars and the placed-object sets at 32 to 512 px likewise; text to crop mAP reported; GPU ms per crop from the model server; rate at 32 in flight on COCO and the 20 MP set with the named bound |
| C2 (measurement) | point 3 | its bar |

