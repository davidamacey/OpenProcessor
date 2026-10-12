# Choosing and refreshing the parity reference

`scripts/bench/parity.py compare` measures a candidate capture against a stored reference
(`artifacts_local/bench/parity/<tag>/`, gitignored). It answers "did the numbers move", not "did they
get worse". A reference is tied to the engine files it was captured with, so a TensorRT rebuild of the
same weights can fail the gates against an old reference while the candidate is as good as the engine
that made the reference. Decide that on ground truth, not on the gates.

## When the reference must be replaced

A candidate that differs from its reference only by an engine rebuild (same ONNX, same bake, same
inputs) and fails the `precision` profile. Check first that a rebuilt control, with none of the change
under test, fails the same way: then the failure is the reference's age, not the change.

## Procedure

1. Run the old and the rebuilt detector engines over the pinned set and score both against the COCO
   ground truth:

   ```bash
   # engines loaded in one Triton under distinct names, e.g. det_old (FP32 input), det_new (uint8 input)
   python scripts/bench/detector_capture.py --url 127.0.0.1:PORT --manifest scripts/datasets/manifests/coco_bench_2000.json \
       --images IMAGES_DIR --model det_old:fp32 --model det_new:uint8 --out-dir dets/
   .venv/bin/python scripts/bench/detector_quality.py --annotations instances_train2017.json \
       --annotations instances_val2017.json --manifest scripts/datasets/manifests/coco_bench_2000.json \
       --dets old=dets/det_old.json --dets new=dets/det_new.json --conf 0.4 --out report.json
   ```

   Add `--split val2017` for the images no COCO-trained detector has seen (1,279 of the 2,000; the other
   721 are train2017 images). Capture the old engine twice: the difference between its two runs is the
   noise to read the rest against.
2. Verdict. Equal or better (AP, AP50, recall and precision at the product confidence floor 0.4, and no
   net loss in the IoU of moved boxes to the truth): capture a new reference, twice (the detector is not
   bit-reproducible between runs), from the final configuration. Worse: keep the old reference, do not
   capture, and fix the engine build.
3. Capture `ref-<label>` and `ref-<label>-repeat` on the final stack, then compare the two to each other
   with the `precision` profile. That pair is the noise floor of the new reference; a gate that fails
   inside the pair cannot be read against it (P-cluster-purity sits close to its 0.5 point limit there).
4. Keep the previous reference until the new one has been used for one comparison.

## Record: old against rebuilt detector engine (set A, 2,000 pinned COCO images)

The old engine is the plan of the v0.5.0 install (built 26 Sep). The rebuilt engines come from the same
ONNX and bake on TensorRT 11.1: `rebuilt` takes FP32 input, `uint8` takes uint8 input and scales in the
graph. One RTX A6000, same images, same letterbox, boxes in original pixels. Raw reports:
`docs/benchmarks/parity/detector_quality_rebuild/`. The 721 train2017 images are inside the training
data of the COCO detector, so absolute numbers are high; the comparison between engines is what counts.

Whole set (2,000 images, 14,692 ground-truth boxes), confidence floor 0.4, IoU 0.5:

| Engine | AP | AP50 | AR@100 | precision | recall | TP | FP | FN | mean IoU of TPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old | 0.4373 | 0.5698 | 0.4900 | 0.8579 | 0.5503 | 8085 | 1339 | 6607 | 0.8735 |
| old, second run | 0.4372 | 0.5698 | 0.4900 | 0.8579 | 0.5503 | 8085 | 1339 | 6607 | 0.8735 |
| rebuilt, FP32 input | 0.4375 | 0.5701 | 0.4902 | 0.8575 | 0.5505 | 8088 | 1344 | 6604 | 0.8735 |
| rebuilt, uint8 input | 0.4376 | 0.5702 | 0.4903 | 0.8576 | 0.5505 | 8088 | 1343 | 6604 | 0.8736 |
| rebuilt, uint8, second run | 0.4376 | 0.5701 | 0.4903 | 0.8576 | 0.5505 | 8088 | 1343 | 6604 | 0.8735 |

Held-out images only (val2017, 1,279 images): AP 0.4165 (old) against 0.4168 (rebuilt, FP32) and 0.4169
(rebuilt, uint8); recall 0.5311 against 0.5312 and 0.5312; precision 0.8376 against 0.8367 and 0.8369.

Boxes at confidence 0.4 or above, old engine against each other engine, matched by class and IoU 0.5:

| Comparison | boxes matched | moved (IoU to old box < 0.99) | closer to truth | farther from truth | unchanged | mean change of IoU to truth | only in old (on / off truth) | only in new (on / off truth) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| old, second run | 9,679 | 61 | 29 | 20 | 12 | +0.0005 | 0 / 0 | 0 / 0 |
| rebuilt, FP32 input | 9,666 | 565 | 219 | 193 | 153 | -0.0007 | 4 / 9 | 10 / 12 |
| rebuilt, uint8 input | 9,666 | 558 | 218 | 186 | 154 | -0.0001 | 4 / 9 | 10 / 11 |

**Verdict: equal.** The rebuilt engine is within 0.0003 AP of the old one in every view, which is the
size of the difference between two runs of the old engine plus a few boxes at the threshold. About 6 % of
the confident boxes move by more than 1 % IoU, and they move both ways: 219 closer to the truth, 193
farther, 153 unchanged within 0.002; the mean change is -0.0007 IoU. Of the 35 boxes present in only one
engine, the rebuilt one adds 10 on a true object and 12 off one, the old one has 4 and 9. That is why the
`precision` gates (98 % of images with every box at IoU 0.95) fail against the old reference: the
engine is a different build, not a worse one. Per class, 21 of 80 classes gain more than 0.001 AP and 17
lose more than 0.001; the largest change is +0.008 AP.

The references captured from the final configuration are `ref-u8fp16` and `ref-u8fp16-repeat`; compared
to each other they pass every `precision` gate (images with identical boxes 99.4 %, crop vectors p01
0.99995, kNN overlap 0.9955, cluster purity drop 0.45 points).
