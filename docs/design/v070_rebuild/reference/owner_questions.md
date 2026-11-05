# 00d: the research owner questions, de-duplicated (one list)

| Field | Value |
|---|---|
| Status | written 2026-11-04 by the research integration pass; the owner answered 11 rows the same day (section 0, recorded in the [decision_register.md](decision_register.md) register); a further 59 were answered as recommended on 2026-11-05 (section 0b); RD-61, RD-62 and RD-67 were answered by the owner the same day (below); **7 stay open, all asked after the tag (batch 3)**: RD-9, RD-11, RD-12, RD-13, RD-20, RD-26, RD-78, each with a recommendation the plan already follows |
| Source | the section 10 owner questions of the seven research plans (112 raw: GPU pipeline J1-J12, clustering J1-J17, embeddings J1-J14, near-duplicates J1-J15, spatial J1-J15, plate text J1-J13, auto-label J1-J26). Merged where two plans ask the same thing; the "From" column keeps every source, so no raw question is lost |
| How to answer | one word per row is enough ("yes", "no", or the other option). The 7 open rows are asked in small batches. The plan already follows each recommendation and marks the place "pending RD-n", so a "no" is a small, named removal |
| Where answers go | the decision register of [decision_register.md](decision_register.md) section 6 (one row per answered RD-n), then the research plan's own J-row is marked answered |

## 0. Answered by the owner on 2026-11-04 (approved; the rows are removed from the tables below)

| ID | Question | Answer, in plain words | Where the plan follows it |
|---|---|---|---|
| RD-1 | Build every research seam into the rebuild now? | **Approved as recommended: every seam now.** The review's deferrals (Q-A of [review_2_research.md](review_2_research.md)) are rejected, the plate-text index normalizer included; its close, settings and reopen step on upgraded indexes is made safe and bounded (per index, progress, dry run, rollback, stated downtime). "Keep all seams now and ensure the plans are updated accordingly to do in the right order and waves": W2.3 is split into W2.3a (ports) and W2.3b (research families) | [00_master_plan.md](../00_master_plan.md) sections 2-4; [03d](../03d_stage4_job_families.md); [02c](../02c_migration_ops_tests.md) section 2.2 |
| RD-2 | Prototypes on idle GPU during W2-W4? | **Approved, modified:** "we want to implement what we can of the research and prototypes, but we do not want long GPU hours". Working, known code first; every prototype is a short tiered run on a small seeded sample with early stop; no long sweeps; the default is the current proven approach, replaced only on a clear short-run win; **about 10 GPU hours in total before the tag** (replaces 31 + 62); anything beyond the cap needs the owner's OK | [08_research_prototype_track.md](../08_research_prototype_track.md) sections 2-4 |
| RD-4 | Plate tables in their own revision? | **Approved:** their own revision `0004_region_text` in W5e | [06e](../06e_w5e_plate_text.md); [02d](../02d_research_seam_schema.md) section 0 |
| RD-5 | Capture metadata at ingest in W4? | **Approved** with every seam (RD-1, Q-A); the reason is the capture-date filter and the plate consensus basis (review 3.6) | [05](../05_b4_stateless_api.md) section 6c |
| RD-7 | Prototype code tracked with exact pins and a lock file? | **Approved** | 00e header |
| RD-8 | Set P on local NVMe, "met / not met" only? | **Approved**; one shared seeded copy under one manifest (review 3.7) | [11](risks_cleanup_host.md) rule 10 |
| RD-10 | The research datasets, local only, aggregates with citations? | **Approved, but skip UFPR-ALPR and LPLC** (and so RodoSol): freely downloadable plate sets plus set P | the dataset register, 00e section 4.1 |
| RD-23 | Cluster memberships in PostgreSQL? | **Approved** | 02d section 2.2 |
| RD-25 | A `cluster` queue and `cluster-worker`, CPU in v0.7.0? | **Decided: yes** (review Q-D) | [03a](../03a_stage1_foundation.md) 1B-8 |
| RD-51 | Store no GPS coordinates in v0.7.0? | **Reversed: store GPS coordinates now**, behind a per-project `location_policy` (`drop`, `store_coarse`, `store`; default `store`, keep) in the W2.1 schema | [02a](../02a_postgres_schema.md) section 4; 02d section 1; 05 CM-2, CM-4 |
| RD-54 | Strip GPS from exported copies? | **No: exports keep the embedded GPS, as today**, by default; a project whose own `location_policy` is `drop` gets GPS-free exports (review RN-4, the project's policy, not a new question) | 05 CM-4, CM-5; [07](../07_release_acceptance.md) RF-84 |

Also answered from the review (review_2_research.md section 4): **Q-A** keep all seams now (above,
RD-1); **Q-B** add PP-OCRv6 as one short reader comparison against PP-OCRv5 in the plate-text and
B5c OCR work ([06c](../06c_b5_crops_ocr_faces.md), `pt:P1`); **Q-C** no academic requests: UFPR-ALPR
and LPLC are skipped (RD-10); **Q-D** the `cluster` queue and worker are decided (RD-25).

RD-3 (W5e as its own wave) stays open in name, but the RD-4 answer places the plate tables in a W5e
revision, so the plan treats W5e as settled.

## 0b. Answered as recommended on 2026-11-05 (59 rows)

The owner directed on 2026-11-05 that the planner answers what the docs and best practice can answer, and asks the owner only the rest. Each row below follows the recommendation the plan already carried; the register row is D-78 in [decision_register.md](decision_register.md). A later owner "no" is still a small, named removal. The "pending RD-n" marks in the plans are cleared for these rows.

| ID | Question in plain words | Answer | Needed by | From |
|---|---|---|---|---|
| RD-3 | Land registration-plate text inside v0.7.0 as its own wave W5e, after W5d and before W6? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes (you named plate text a must; it needs W5c's OCR output first) | before W5c | pt J3 |
| RD-6 | One GitHub milestone per post-tag research wave (RW-1..RW-6), with #466 and #398 staying on v0.8.0? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | after the tag | gpu J11; al J16 |
| RD-14 | If CPU Huffman frees enough GPU time, let a deployment choose CPU or GPU Huffman in its serving profile (never adaptive at run time)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes, one measured default | W5b | gpu J5 |
| RD-15 | A C++ Triton backend in the repository if the through-Triton gap reaches the bar? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): only on the bar, with its own plan file | W5b | gpu J6 |
| RD-16 | One Triton per card as the two-GPU default? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): decide after `gpu:PR-7`; either answer is a configuration change | RW-1 | gpu J7 |
| RD-17 | Keep the custom ctypes nvJPEG lanes until a library reaches GPU Huffman at real batch sizes with equal pixels? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes; re-test at every Triton image move | W5b | gpu J8 |
| RD-18 | DCGM profiling fields in the bench and monitoring stacks (the exporter may need extra capabilities)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes; off in the minimal install | W2.4 | gpu J9 |
| RD-19 | One INT8 second round (3 GPU hours) after the tag, closing the line for good if it fails? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-1 | gpu J10 |
| RD-21 | May the PE text tower move to a GPU TensorRT plan if it passes, with the CPU plan kept selectable per deployment? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-2 | emb J8 |
| RD-22 | A VRAM budget for an optional embedder (2 GiB text, 4 GiB vision); PE-Core-G not offered unless you ask? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-2 | emb J9 |
| RD-24 | Activate a new partition automatically when it moves no anchored or validated item and keeps id stability at 0.95 or more, else wait for a person? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3b | clu J3 |
| RD-27 | Average-linkage clustering above 150k items: block-wise with scikit-learn, no C++ build in this repository? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-3 | clu J6 |
| RD-28 | Verification tables and a local `cluster_verify` job in the product with the Cropwright review screens (v0.8.0)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes, later | v0.8.0 | clu J16 |
| RD-29 | Move the cluster-refresh and IVF-retrain environment values into the `cluster_profile` and retire the variables? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes, with the partition change in W2.3b | W2.3b | clu J17 |
| RD-30 | Keep the 0.6.0 vector field names as registry data instead of renaming them in the upgrade? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): keep (a rename rewrites every vector for no behaviour change) | W2.2 | emb J2 |
| RD-31 | Delete the in-process torch and onnx text encoders in W4 (every service requires Triton under D-69)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W4 | emb J3 |
| RD-32 | Bump OpenSearch 3.6 to 3.9 only after the 1M index runs, as its own change with a parity gate? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes, not inside v0.7.0 | RW-2 | emb J4 |
| RD-33 | Combine across different embedding spaces: copy matching spaces and mark the rest deferred (refuse only when the target lacks a target type)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3a | emb J7 |
| RD-34 | The core `visual_search_*` indexes stay outside the registry in v0.7.0? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.2 | emb J10 |
| RD-35 | The in-product evaluation is a hard gate for a space flip, with `force=true` as your waiver? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.2 | emb J11 |
| RD-36 | New box spaces as separate nested fields (write isolation) rather than a second leaf? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.2 | emb J12 |
| RD-37 | Adopt the #347 plan's calls as a block (staged A-D with its stop rule, generic names in tracked files, publish aggregates, the catalog's default VLM for captions, multilingual skipped), with the embeddings plan owning the encoder and frame experiments and clustering reading their verdicts? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-2 | emb J5, J13 |
| RD-38 | Close #435 if PE frame search misses its bar, else build a frame route over the images index (no third `target`)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-2 | emb J14 |
| RD-39 | At a holdout freeze, move the held-out item's item-level group only (not every object on near-duplicate frames)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3b | nd J2 |
| RD-40 | A separate looser "leak" cut for the holdout, equal to the dedup cut until the leakage prototype? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3b | nd J3 |
| RD-41 | Freeze and training export refuse on stale near-duplicate state (409, with a rescore button) instead of starting a rescore themselves? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3b | nd J4 |
| RD-42 | After a freeze, move new near-duplicates of held-out items into the holdout automatically (audited), rather than only excluding them from training? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3b | nd J5 |
| RD-43 | SHA-256 of the bytes replaces `imohash` as the exact duplicate key, enforced by a unique constraint? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes; the cost is measured | W2.3b | nd J6 |
| RD-44 | Core `/ingest` groups asynchronously (the response lists candidates; the group appears after the core job)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes (a wire change for core clients) | W4 | nd J7 |
| RD-45 | Export: `collapse_near_duplicates` reading the stored groups replaces the per-request `dedup_threshold`? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3b | nd J8 |
| RD-46 | A perceptual hash tier: PDQ only if it passes, else none; a copy-detection model only past its stop rule? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-4 | nd J9 |
| RD-47 | Per-project thresholds through a revisioned `near_dup_profile`, seeded from today's values? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3b | nd J10 |
| RD-48 | Region boxes inherit their item's group (no region-level grouping)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3b | nd J13 |
| RD-49 | Retire `OP_CROP_DUP_THRESHOLD` instead of validating it? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3b | nd J14 |
| RD-50 | Rename the core index fields to `near_dup_*`, keeping the `/query/duplicates` answers? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W4 | nd J15 |
| RD-52 | Never store a camera body serial raw (a keyed hash only if bursts need it)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W4 | sp J3 |
| RD-53 | Back-fill capture metadata on upgraded 0.6.0 projects (header-only read; a missing file counted)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W4 | sp J4 |
| RD-55 | Colour and view tags after the tag, each only after its prototype bar? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-5 | sp J6 |
| RD-56 | pycolmap only for matching and reconstruction (Kornia only if throughput is too low)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-5 | sp J8 |
| RD-57 | Dense reconstruction, splats and a viewer only after a GO (COLMAP dense plus gsplat first; OpenMVS only with its AGPL terms reviewed)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-5 | sp J12 |
| RD-58 | A geo-photo timeline in v0.8.0 under the location policy; reverse geocoding only if it needs place names? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | v0.8.0 | sp J13 |
| RD-59 | A scene product lives in curation, project-scoped, not in the core API? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-5 | sp J14 |
| RD-60 | The former #400 calls: run the attribute experiments on the research harness now, and file "one vote per near-duplicate group" as its own loop item? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes to both | RW-5 | sp J15 |
| RD-63 | Defer keyed-mode near-match keys until the misread rate is measured? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W5e | pt J8 |
| RD-64 | No issuing-jurisdiction field in v0.7.0? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W5e | pt J9 |
| RD-65 | Raw-mode retention optional (none keeps text until a purge)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W5e | pt J10 |
| RD-66 | Consensus writes the group's reading into `text` and keeps each frame's own reading in `text_frame`? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W5e | pt J11 |
| RD-68 | Burst grouping for plates reads the W4 capture time (no separate `captured_at` field), if the linking prototype shows time windows beat near-duplicate groups? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W5e | pt J13; sp section 5.1 |
| RD-69 | Make the `autolabel` job pausable at stage and chunk boundaries too (assist runs stay cancel-only)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3a | al J3 |
| RD-70 | Cancel is final for every job kind; "stop for now" is pause (today a cancelled loop can resume)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.2 | al J4 |
| RD-71 | When a budget limit is reached, pause the run so the budget can be raised, rather than completing it? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): pause | W2.3a | al J5 |
| RD-72 | Propagated labels stay out of the training export until a person accepts them (revisit after the teacher-student prototype)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3a | al J6 |
| RD-73 | Fold the `autolabel` job's VLM sweep into the loop as `strategy: sweep` in v0.8.0? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | v0.8.0 | al J7 |
| RD-74 | Read the daily VLM cap from `vlm_usage_hourly` and allow an optional price per million tokens on an endpoint, both in v0.8.0? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | v0.8.0 | al J9 |
| RD-75 | Retention: loop proposals and selections deleted 90 days after the run, `vlm_usage_hourly` kept 400 days? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.1 | al J10 |
| RD-76 | Box proposals stay in the region stage, not in the loop (revisit after #84)? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | - | al J12 |
| RD-77 | Loop slice length 600 s with a 30 / 35 minute slice limit, tuned after the live test? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | W2.3a | al J13 |
| RD-79 | The loop starts a training run only on a click (never by itself, never a promotion) in the first release? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | v0.8.0 | al J15 |
| RD-80 | Adopt the detector-class-roots (Z5) protocol calls as a block: about 35,200 train images with ids and hashes only; bar A; `within_key` unbarred; the `k<hash>` root key with `root_name` on the wire; `detector_class` with hold-back as the default after adoption; the IVF pool including `none`; no root cap but an audit above 20,000 members; a held-back review set; keys by normalized name; the hosted judge only after its agreement check? | **As recommended** (planner, per owner direction 2026-11-05: answer from docs and best practice): yes | RW-6 | al J17-J26 |


### Answered by the owner on 2026-11-05 (RD-61, RD-62, RD-67; D-80)

| ID | Question in plain words | Answer | From |
|---|---|---|---|
| RD-61 | Default privacy mode and deployment ceiling `raw` (today's behaviour), operators lowering the ceiling? | **Answered by the owner 2026-11-05: raw text, no privacy modes.** Plate text is ordinary searchable label data; there is no deployment ceiling. | pt J1 |
| RD-62 | Approve the privacy model (modes off, keyed, raw; the ceiling; master secret plus per-project salt; a CLI-only purge; the sink test)? | **Answered by the owner 2026-11-05: no privacy model.** Dropped: keyed and off modes, master secret, per-project salt, plate-specific CLI purge, sink-inventory test (project delete removes the text). | pt J2 |
| RD-67 | A detector trained on your private plate-box set may be used in the plate-box prototype and in your deployments, never shipped publicly? | **Answered by the owner 2026-11-05: yes.** The owner uses a private plate detector in their own deployments; other users bring their own (for example from Hugging Face) through the existing model upload path; no private model ships in the repository or the images. | pt J12 |

## A. Placement in the program

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|

## B. Research rules: code, data, people

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-9 | Set P, faces and plates never go to a hosted model unless you name a sample; faces on set P are judged by the local VLM and you only? | yes | PW-1 | clu J8, J9 |
| RD-11 | Models with unclear training-data or weight terms (SALAD, AnyLoc ViT-G, Orient Anything, the colour-name table, GeoCalib, SSCD, a dedicated plate reader): measurement only, and shipping only as an opt-in installer group with the gap stated in ATTRIBUTION? | yes | PW-2 | sp J7; nd J9 (SSCD); pt J6 |
| RD-12 | About 15 hours of your (or a delegate's) time over the research: the clustering blind re-judge (8-10 h, a second person for a 30-sheet human ceiling if one exists who has not seen the disagreement lists), the auto-label review study (about 1 h, Label Studio in the research venv), plate transcription, two near-duplicate and spatial checks; a CSV with aliased images as the re-judge tool until Cropwright has review screens? | yes | after the tag (plate transcription in PW-1) | clu J12, J14; al J8 |
| RD-13 | The hosted visual judge only after it passes planted-sheet agreement on public data; the Message Batches client only if you provide a key? | yes | after the tag | clu J13; al J26 |

## C. GPU and serving

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-20 | Hardware with a JPEG decoder (A100 or H100 class) in scope? | no; the `hw_jpeg_decoder` column keeps the door open | - | gpu J12 |

## D. Clustering

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-26 | igraph and leidenalg (GPL) as product dependencies if Leiden wins? | decide only on a Leiden PASS; fallback scikit-learn average linkage | RW-3 | clu J5 |

## E. Embeddings and search

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|

## F. Near-duplicates

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|

## G. Spatial attributes and capture metadata

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|

## H. Registration-plate text

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|

## I. The auto-label assistant

| ID | Question in plain words | Recommendation | Needed by | From |
|---|---|---|---|---|
| RD-78 | The first product precision claim needs a Wilson lower bound of at least 0.93 and a point estimate of at least 0.95 against a blind human audit, on a public 10,000-image project? | yes | RW-6 | al J14 |

## Count

112 raw questions; 80 after merging. **11 answered by the owner on 2026-11-04** (section 0), **59 answered as
recommended on 2026-11-05** (section 0b, planner, per owner direction), **3 answered by the owner on
2026-11-05** (RD-61, RD-62, RD-67); **7 open**: RD-9, RD-11, RD-12, RD-13, RD-20, RD-26, RD-78, all
asked after the tag (batch 3). None is needed before any wave.
