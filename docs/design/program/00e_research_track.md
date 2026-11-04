# 00e: the research prototype track and the research waves

| Field | Value |
|---|---|
| Status | plan, written 2026-11-04 by the research integration pass; nothing here has run |
| What it holds | every research prototype of the seven research plans in one order: its plain name, data, GPU hours, what it waits for, the window it runs in, the decision it feeds and the wave that lands that decision; the post-tag research waves RW-1..RW-6 in lane detail. [00_master_plan.md](00_master_plan.md) carries the summary (sections 2, 3, 4.10, 5) |
| Pending | the windows during W2-W4 are the recommended answer to owner question RD-2 ([00d](00d_owner_questions.md)), which amends MQ-9 ("research experiments after the tag"). With RD-2 = "no", every PW row below moves to the first post-tag window and the decisions it feeds are made on today's defaults |
| Rules | the prototype protocol of each research plan (its section 6) and the host rules of [11](11_risks_cleanup_host.md) section 1, rules 9 and 10: local Python venvs, no Docker image builds, GPU 0 then GPU 2 when agreed, one GPU job at a time under the per-card lock, a written GPU-hour cap per session, a pause when the program or a peer asks, set P under one rule (RD-8) |

## 1. Prototype names

Three plans number their prototypes `P0`.. and the clustering plan asked the GPU plan to prefix its
own, so the program writes every prototype with its topic first (resolution X-12 of
[00b](00b_traceability.md) section 5.1). The research plans keep their own numbering inside.

| Prefix | Topic | Plan | Prototypes |
|---|---|---|---|
| `gpu:` | GPU image processing at speed | [research_gpu_pipeline_plan.md](../research_gpu_pipeline_plan.md) section 6 | PR-0..PR-9 |
| `clu:` | clustering and the visual judge | [research_clustering_plan.md](../research_clustering_plan.md), protocol in [research_clustering_plan_b_protocol.md](../research_clustering_plan_b_protocol.md) section 6 | CL-0..CL-9 |
| `emb:` | embeddings and search | [research_embeddings_search_plan.md](../research_embeddings_search_plan.md) section 6.6 | EP-0..EP-7 |
| `nd:` | near-duplicates | [research_near_duplicates_plan.md](../research_near_duplicates_plan.md) section 6 | P0..P7 |
| `sp:` | spatial attributes and capture metadata | [research_spatial_attributes_plan.md](../research_spatial_attributes_plan.md) section 6 | P-0..P-9 |
| `pt:` | registration-plate text | [research_plate_text_plan.md](../research_plate_text_plan.md) section 6 | P0..P6 |
| `al:` | the auto-label assistant | [research_auto_label_assist_plan.md](../research_auto_label_assist_plan.md) section 6.4 | AL0..AL6 |

## 2. Windows

| Window | When | GPU | Never overlaps | Contents (section 3) | GPU hours |
|---|---|---|---|---|---|
| CPU slots | any time from W2.0 | none | a sweep's T1 run on the same host (nice 19, at most 8 threads) | `clu:CL-0`, `emb:EP-0`, `nd:P0` (CPU part), `pt:P0`, `sp:P-0`, `sp:P-1`; after the tag the CPU scoring of every topic | 0 |
| **PW-1** | during W2.1 and W2.2 (the rebuild uses no GPU) | GPU 0; GPU 2 when agreed | the W2.1 and W2.2 part-check bring-ups (CPU, but they run the stack) | `gpu:PR-0`, `gpu:PR-1`, `gpu:PR-2`, `nd:P0`, `nd:P1` | about 7 |
| **PW-2** | during W2.3 and W2.4, outside live harness run 2 | GPU 0; GPU 2 when agreed | live run 2 (W2.4) | `gpu:PR-3`, `gpu:PR-4`, `pt:P1`, `pt:P2`, `pt:P3` | about 15 |
| (W3) | GPU window 1 holds GPU 0 for about 1.5 days | GPU 2 only, and only when agreed | GPU window 1 on GPU 0 | nothing planned; a PW-2 or PW-3 overrun may use GPU 2 with consent | 0 |
| **PW-3** | during W4, outside live harness run 3 | GPU 0; GPU 2 when agreed | live run 3 (W4), the 4B-5 benchmark (first job of GPU window 2) | `gpu:PR-5`, `gpu:PR-6`, `pt:P4`, `pt:P5`, `pt:P6`, `nd:P2`; `al:AL0` only if room remains | about 9 (12 with AL0) |
| GPU windows 2 and 3 | W5a-W5e, W6 | program use only | - | no prototype; the W5e gate is the program's | 0 |
| **RG-1** | first window after the tag | both A6000s, one agreed slot | any v0.8.0 GPU window (V0, V1, V-EM, V2) | `gpu:PR-7`, `gpu:PR-8` | about 3.5 |
| **RG-2** | after RG-1 | GPU 0 | as above | `clu:CL-2`, `clu:CL-3`, `emb:EP-1`, `emb:EP-2`, `emb:EP-4` stages A-B (one F1 encode shared, "encode once") | about 6 |
| **RG-3** | any time after RG-2 | GPU 0 (the VLM up once for all three) | as above | `al:AL0` (if not in PW-3), `al:AL1`, `al:AL4`, `clu:CL-1` local half, `clu:CL-6` | about 10 |
| **RG-4** | after `clu:CL-2`'s verdict | GPU 0, GPU 2 when agreed | as above | `clu:CL-4`, `clu:CL-7`, `clu:CL-9`, `emb:EP-3`, `emb:EP-6` (one COCO train2017 encode shared) | about 15 |
| **RG-5** | after RG-4 | GPU 0 | as above | `emb:EP-4` stage C, `emb:EP-5`, `sp:P-2`, `sp:P-3`, `sp:P-4`, `sp:P-5`, `sp:P-8`, `nd:P3`, `nd:P4`, `nd:P6` | about 12 |
| **RG-6** | after RG-5 | GPU 0 | as above | `al:AL5`, `al:AL6`, `gpu:PR-9`, `emb:EP-7` | about 16 |

A preprocessing change adopted in B5 (decode, resize, crop) must pass the embedding parity gate
before RG-2, else the stored vectors the research reads are stale (clustering protocol section 6.1).

## 3. Every prototype

GPU hours are the plans' estimates (tier 1, tier 2 and one confirmation). "Feeds" is the decision;
"Lands in" is the wave whose lane uses it.

| Prototype (plain name first) | Data | GPU h | Other time | Waits for | Window | Feeds | Lands in |
|---|---|---|---|---|---|---|---|
| the harness and the honest GPU number (`gpu:PR-0`) | 50 images of sets A, B, P | 1 | - | none | PW-1 | 5a-R0; the programme stop rule (SM active above 85 % with PE at least 70 % of GPU ms) | W5a |
| the decoder matrix (`gpu:PR-1`) | sets B, P, A, OI 1,024 px, EXIF fixtures | 3 | - | PR-0 | PW-1 | 5a-S4; the decoder per size class (RD-14, RD-17) | W5a, W5b |
| resize and crop on CV-CUDA (`gpu:PR-2`) | sets B, P, A | 1.5 | - | PR-0 | PW-1 | 5b-G2 (a library that passes replaces the custom kernels) | W5b |
| the resident-frame pipeline, direct (`gpu:PR-3`) | sets B, P | 4 | - | PR-1, PR-2 | PW-2 | the direct ceiling; one instance with streams vs two | W5b |
| the through-Triton cost (`gpu:PR-4`) | sets B, P | 3 | - | PR-3 | PW-2 | 5b-G4 (a C++ backend only on the bar, RD-15); #403 MPS | W5b |
| faces on uint8 and the GPU (`gpu:PR-5`) | Open Images faces, set A persons, LFW pairs | 1.5 | - | PR-0 | PW-3 | 5d-P5 | W5d |
| the OCR recognition tail (`gpu:PR-6`) | OCR set, 1,500 COCO text regions | 1 | - | PR-0 | PW-3 | 5c-O2, 5c-O3 | W5c |
| two cards (`gpu:PR-7`) | set B x 4, set A, set P | 2 | - | PR-3, PR-4 | RG-1 | one Triton per card (RD-16); device-scoped queues | RW-1 |
| interactive under bulk (`gpu:PR-8`) | set B bulk, set A interactive | 1.5 | - | PR-4 | RG-1 | Triton priorities or a reserved instance | RW-1 |
| opt-in INT8, second round (`gpu:PR-9`) | the pinned 2,000, CARS196, SOP | 3 | - | none | RG-6 | INT8 line closes for good or ships opt-in (RD-19) | RW-1 |
| registration, harness, judge fixes (`clu:CL-0`) | none | 0 | 2 CPU h | none | CPU slot | every later CL read | - |
| the judge per kind (`clu:CL-1`, #406) | COCO, OI vehicles, LFW, a set-C sample | 2 | about 300 hosted sheets; 4-5 person h | CL-0 | RG-3 (local half), CPU and person slots | which kinds a judge may rank (RD-13) | RW-3 |
| class-level items (`clu:CL-2`) | F1; CUB, SOP; set P samples | 1 | 6 CPU h | CL-0 | RG-2 | a new method value or "IVF stays" | RW-3 |
| region boxes (`clu:CL-3`) | B1 | 1 | 2 CPU h | CL-0 | RG-2 | box method | RW-3 |
| scale with quality (`clu:CL-4`) | COCO train2017 (about 860k crops), Deep1B and Text-to-Image slices | 4 | 12-16 CPU h | CL-2 verdict | RG-4 | the 6 h `cluster_run` limit; size ceiling of a winner | RW-3 |
| stability and incremental assignment (`clu:CL-5`) | F1, set P | 0 | 3 CPU h | CL-2 or CL-4 | CPU slot | carry-over and bands defaults; S-later constraints | RW-3 |
| faces (`clu:CL-6`) | LFW, CALFW, CPLFW, OI faces, set P faces (local only) | 3 | 4 CPU h; 1.5 person h | CL-0 | RG-3 | person grouping method (faces lane L5) | RW-3 |
| viewpoint and prototypes (`clu:CL-7`) | CO3D, Objectron (local), F1 strata | 1 | 2 CPU h | CL-2 | RG-4 | prototypes per cluster | RW-3 |
| whole images (`clu:CL-8`) | `coco_bench_2000`, OI val sample | 0 | 2 CPU h | CL-0 | CPU slot | flat IVF or hierarchical k-means | RW-3 |
| GPU agreement and scale (`clu:CL-9`) | COCO train2017; Deep1B slices | 4 | - | CL-2 | RG-4 | the RAPIDS layer (RD-25); `cluster_backend_validations` rows | RW-3 |
| registration (`emb:EP-0`) | none | 0 | docs | none | CPU slot (any time) | every later EP read | - |
| is the offline harness the product (`emb:EP-1`) | D1 | (in RG-2's 3) | - | EP-0 | RG-2 | trust in every EP row | RW-2 |
| PE whole-frame search (`emb:EP-2`, #435) | D1, D1k; set P 200 queries | (in RG-2's 3) | - | EP-1 | RG-2 | #435 closes or a frame route (RD-38) | RW-2 |
| index settings at 1M real vectors (`emb:EP-3`) | D6, D7, D8; set P vectors | (in RG-4's 6) | CPU index runs | EP-1 | RG-4 | OpenSearch 3.9 and quantization (RD-32) | RW-2 |
| the #347 encoder evaluation (`emb:EP-4`) | D1-D5, D9; set P confirmation | stages A-B in RG-2, stage C 6 in RG-5 | - | EP-1 | RG-2, RG-5 | optional spaces; #347 | RW-2, v0.8.0 |
| hybrid OCR text search (`emb:EP-5`) | D3 with PP-OCRv5 strings | (in RG-5) | - | EP-4 placement A | RG-5 | a text space for OCR | RW-2 |
| text encode on the GPU and a shared cache (`emb:EP-6`) | D1 captions; set P query log | (in RG-4's 6) | - | EP-1 | RG-4 | RD-21 | RW-2 |
| the dual-field swap rehearsal (`emb:EP-7`) | D6 subset, `-p op-rs-emb-swap` | 1 | CPU | the v0.7.0 seams shipped | RG-6 | v0.8.0 F9 readiness | v0.8.0 F9 |
| harness and parity (`nd:P0`) | public | 0.2 | 2 CPU h | none | PW-1 (CPU part any time) | trust in every nd row | - |
| copy-level cuts per level and space (`nd:P1`) | constructed copies, Copydays, DISC21 sample | 1-2 | 2 CPU h | P0 | PW-1 | `near_dup_profile` seed defaults (RD-47): read before the W2.3 cut, ND-A seeds them; else ND-E | W2.3 or RW-4 |
| exact, perceptual and copy-model tiers (`nd:P2`) | as P1 | 0.5 | 1 CPU h | P1 | PW-3 | SHA-256 cost (RD-43); PDQ (RD-46); SSCD | RW-4 |
| bursts and sequences (`nd:P3`) | California-ND, Holidays, set P | 0.3 | 1 person h | P1 | RG-5 | burst grouping | RW-4 |
| crop level (`nd:P4`) | constructed copies, set P | 0.5 | - | P1 | RG-5 | a second item-level candidate source | RW-4 |
| holdout leakage (`nd:P5`) | constructed copies, California-ND, set P | 0 | 1 CPU h | P1, P4 | CPU slot | the `leak` cut (RD-40) | RW-4 |
| throughput at high resolution (`nd:P6`) | `commons_hires_500`, set P, OI 100k vectors | 0.5 | 1 CPU h | P2 | RG-5 | ingest cost of a hash tier | RW-4 |
| visual check (`nd:P7`) | boundary sheets of P1, P3, P4 | 0 | 2 person h | P1, P3, P4 | person slot | sign-off of the cuts | RW-4 |
| registration and fetch checks (`sp:P-0`) | - | 0 | docs | none | CPU slot | every sp read | - |
| capture parse against ExifTool (`sp:P-1`) | ExifTool samples, public sets, YFCC100M metadata, set P counts | 0 | CPU minutes | P-0 | CPU slot during W2-W3 | precondition of the W4 capture lane (RD-5) | W4 |
| view tag (`sp:P-2`) | Pascal3D+, CompCars local | 0.5 | CPU | P-0 | RG-5 | attribute signal `view` (RD-55) | RW-5 |
| colour (`sp:P-3`) | PACO-LVIS on COCO, jitter, near-duplicate pairs | 0.5 | CPU | P-0, `nd:P1` | RG-5 | attribute signal `colour` | RW-5 |
| place retrieval (`sp:P-4`) | IMC, Cambridge, GLDv2 distractors | 2.5 | - | P-0 | RG-5 | S1 gate G-1, G-2 | RW-5 |
| pair verification (`sp:P-5`) | P-4 pairs | 1 | - | P-4 | RG-5 | G-3, G-4 | RW-5 |
| grouping (`sp:P-6`) | P-5 edges | 0 | CPU minutes | P-5 | CPU slot | G-5 | RW-5 |
| reconstruction (`sp:P-7`) | IMC test scenes | 0 | 2-6 CPU h | P-6 | CPU slot | G-6 and an S1 GO | RW-5 |
| priors and cross-time (`sp:P-8`) | MSLS, Aachen, RobotCar, YFCC geotagged | 0.5 | - | P-4 | RG-5 | reported, not gated | RW-5 |
| set P, owner-run (`sp:P-9`) | set P, frozen sample | owner's | owner's | P-1, P-4..P-6 | owner's window | counts only, no location value | RW-5 |
| data and harness (`pt:P0`) | train split, manifests, local transcription | 0 | CPU, person | none | CPU slot | every pt read | - |
| which reader (`pt:P1`) | Open Images, OpenALPR, CCPD, LPLC, set P | 6 | - | P0 | PW-2 | the W5e reader (RD-11) | W5e |
| the confident tier (`pt:P2`) | dev halves | 1 | - | P1 | PW-2 | `text_confident_char_min` (needs `rec_char_min`, W5c) | W5e |
| how much bursts add (`pt:P3`) | UFPR-ALPR, set P | 1 | - | P1 | PW-2 | the consensus method | W5e |
| which frames are one vehicle (`pt:P4`) | set P capture times | 1 | - | P0, `sp:P-1` | PW-3 | the consensus basis (RD-68) | W5e |
| reader throughput (`pt:P5`) | 1,000 plate crops | 1 | - | P1 | PW-3 | the reader lane | W5e |
| plate boxes: segmenter vs a private-set detector (`pt:P6`) | set P, Open Images | 4 | - | P0 | PW-3 | plate box source (RD-67) | W5e |
| label rate (`al:AL0`) | 500 COCO and 500 set-P crops | 3 | - | none | PW-3 if room, else RG-3 | budget defaults | RW-6 |
| VLM precision and noise model (`al:AL1`) | COCO project, OI val pin, LVIS, set P | 1.5 | - | AL0 | RG-3 | the propagation arm | RW-6 |
| human review cost and precision (`al:AL2`) | public crops in Label Studio | 0 | about 1 person h | AL3 | person slot | review modes (RD-12) | RW-6 |
| selection strategies by replay (`al:AL3`) | AL1 caches | 0 | CPU, 5 seeds | AL1 | CPU slot | a new default strategy only past the bar | RW-6 |
| cheap proposers and triage (`al:AL4`) | AL1 crops | 0.5 | - | AL1 | RG-3 | proposers for ordering or acceptance | RW-6 |
| teacher-student and promotion (`al:AL5`, #84) | COCO vehicles | 6 | - | AL1 | RG-6 | propagated labels in export (RD-72) | RW-6 |
| detector-class roots (`al:AL6`, Z5, #398) | the Z5 set | 6 | 2 CPU days | `clu:CL-0` harness | RG-6 | Z5-I (RD-80) | RW-6 |

## 4. GPU hours

| Part | GPU hours |
|---|---|
| before the tag, PW-1..PW-3 (pending RD-2) | about 31 (about 34 with `al:AL0`) |
| after the tag, RG-1..RG-6 | about 62 (about 59 if `al:AL0` ran in PW-3) |
| **total** | **about 93**, up to about 100 at the plans' upper estimates (the clustering plan's own J15 says about 23 h for its part where section 6.1 of its protocol sums about 17) |
| per topic | GPU pipeline 21.5; plate 14; near-duplicates 3.5-4.5; spatial 5 (plus the owner's set-P run); embeddings 16; clustering 16-23; auto-label 17 |

Person time over the whole track: about 15 hours (RD-12).

## 5. The research waves after the tag (RW-1..RW-6)

Each wave: an integration branch `int/rw<n>-<topic>`, lanes in their own worktrees, the prototype
lanes first (tracked code under `scripts/research/<topic>/`, registration commits before any score),
then the landing lanes only on a PASS, one sweep, one PR; GitHub milestone per RD-6. Landings that
belong to v0.8.0 features are planned in [v080_platform_breadth_plan.md](../v080_platform_breadth_plan.md)
section 2.11.

| Wave | Prototypes (windows) | Landing lanes, each only on its PASS | Issues | Gate |
|---|---|---|---|---|
| RW-1 GPU serving breadth | `gpu:PR-7`, `PR-8` (RG-1), `PR-9` (RG-6) | `rw1/two-cards` (a serving profile with two servers, the generated second Triton service, router defaults, device-scoped queues derived from `gpu_devices`); `rw1/interactive-priority` (Triton `priority_levels` or a reserved instance); `rw1/int8-opt-in` (a separate `model_engines` row no default profile selects); `rw1/cuda-graphs-small` (E-5 if 5a-S5 left it); `rw1/mps` only on `PR-4` evidence; the serving head-to-head under its own rule | #403, #231, the #268 remainder | two-server router and compose tests; the E-13 confirmation rows; INT8 every #213 gate |
| RW-2 Embeddings and search | `emb:EP-0`..`EP-7` (RG-2, RG-4, RG-5, RG-6) | the v0.8.0 F9 swap reduced to "register, build, evaluate, flip" (#365); F2 `group_by` composing with `space` (#436); one optional-space feature lane per passing placement (catalog row, Triton models, ATTRIBUTION, docs, off by default); the OpenSearch 3.9 bump as its own pinned change after EP-3 | #347, #365, #435, #436 | the registered bars of EP-0; EP-7's rehearsal passes; vector-store parity overlap@10 >= 0.95 for the bump |
| RW-3 Clustering and the visual judge | `clu:CL-0`..`CL-9` (RG-2, RG-3, RG-4, CPU slots) | L3 a method value (on CL-2 or CL-4 PASS); L4 suggestions and constraints with the S-later tables (on CL-5 PASS); L5 faces on partitions (on CL-6 PASS); L6 verification tables and `cluster_verify` with the v0.8.0 Cropwright review screens | #346, #406 | the registered bars; a partition that moves a protected member or has id stability below 0.95 never ships |
| RW-4 Near-duplicate evidence | `nd:P2`..`P7` (and P0-P1 if PW-1 did not run them) | ND-E: the seeded defaults from evidence (existing projects keep their active profile revision; a new one is offered) and the PDQ tier only on a P2 pass | #416 | each P bar; the incremental-equals-full test |
| RW-5 Spatial and attributes | `sp:P-2`..`P-9` (RG-5, CPU and owner slots) | L-A1 attribute signals (revision `00NN_attribute_signals`) per passing signal; L-T1 the geo-photo timeline in v0.8.0 (RD-51, RD-58); L-SP1 scenes after an S1 GO, its own design doc first | #348, #400 | G-1..G-6; the location canary |
| RW-6 The auto-label assistant, measured | `al:AL0`..`AL6` (RG-3, RG-6, CPU and person slots) | v0.8.0 `feat/loop-review` (review queue and metrics routes, the Cropwright review grid); `feat/loop-strategies` (only AL3 or AL4 winners); `feat/loop-z5i` (on a Z5 PASS); `feat/loop-retrain` (#84) | #466 (remainder), #398, #84 | the J14 acceptance bar (RD-78) for the first product claim |
