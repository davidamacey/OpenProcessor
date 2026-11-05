# 00e: the research prototype track and the research waves

| Field | Value |
|---|---|
| Status | plan, written 2026-11-04 by the research integration pass and re-planned the same day on the owner's answers (RD-2, RD-7, RD-8, RD-10 with Q-C, Q-B); nothing here has run |
| What it holds | every research prototype of the seven research plans in one order: its plain name, data, GPU hours, what it waits for, the window it runs in, the decision it feeds and the wave that lands that decision; the post-tag research waves RW-1..RW-6 in lane detail. [00_master_plan.md](00_master_plan.md) carries the summary (sections 2, 3, 4.10, 5) |
| Decided (owner, 2026-11-04) | RD-2 and RD-7, in the owner's words: "we want to implement what we can of the research and prototypes, but we do not want long GPU hours". So: working, known code first; every prototype before the tag is a **short tiered run on a small seeded sample with early stop**, no long sweeps; the default is always today's proven approach, replaced only on a clear short-run win; **about 10 GPU hours in total before the tag**, across every topic (this replaces the earlier 31 before and 62 after); anything beyond the cap needs the owner's OK. Prototype code is tracked in `scripts/research/<topic>/` with exact pins and a committed lock file (RD-7). This amends MQ-9 ("research experiments after the tag") |
| Rules | section 2.1 below (the short-run protocol), the prototype protocol of each research plan (its section 6, now capped by section 2.1) and the host rules of [11](reference/risks_cleanup_host.md) section 1, rules 9 and 10: local Python venvs, no Docker image builds, GPU 0 then GPU 2 when agreed, one GPU job at a time under the per-card lock `gpu<N>.lock` (00b X-14), a pause when the program or a peer asks, one shared local copy of set P under one manifest (RD-8) |

## 1. Prototype names

Three plans number their prototypes `P0`.. and the clustering plan asked the GPU plan to prefix its
own, so the program writes every prototype with its topic first (resolution X-12 of
[00b](reference/traceability.md) section 5.1). The research plans keep their own numbering inside.

| Prefix | Topic | Plan | Prototypes |
|---|---|---|---|
| `gpu:` | GPU image processing at speed | [09g_research_gpu_pipeline.md](09g_research_gpu_pipeline.md) section 6 | PR-0..PR-9 |
| `clu:` | clustering and the visual judge | [09b_research_clustering.md](09b_research_clustering.md), protocol in [09c_research_clustering_protocol.md](09c_research_clustering_protocol.md) section 6 | CL-0..CL-9 |
| `emb:` | embeddings and search | [09a_research_embeddings_search.md](09a_research_embeddings_search.md) section 6.6 | EP-0..EP-7 |
| `nd:` | near-duplicates | [09d_research_near_duplicates.md](09d_research_near_duplicates.md) section 6 | P0..P7 |
| `sp:` | spatial attributes and capture metadata | [09f_research_spatial_attributes.md](09f_research_spatial_attributes.md) section 6 | P-0..P-9 |
| `pt:` | registration-plate text | [09h_research_plate_text.md](09h_research_plate_text.md) section 6 | P0..P6 |
| `al:` | the auto-label assistant | [09e_research_auto_label_assist.md](09e_research_auto_label_assist.md) section 6.4 | AL0..AL6 |

## 2. Windows

### 2.1 The short-run protocol (every prototype, owner 2026-11-04)

1. **Tier 0, CPU smoke:** 50 items, the harness end to end, no GPU. A broken harness never reaches
   a card.
2. **Tier 1, the short run:** a small seeded sample (the row's "Sample" in section 3, fixed seed,
   manifest committed), each arm compared with **today's approach as the control arm**, inside the
   row's GPU budget. **Early stop:** an arm stops as soon as its 95 % interval lies below the
   control or cannot reach the registered margin with the remaining sample; the run stops when
   every arm but the control has stopped.
3. **Tier 2, the confirmation:** only for an arm that won tier 1 by the registered margin, on a
   second seeded sample of the same size, paid from the reserve (section 4). No tier 2, no change.
4. **Default:** today's proven approach ships unless tier 2 confirms a win. A decision whose
   prototype did not run, stopped early or overran is taken on today's behaviour and the lane says so.
5. **Budget:** the coordinator keeps one ledger of GPU hours (section 4). A run that would pass its
   row's budget stops at the budget. Nothing beyond the 10-hour cap runs without the owner's OK.
6. **No long sweeps:** no full-factorial grids; at most three arms plus the control per prototype.

### 2.2 The windows

| Window | When | GPU | Never overlaps | Contents (section 3) | GPU hours (tier 1) |
|---|---|---|---|---|---|
| CPU slots | any time from W2.0 | none | a sweep's T1 run on the same host (nice 19, at most 8 threads) | `clu:CL-0`, `emb:EP-0`, `nd:P0` (its CPU part only), `pt:P0`, `sp:P-0`, `sp:P-1` | 0 (the 0.2 GPU hours of `nd:P0` are its parity run, counted in PW-1) |
| **PW-1** | during W2.1 and W2.2 (the rebuild uses no GPU) | GPU 0; GPU 2 when agreed | the W2.1 and W2.2 part-check bring-ups (CPU, but they run the stack) | `gpu:PR-0`, `gpu:PR-1`, `gpu:PR-2`, `nd:P0`, `nd:P1`, `nd:P2` (SSCD arm cut) | about 3.4 |
| **PW-2** | during W2.3a, W2.3b and W2.4, outside live harness run 2 | GPU 0; GPU 2 when agreed | live run 2 (W2.4) | `pt:P1` (with PP-OCRv6, Q-B), `pt:P2`, `pt:P3`, `pt:P4`, `pt:P5` | about 2.9 |
| (W3) | GPU window 1 holds GPU 0 for about 1.5 days | none | GPU window 1 on GPU 0 | nothing | 0 |
| **PW-3** | during W4, outside live harness run 3 | GPU 0; GPU 2 when agreed | live run 3 (W4), the 4B-5 benchmark (first job of GPU window 2) | `gpu:PR-5`, `gpu:PR-6`, `al:AL0` (shortened) | about 1.5 |
| reserve | any PW window | as the window | as the window | tier-2 confirmations only | about 2.2 |
| GPU windows 2 and 3 | W5a-W5e, W6 | program use only | - | no prototype; the W5e gate is the program's | 0 |

**Cut before the tag** (each decision taken on today's approach; each prototype may be proposed for
a post-tag window, which needs the owner's OK): `gpu:PR-3` and `gpu:PR-4` (W5b keeps today's
instance shape and the Python backend; lane 5b-G4, the C++ backend, is not opened; MPS stays off);
`gpu:PR-7`, `PR-8`, `PR-9`; `pt:P6` (plate boxes come from the segmenter with the text hint, today's
path); the SSCD arm of `nd:P2` (no copy-model tier in v0.7.0); `nd:P3`..`P7`; every `emb:` and
`clu:` GPU run (`clu:CL-0` and `emb:EP-0` stay as CPU work); `sp:P-2`..`P-9`; `al:AL1`..`AL6`.

**After the tag (not scheduled).** The windows below are the plans' proposals, kept so a later
session can load them; none opens without the owner's OK, each runs the same short-run protocol,
and each prototype's tier 1 is at most 2 GPU hours unless the owner says otherwise.

| Window | When | GPU | Never overlaps | Contents (section 3) | GPU hours (plans' upper estimate) |
|---|---|---|---|---|---|
| **RG-1** | first window after the tag, on the owner's OK | both A6000s, one agreed slot | any v0.8.0 GPU window (V0, V1, V-EM, V2) | `gpu:PR-3`, `gpu:PR-4`, `gpu:PR-7`, `gpu:PR-8` | about 10.5 |
| **RG-2** | after RG-1 | GPU 0 | as above | `clu:CL-2`, `clu:CL-3`, `emb:EP-1`, `emb:EP-2`, `emb:EP-4` stages A-B (one F1 encode shared, "encode once") | about 6 |
| **RG-3** | any time after RG-2 | GPU 0 (the VLM up once for all three) | as above | `al:AL1`, `al:AL4`, `clu:CL-1` local half, `clu:CL-6`; `pt:P6` | about 11 |
| **RG-4** | after `clu:CL-2`'s verdict | GPU 0, GPU 2 when agreed | as above | `clu:CL-4`, `clu:CL-7`, `clu:CL-9`, `emb:EP-3`, `emb:EP-6` (one COCO train2017 encode shared) | about 15 |
| **RG-5** | after RG-4 | GPU 0 | as above | `emb:EP-4` stage C, `emb:EP-5`, `sp:P-2`, `sp:P-3`, `sp:P-4`, `sp:P-5`, `sp:P-8`, `nd:P3`, `nd:P4`, `nd:P6` | about 12 |
| **RG-6** | after RG-5 | GPU 0 | as above | `al:AL5`, `al:AL6`, `gpu:PR-9`, `emb:EP-7` | about 16 |

A preprocessing change adopted in B5 (decode, resize, crop) must pass the embedding parity gate
before RG-2, else the stored vectors the research reads are stale (clustering protocol section 6.1).

## 3. Every prototype

"GPU h" is the tier-1 budget of section 2.1 for a run before the tag, and the plans' own upper
estimate for a run after it. "Feeds" is the decision; "Lands in" is the wave whose lane uses it.
Every "Data" entry is a row of the dataset register (section 4.1).

| Prototype (plain name first) | Data | GPU h | Other time | Waits for | Window | Feeds | Lands in |
|---|---|---|---|---|---|---|---|
| the harness and the honest GPU number (`gpu:PR-0`) | 50 images of sets A, B, P | 0.5 | - | none | PW-1 | 5a-R0; the programme stop rule (SM active above 85 % with PE at least 70 % of GPU ms) | W5a |
| the decoder matrix (`gpu:PR-1`) | 200 images per size class of sets B, P, A, OI 1,024 px, EXIF fixtures; today's decoder is the control | 1.5 | - | PR-0 | PW-1 | 5a-S4; the decoder per size class (RD-14, RD-17) | W5a, W5b |
| resize and crop on CV-CUDA (`gpu:PR-2`) | 200 images of sets B, P, A; today's kernels are the control | 0.5 | - | PR-0 | PW-1 | 5b-G2 (a library that passes replaces the custom kernels) | W5b |
| the resident-frame pipeline, direct (`gpu:PR-3`) | sets B, P | 4 | - | PR-1, PR-2 | RG-1 (cut before the tag) | the direct ceiling; one instance with streams vs two (W5b keeps today's shape) | RW-1 |
| the through-Triton cost (`gpu:PR-4`) | sets B, P | 3 | - | PR-3 | RG-1 (cut before the tag) | a C++ backend only on the bar (RD-15; 5b-G4 not opened in v0.7.0); #403 MPS | RW-1 |
| faces on uint8 and the GPU (`gpu:PR-5`) | 500 Open Images faces, 200 set-A persons, 1,000 LFW pairs | 0.5 | - | PR-0 | PW-3 | 5d-P5 | W5d |
| the OCR recognition tail (`gpu:PR-6`) | 500 COCO text regions | 0.5 | - | PR-0 | PW-3 | 5c-O2, 5c-O3 | W5c |
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
| harness and parity (`nd:P0`) | 200 public images | 0.2 (parity run, in PW-1) | 2 CPU h (CPU slot, any time) | none | PW-1 for the GPU part | trust in every nd row | - |
| copy-level cuts per level and space (`nd:P1`) | 2,000 constructed copies of COCO and Open Images, Copydays | 0.5 | 2 CPU h | P0 | PW-1 | `near_dup_profile` seed defaults (RD-47): read before the W2.3b cut, ND-A seeds them; else today's 0.98 | W2.3b |
| exact and perceptual tiers (`nd:P2`; the SSCD copy-model arm cut) | as P1 | 0.2 | 1 CPU h | P1 | PW-1 | SHA-256 cost (RD-43); PDQ (RD-46) | W2.3b (RD-43), RW-4 (PDQ) |
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
| which reader (`pt:P1`) | 1,000 plate crops: Open Images, the OpenALPR benchmark, CCPD strata, set P; arms PP-OCRv5 (today, control), PP-OCRv6 (owner Q-B, one comparison), one dedicated plate reader | 1.5 | - | P0 | PW-2 | the W5e reader (RD-11) and the B5c OCR model ([06c](06c_b5_crops_ocr_faces.md)) | W5c, W5e |
| the confident tier (`pt:P2`) | P1's dev halves | 0.5 | - | P1 | PW-2 | `text_confident_char_min`; the script computes the per-character minimum from the recognition logits in its own venv, so it does not wait for W5c's `rec_char_min` output (review RR-24) | W5e |
| how much bursts add (`pt:P3`) | set P bursts only (UFPR-ALPR skipped by the owner) | 0.3 | - | P1 | PW-2 | the consensus method | W5e |
| which frames are one vehicle (`pt:P4`) | set P capture times | 0.3 | - | P0, `sp:P-1` | PW-2 | the consensus basis (RD-68) | W5e |
| reader throughput (`pt:P5`) | 1,000 plate crops | 0.3 | - | P1 | PW-2 | the reader lane | W5e |
| plate boxes: segmenter vs a private-set detector (`pt:P6`) | set P, Open Images | 4 | - | P0 | RG-3 (cut before the tag) | plate box source (RD-67, answered yes: the owner's private detector stays in their own deployments; others bring their own through model upload; D-80); W5e uses the segmenter with the text hint | after the tag |
| label rate (`al:AL0`) | 100 COCO and 100 set-P crops | 0.5 | - | none | PW-3 | budget defaults (4D-5 ships today's limits until then) | RW-6 |
| VLM precision and noise model (`al:AL1`) | COCO project, OI val pin, LVIS, set P | 1.5 | - | AL0 | RG-3 | the propagation arm | RW-6 |
| human review cost and precision (`al:AL2`) | public crops in Label Studio | 0 | about 1 person h | AL3 | person slot | review modes (RD-12) | RW-6 |
| selection strategies by replay (`al:AL3`) | AL1 caches | 0 | CPU, 5 seeds | AL1 | CPU slot | a new default strategy only past the bar | RW-6 |
| cheap proposers and triage (`al:AL4`) | AL1 crops | 0.5 | - | AL1 | RG-3 | proposers for ordering or acceptance | RW-6 |
| teacher-student and promotion (`al:AL5`, #84) | COCO vehicles | 6 | - | AL1 | RG-6 | propagated labels in export (RD-72) | RW-6 |
| detector-class roots (`al:AL6`, Z5, #398) | the Z5 set | 6 | 2 CPU days | `clu:CL-0` harness | RG-6 | Z5-I (RD-80) | RW-6 |

## 4. GPU hours

| Part | GPU hours |
|---|---|
| PW-1 | about 3.4 (`gpu:PR-0` 0.5, `PR-1` 1.5, `PR-2` 0.5; `nd:P0` 0.2, `P1` 0.5, `P2` 0.2) |
| PW-2 | about 2.9 (`pt:P1` 1.5, `P2` 0.5, `P3` 0.3, `P4` 0.3, `P5` 0.3) |
| PW-3 | about 1.5 (`gpu:PR-5` 0.5, `PR-6` 0.5; `al:AL0` 0.5) |
| reserve (tier-2 confirmations only) | about 2.2 |
| **before the tag, total** | **about 10, the owner's cap** (was about 31); per topic: GPU pipeline 3.5, plate 2.9, near-duplicates 0.9, auto-label 0.5, spatial, embeddings and clustering 0 (CPU only) |
| after the tag | not scheduled; the plans' upper estimates sum to about 70 (RG-1..RG-6), each window on the owner's OK with tier 1 at most 2 GPU hours per prototype |

CPU-hours cap for the CPU-slot prototypes: the host is shared, so each runs at `nice 19` with at most 8
threads, and the CPU slots total **at most 20 CPU hours** before the tag (the tables list about 10 to 15);
a run that would pass the total stops and the owner is asked.

Person time before the tag: about 3 hours (`pt:P0` transcription checks, `nd:P1` boundary sheets).

### 4.1 Dataset register (review RR-11; owner RD-10, 2026-11-04)

The owner approved RD-10 for this register, **without UFPR-ALPR and LPLC** (and so without RodoSol):
freely downloadable sets and set P only. No set needs a signed request before the tag. Every set is
used locally, never committed or redistributed, and only aggregates are published, with a citation.
Each prototype's registration commit records the set's terms as read at download (the `P0` rows),
so a changed licence stops that prototype, not the program.

| Set | Terms (as cited by the topic plan) | Local only | Used before the tag by | Window |
|---|---|---|---|---|
| set P (the owner's private photos) | private | yes; one shared seeded copy under one manifest, removed after PW-3 and the removal recorded (11 rule 10, RD-8) | `gpu:PR-0..PR-2`, `pt:P1..P5`, `al:AL0` | PW-1..PW-3 |
| sets A and B (the program's public benchmark sets, [06a](06a_b5_baseline.md)) | public sources of 07a | yes | `gpu:PR-0..PR-2`, `PR-5` | PW-1, PW-3 |
| COCO 2017 (and COCO-Text regions) | annotations CC BY 4.0; images under their Flickr licences | yes | `gpu:PR-6`, `nd:P0..P2`, `al:AL0` | PW-1, PW-3 |
| Open Images V7 (faces, plates, vehicles, 1,024 px sample) | annotations CC BY 4.0; images listed as CC BY 2.0 | yes | `gpu:PR-1`, `PR-5`, `nd:P1`, `pt:P1` | PW-1..PW-3 |
| LFW pairs | research use, as the dataset site states | yes | `gpu:PR-5` | PW-3 |
| Copydays | research use, as the INRIA page states | yes | `nd:P1`, `P2` | PW-1 |
| OpenALPR benchmark | repository AGPL-3.0; the images' own licence is not stated | yes, never redistributed | `pt:P1`, `P5` | PW-2 |
| CCPD | README states MIT | yes | `pt:P1`, `P5` | PW-2 |
| EXIF fixtures, constructed copies | made locally from the sets above | yes | `gpu:PR-1`, `nd:P1`, `sp:P-1` | PW-1, CPU |
| ExifTool sample images, YFCC100M metadata | as their sites state; metadata only for YFCC100M | yes | `sp:P-1` | CPU |
| **dropped by the owner** | UFPR-ALPR, LPLC, RodoSol-ALPR (academic, non-commercial, on request) | - | none: `pt:P3` uses set P bursts only, `pt:P1` calibrates without LPLC's legibility levels | - |
| after the tag (on the owner's OK per window) | DISC21, CALFW, CPLFW, CO3D, Objectron, California-ND, INRIA Holidays, MSLS, Pascal3D+, PACO-LVIS, MegaScenes, CompCars, CARS196, CUB, SOP, Cambridge Landmarks, Aachen Day-Night, RobotCar Seasons, GLDv2, IMC, Deep1B, Text-to-Image, the VIBE sets: several are non-commercial; each gets a row here with its terms before its window opens | yes | RG-1..RG-6 | after the tag |

## 5. The research waves after the tag (RW-1..RW-6)

Each wave: an integration branch `int/rw<n>-<topic>`, lanes in their own worktrees, the prototype
lanes first (tracked code under `scripts/research/<topic>/`, registration commits before any score),
then the landing lanes only on a PASS, one sweep, one PR; GitHub milestone per RD-6. Landings that
belong to v0.8.0 features are planned in [10_v080_platform_breadth.md](10_v080_platform_breadth.md)
section 2.11.

| Wave | Prototypes (windows) | Landing lanes, each only on its PASS | Issues | Gate |
|---|---|---|---|---|
| RW-1 GPU serving breadth | `gpu:PR-3`, `PR-4`, `PR-7`, `PR-8` (RG-1), `PR-9` (RG-6) | `rw1/two-cards` (a serving profile with two servers, the generated second Triton service, router defaults, device-scoped queues derived from `gpu_devices`); `rw1/interactive-priority` (Triton `priority_levels` or a reserved instance); `rw1/int8-opt-in` (a separate `model_engines` row no default profile selects); `rw1/cuda-graphs-small` (E-5 if 5a-S5 left it); `rw1/mps` and a C++ backend (5b-G4, its own plan file, RD-15) only on `PR-4` evidence; the serving head-to-head under its own rule | #403, #231, the #268 remainder | two-server router and compose tests; the E-13 confirmation rows; INT8 every #213 gate |
| RW-2 Embeddings and search | `emb:EP-0`..`EP-7` (RG-2, RG-4, RG-5, RG-6) | the v0.8.0 F9 swap reduced to "register, build, evaluate, flip" (#365); F2 `group_by` composing with `space` (#436); one optional-space feature lane per passing placement (catalog row, Triton models, ATTRIBUTION, docs, off by default); the OpenSearch 3.9 bump as its own pinned change after EP-3 | #347, #365, #435, #436 | the registered bars of EP-0; EP-7's rehearsal passes; vector-store parity overlap@10 >= 0.95 for the bump |
| RW-3 Clustering and the visual judge | `clu:CL-0`..`CL-9` (RG-2, RG-3, RG-4, CPU slots) | L3 a method value (on CL-2 or CL-4 PASS); L4 suggestions and constraints with the S-later tables (on CL-5 PASS); L5 faces on partitions (on CL-6 PASS); L6 verification tables and `cluster_verify` with the v0.8.0 Cropwright review screens | #346, #406 | the registered bars; a partition that moves a protected member or has id stability below 0.95 never ships |
| RW-4 Near-duplicate evidence | `nd:P3`..`P7` and the SSCD arm of `nd:P2` (and P0-P2 if PW-1 did not run them) | ND-E: the seeded defaults from evidence (existing projects keep their active profile revision; a new one is offered) and the PDQ tier only on a P2 pass | #416 | each P bar; the incremental-equals-full test |
| RW-5 Spatial and attributes | `sp:P-2`..`P-9` (RG-5, CPU and owner slots) | L-A1 attribute signals (revision `00NN_attribute_signals`) per passing signal; L-T1 the geo-photo timeline in v0.8.0 (RD-51, RD-58); L-SP1 scenes after an S1 GO, its own design doc first | #348, #400 | G-1..G-6; the location canary |
| RW-6 The auto-label assistant, measured | `al:AL1`..`AL6` (RG-3, RG-6, CPU and person slots; `al:AL0` ran short in PW-3) | v0.8.0 `feat/loop-review` (review queue and metrics routes, the Cropwright review grid); `feat/loop-strategies` (only AL3 or AL4 winners); `feat/loop-z5i` (on a Z5 PASS); `feat/loop-retrain` (#84) | #466 (remainder), #398, #84 | the J14 acceptance bar (RD-78) for the first product claim |
