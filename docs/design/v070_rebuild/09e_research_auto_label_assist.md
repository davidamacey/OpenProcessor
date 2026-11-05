# Auto-label assist loop: research and the seams for the clean rebuild

> Contracts (tables, fields, job kinds) live in 02d_research_seam_schema.md and the lane files; this file is the rationale and the prototype design.
> A coding lane reads 02d and its lane section, not this whole file.

**Superseded parts:** the landing lanes, waves and file ownership of this plan's section 8, its GPU lock and its GPU-hour estimates are superseded by the program (review RR-16, owner answers of 2026-11-04): lanes in [03d_stage4_job_families.md](03d_stage4_job_families.md) (W2.3a and W2.3b), [03b_stage2_control_data.md](03b_stage2_control_data.md), [03f_stage6_scheduler_scaleout.md](03f_stage6_scheduler_scaleout.md), [05_b4_stateless_api.md](05_b4_stateless_api.md) and [06e_w5e_plate_text.md](06e_w5e_plate_text.md); one lock `gpu<N>.lock` per card shared by every topic ([traceability.md](reference/traceability.md) X-14); short tiered runs under a cap of about 10 GPU hours before the tag ([v070_rebuild/08_research_prototype_track.md.md](08_research_prototype_track.md) section 2.1). A fresh agent loads those files first.

Status: plan, research. Nothing here is implemented. Written 2026-11-04 against `origin/main` at
`49731595`. Part A (the seams, section 5) is design input to program B2 and must land in the B2 plan
files before B2 stage 1 starts coding. Part B (the prototypes, section 6) runs after the v0.7.0 tag
(MQ-9 in [00_master_plan.md](00_master_plan.md)) unless the owner answers J1 otherwise.
Part C (the product features beyond the B2 port, sections 7-8) is v0.8.0 work.

Issues: #466 (pause, events, token budgets, label rate), #398 (detector-class roots with a hold-back
guard; its full plan is absorbed here as section 6.8 and the old file is deleted), #84 (the
teacher-student loop on public COCO), #406 (the judge and the human re-judge; consumed, not owned),
#67 (processing plans; only the boundary is stated here). Tracker for the clustering side: #346.

Owner direction (2026-11-04): "We need the auto label to be a good place to assist in labelling,
and that we can control and pause and start it where needed." This is a clean backend rewrite:
research to best practice first; prototypes are plain local-venv GPU scripts (no Docker builds) on
set P and public data (COCO, Open Images, LVIS) to measure the label rate, label precision and the
human-review load; plan first, run nothing.

Audience: a fresh agent with no conversation memory. Each seam (section 5) names its table, column,
state, route or event, its type, and the plan file and lane that own it. Each prototype (section 6)
names its data, arms, metrics, repetitions, GPU time and stop rule.

Notation: `{P}` is `/curation/projects/{project}`. Routes written `{P}/...` that this plan adds do
not exist yet; routes written in full exist today. "Set P" is the owner's private collection of
high-resolution photos; nothing in this file names its location, folders or files.

## What this plan supersedes

| Document | Verdict | What moves here |
|---|---|---|
| `docs/design/issue_plans/issue-398-plan.md` | **absorbed and deleted** in this commit | everything: Z5 registration and run (section 6.8), Z5-I product shape (section 7.6), traps (section 9), its J1-J10 (section 10, J17-J26) |
| [auto_label_loop_plan.md](../auto_label_loop_plan.md) | **partly superseded, kept** | for the rebuild this plan replaces its job-directory storage (3.1), event file (3.2), write-ahead budget file (5.8), worker and trigger protocol (D4, 8), control routes (10) and the Run panel controls (11). Its algorithm (5.2-5.7), statistics (6) and quality safeguards (7) stay the as-built reference and are cited, not restated. It is cited by `docs/WHITEPAPER.md` and the docs-site guide, so it is not deleted |
| [selectable_assist_plan.md](../selectable_assist_plan.md) | not superseded | the `assist_run` family is a separate, short, human-driven job; the loop reuses its core (sampling, `ask_sample`, proposal) |
| [cq3_label_operations.md](../cq3_label_operations.md) | not superseded; **reused** | the loop's propagation and rollback become label operations (section 5.6) instead of a second undo mechanism |
| [processing_plans_plan.md](../processing_plans_plan.md) | not superseded; boundary only | its rule "VLM is not a branch" stands: VLM labeling stays a worker and loop feature, never an ingest plan stage |

---

## 1. Plain summary

Today OpenProcessor has four automated labelers that do not share a control surface: the
continuous VLM worker, the `autolabel` pipeline job (embed, cluster, VLM sweep), user-scoped assist
runs, and the cluster-sampled **assist loop** (#286). The loop is the one built to scale: it samples
each cluster, asks the VLM, propagates the majority class when an exact statistical gate passes,
splits what fails, audits with a second model, stays inside a hard budget and rolls back on
request. It cannot be paused by a person, it streams no events, it has no token budget, its label
rate was never measured, its precision against a person was never measured, and its yield on a
2,500-image COCO project was about one VLM crop per labeled item (#286), far from the 1:20 target.

The rebuild (program B2) moves every long job onto Celery with job rows in PostgreSQL. This plan
says what the loop needs from that rebuild so that it becomes a real labeling assistant:

1. **Its own durable record in PostgreSQL**: runs, iterations, nodes, selections, proposals, human
   review decisions, budgets and a budget ledger, plus a per-hour VLM usage table every VLM caller
   writes. The 24-hour `events` table is the UI feed, not the record.
2. **Control that means the same thing everywhere**: start, pause, resume, cancel and edit budget,
   with pause as a cooperative stop at a node boundary (never a Celery revoke), a pause reason the
   system can also set (audit failed, budget reached, project paused, VLM taken for training), and a
   handler that runs in time slices so one project's long loop never starves another's.
3. **Budgets that cannot be exceeded across processes**: one atomic SQL statement reserves crops,
   requests and tokens before every upstream call; tokens come from the endpoint's `usage` report.
4. **The lock rule at every write**: selection excludes locked items, every write re-checks under
   OCC, human review writes are label operations, and machine labels never become validated.
5. **Measurement built in**: labels per hour, VLM crops per labeled item, precision against a blind
   human sample with a Wilson interval, and review minutes per 100 labels, all computed by the
   backend and served to Cropwright.

The research (after the tag) then measures the label rate (AL0), the VLM's precision on four
datasets (AL1), the human review cost and precision (AL2), and compares selection strategies by
replaying cached real VLM answers (AL3, CPU only), before any new strategy enters the product.

---

## 2. What exists and what was measured

### 2.1 Code

| Piece | Where | State |
|---|---|---|
| Loop driver, node step, gate, propagate, split, audit, rollback, plan | `src/services/curation/assist_loop/` (`driver.py`, `node_step.py`, `evaluate.py`, `propagate.py`, `splitter.py`, `audit.py`, `rollback.py`, `plan.py`) | as built in #286; algorithm in [auto_label_loop_plan.md](../auto_label_loop_plan.md) 5-7 |
| Exact gate and tree | `src/domain/loop_stats.py` (hypergeometric `required_sample`, Wilson bound), `src/domain/loop_tree.py` (path grammar `r<int>(.<int>)*`, seeded child draws, best-first key) | pure, tested |
| Partition seam | `assist_loop/partition.py` `PartitionReader` (`survey`, `roots`, `root_of`); one implementation `ClusterFieldPartition` | the place a second partition plugs in (Z5-I) |
| Loop record | `assist_loop/store.py`: job directory, `events.jsonl` with fsync per line, replay; `assist_loop/budget.py`: write-ahead reservations in `budget.json` under a `threading.Lock` | file-based; lane 4D of B2 deletes it ([03d](03d_stage4_job_families.md) 4D-2) |
| Worker | `scripts/curation/assist_loop_worker.py` (trigger claim by `unlink`, `pause.flag`, `cancel.flag`) | deleted by lane 4D |
| Routes | `src/routers/curation/assist_loop.py`: `/curation/projects/{project}/assist/loops` (plan or start), list, state, nodes, node members, `/cancel`, `/resume`, `/rollback` | no pause route, no events, no budget edit |
| Pause today | the driver honours `pause.flag` (`pause_reason: user`) and pauses itself on a failing audit (`pause_reason: audit`); nothing writes `pause.flag` except tests | #466 |
| Tokens | `src/services/labeling/vlm_client.py` posts `/chat/completions` and never reads `usage` | #466 |
| Daily VLM cap | `src/services/curation/vlm_scope.py::daily_budget_remaining` counts items with a VLM attempt today in OpenSearch (`max_crops_per_day`) | per project, all callers, crops only |
| `autolabel` job | `src/services/curation/autolabel/` + `scripts/curation/auto_label_worker.py`: embed, IVF retrain, VLM sweep of unvalidated items (`autolabel/selection.py`); cancel at stage boundaries, no pause | lane 4D 4D-1 ports it |
| Active-learning pieces | probe inference with real class entropy and margin (`src/services/curation/probe_predictions.py`), the uncertainty review tab (`review_tabs.py`, `review_sorts.py` `uncertainty_entropy`), confident-learning mistakenness (`item_scores/mistakenness.py`), k-center greedy core-set selection (`selection/kcenter_greedy.py`), human accuracy audit of machine labels (`src/domain/audit_math.py`, `/curation/projects/{project}/audit/queue`, `/curation/projects/{project}/audit/report`) | built, but separate from the loop |
| Lock rule | `src/clients/occ_locks.py` (`is_locked_class`: human marker, validated import, `test_holdout`; `is_locked_box`; `is_locked_item`), `class_write_guard.py` (`class_write_locked` adds any `class_validated`) | the rule every automated writer consults |
| Label operations | `src/services/curation/label_ops/` (op record, members with prior state, undo that restores only items still as left, never the holdout) | CQ3; moves to Postgres in lane 2D |
| Frontend | `frontend/src/lib/components/assistLoop/` (panel, start form, run view, nodes, rollback) | polls state; no pause, no event log |

### 2.2 Measurements on record

| Number | Value | Source |
|---|---|---|
| Live loop, public COCO val2017, 2,500 images, 15,712 crops, 512 clusters, 148 roots of 40 or more, default local VLM on GPU 0 | budget 1,500 crops: 1,443 crops and 516 of 516 requests spent (stop `budget_requests`), **55 items propagated**, 1,069 labeled directly, 305 clusters left for review; budget never exceeded in 7 runs; rollback 138 of 138; same seed same samples on 8 of 8 nodes | #286 closing comment, 2026-11-02 |
| Yield at that size | about 1 VLM crop per labeled item; the 1:20 target needs passing nodes of 700 to 1,100 members at theta 0.9 | #286; [auto_label_loop_plan.md](../auto_label_loop_plan.md) 6 |
| VLM label throughput | 10.0 crops/s through the label-batch route (8 threads x 32 crops), 9.8 crops/s for the continuous worker, one A6000 shared with Triton | `docs/PERFORMANCE.md` ("Measured VLM labeling speed") |
| Tokens per 4-image label call, 81-class registry | 1,306 prompt + 529 completion = 1,835 (about 460 per crop), 9.4 s latency; the model reasons before its JSON | [vlm_serving_sizing.md](../../research/vlm_serving_sizing.md) |
| Cluster purity at the start | 8 % of crops outside their cluster's majority class (9.8 % at low resolution) | #341 baseline, cited in [auto_label_loop_plan.md](../auto_label_loop_plan.md) 1 |
| Detector-class roots (Z4, simulated, oracle labeler) | decisions per item: `key` 0.293, `key_guard` 0.377, IVF loop 0.834; verdict INCONCLUSIVE (open-set bar unreadable) | [cluster_config_z4.md](../../research/cluster_config_z4.md) |
| Hosted visual judge | pair kappa 0.821 on COCO crops, 0.611 on Open Images vehicles; local 4B VLM pair accuracy 0.51 | [visual_judge_e0.md](../../research/visual_judge_e0.md) |

**Never measured:** the loop's label rate under real concurrency (S0 of #286), its propagated
precision against a person, the review time per label in Cropwright, token spend per run, the cost
ratio on a project with large clusters, and whether propagated labels help or hurt a trained
detector.

### 2.3 Facts about the B2 plan that this design depends on (and three gaps found)

The B2 design ([02a](02a_postgres_schema.md), [02b](02b_jobs_locking_events.md))
gives every long job one `jobs` row, a fenced claim, a heartbeat thread that returns
`cancel_requested_at` and `pause_requested_at`, `paused` as a state for the assist loop only,
`checkpoint` jsonb, and an `events` table (24 h retention, payload at most 7,000 bytes) with a
Valkey wake-up and SSE resume. Three gaps matter here:

| Gap | Where | Consequence | Fix (section 5.2) |
|---|---|---|---|
| G-1 | the claim increments `attempt` and refuses at `attempt >= max_attempts` (3); resume from `paused` goes back to `queued` | a loop paused and resumed three times fails `attempts_exhausted` on the fourth resume | count only failed deliveries: `retry_count` |
| G-2 | `assist_loop` routes to queue `assist_loop`, consumer `-c 1`, soft limit 24 h, max queue wait 1 h | one project's long loop holds the only slot; a second project's resumed loop fails `queue_wait_exceeded` after 1 h; a 24 h segment also outlives the 6 h `visibility_timeout`, so the message is redelivered (harmless, the claim skips it, but noisy) | time slices: the handler yields at a node boundary after `slice_s` and the wrapper re-queues it |
| G-3 | [03d](03d_stage4_job_families.md) 4D-2 moves `budget.json` "into `jobs.params` and `checkpoint`" | a jsonb checkpoint cannot be an atomic counter that several in-flight requests reserve against; and `events` rows (24 h) cannot replace `events.jsonl`, which is the loop's durable log | the loop tables of section 5.1 |

---

## 3. Literature and industry review

### 3.1 Active learning and sample selection

| Source | What it says that matters here |
|---|---|
| Settles, *Active Learning Literature Survey* (2009), <https://burrsettles.com/pub/settles.activelearning.pdf>; Ren et al., *A Survey of Deep Active Learning* (ACM CSUR 2021), <https://arxiv.org/abs/2009.00236> | the two families: uncertainty sampling (least confidence, margin, entropy) and representativeness/diversity; batch mode needs both, because a batch of the most uncertain items is redundant |
| Sener and Savarese, *Active Learning for CNNs: A Core-Set Approach* (ICLR 2018), <https://arxiv.org/abs/1708.00489> | k-center greedy over embeddings as a batch selector; the repository already implements it (`selection/kcenter_greedy.py`) |
| Ash et al., *BADGE* (ICLR 2020), <https://arxiv.org/abs/1906.03671> | uncertainty and diversity in one step via gradient embeddings; needs per-sample gradients of the task model, which the product does not have for a VLM labeler |
| Citovsky et al., *Batch Active Learning at Scale* (Cluster-Margin, NeurIPS 2021), <https://arxiv.org/abs/2107.14263> | margin sampling with round-robin over a fixed clustering, run on millions of examples; the closest published match to a product that already holds a cluster partition |
| Haussmann et al., *Scalable Active Learning for Object Detection* (IV 2020), <https://arxiv.org/abs/2004.04699> | at production scale, ensembles and simple uncertainty scores beat random; data curation by the model saves labeling |
| Yuan et al., *MI-AOD* (CVPR 2021), <https://arxiv.org/abs/2104.02324>; Choi et al., *Probabilistic modeling for active learning in object detection* (ICCV 2021), <https://arxiv.org/abs/2103.16130>; Yang et al., *Plug and Play Active Learning for Object Detection* (CVPR 2024), <https://arxiv.org/abs/2211.11612> | detection-specific uncertainty (instance-level, aleatoric and epistemic); MI-AOD and Choi change the detector head; PPAL is plug-and-play (difficulty-calibrated uncertainty plus category-conditioned similarity) |
| Munjal et al., *Towards Robust and Reproducible Active Learning* (CVPR 2022), <https://arxiv.org/abs/2002.09564>; Lüth et al., *Navigating the Pitfalls of Active Learning Evaluation* (NeurIPS 2023), <https://arxiv.org/abs/2301.10625> | random sampling is a strong baseline under good training; many reported gains vanish with proper tuning and repeated seeds. Every strategy comparison here includes random and repeats seeds |
| Northcutt et al., *Confident Learning* (JAIR 2021), <https://arxiv.org/abs/1911.00068>; cleanlab, <https://github.com/cleanlab/cleanlab> (maintained, includes `cleanlab.object_detection`) | find likely label errors from predicted probabilities; the repository's mistakenness score cites it. cleanlab is the library to use for label-issue ranking rather than hand-rolling more |
| Geifman and El-Yaniv, *Selective Classification for Deep Neural Networks* (NeurIPS 2017), <https://arxiv.org/abs/1705.08500>; Angelopoulos et al., *Conformal Risk Control* (ICLR 2024), <https://arxiv.org/abs/2208.02814> | how to pick an auto-accept threshold with a bound on the error rate from a labeled calibration sample; the loop's hypergeometric gate is the per-node analogue |
| Xie et al., *Noisy Student* (CVPR 2020), <https://arxiv.org/abs/1911.04252>; Sohn et al., *STAC* (2020), <https://arxiv.org/abs/2005.04757> | training on pseudo-labels helps only with confidence filtering and a held-out human check; confirmation bias is the documented failure |

### 3.2 Industry labeling tools

| Tool | How its model-assisted loop works | Lesson for OpenProcessor |
|---|---|---|
| Label Studio ML backend, <https://github.com/HumanSignal/label-studio-ml-backend>; active-learning guide, <https://docs.humansignal.com/guide/active_learning> | the backend is a web server with `predict(tasks)` and `fit(event, data)`; webhooks (`ANNOTATION_CREATED`, `ANNOTATION_UPDATED`) trigger `fit`; predictions carry a `score`; task ordering by prediction score (uncertainty sampling) is an Enterprise feature, Community users sort by score in the Data Manager; per-annotation `lead_time` is recorded | the model is a pluggable predictor, training is event-driven and asynchronous, the queue order is the selection strategy, and annotation time is captured per item. Label Studio is the right tool for the AL2 review-time study (section 6.4), not for the product UI |
| CVAT automatic annotation, <https://docs.cvat.ai/docs/annotation/auto-annotation/automatic-annotation/> | detector or re-identifier models (Nuclio serverless functions, Hugging Face, Roboflow, or AI-agent functions) run over a whole task; model labels are **mapped to task labels by name**, unmapped labels are dropped; a confidence threshold; a progress bar and cancel at any time | name-based label mapping is the industry norm (matches the project rule); a whole-task run is cancellable; there is no pause; pre-annotations are then reviewed by people |
| Roboflow Autodistill, <https://github.com/autodistill/autodistill> | a base model (Grounded SAM, Grounding DINO, OWLv2) labels a folder through an ontology `{caption: class}`, then a target model (YOLO) trains on it | the teacher-student pattern of #84; the ontology is a prompt pack; no human review step and no precision check, which is exactly what this plan adds |
| SAM 3 data engine, Meta AI, *SAM 3: Segment Anything with Concepts* (2025), <https://arxiv.org/abs/2511.16719> | AI annotators propose, fine-tuned multimodal AI verifiers check masks and exhaustivity, people correct only what the verifiers flag; adding AI verifiers "more than double[s] the throughput compared to a human-only annotation pipeline" (vendor-reported) | the shape this loop should have: machine proposal, machine verification (the second-opinion audit), human effort only on what fails. The verifiers were trained on human labels first: a verifier is trusted only after its agreement with people is measured |

### 3.3 Open-vocabulary proposers

| Model | Source | Role |
|---|---|---|
| Grounding DINO | Liu et al. 2023, <https://arxiv.org/abs/2303.05499> | text-prompted boxes; strong recall, weak on fine-grained classes |
| OWLv2 | Minderer et al., *Scaling Open-Vocabulary Object Detection* (NeurIPS 2023), <https://arxiv.org/abs/2306.09683> | text- or image-prompted boxes; its own self-training is a pseudo-label loop at web scale |
| SAM 3 | <https://arxiv.org/abs/2511.16719> | concept-prompted detection and segmentation; already the segmenter service of the region stage |

Boxes are the region stage's job (region profiles, the segmenter, VLM verification). The loop's
first target is the **class of an item**; box targets are J12.

### 3.4 VLM labeling practice

| Source | Finding |
|---|---|
| Gilardi et al., *ChatGPT outperforms crowd workers for text-annotation tasks* (PNAS 2023), <https://arxiv.org/abs/2303.15056> | LLM labels can match crowd labels on objective tasks; agreement must be measured per task |
| *Can Vision-Language Models Replace Human Annotators: A Case Study with CelebA Dataset* (2024), <https://arxiv.org/abs/2410.09416> | LLaVA-NeXT agreed with human labels on 79.5 % of 1,000 images and 40 attributes, 89.1 % after re-annotating disagreements; better on objective attributes; 0.49 s per label on an A6000 |
| this repository | the default local VLM reasons before answering (529 completion tokens per 4-image call), so `max_tokens` and the completion share dominate token cost; a hosted judge was reliable at make level only ([visual_judge_e0.md](../../research/visual_judge_e0.md)) |

Practice that follows: a VLM label is a **suggestion with a measured precision per class**, never
ground truth; agreement between two independent labelers (VLM and detector, or two VLMs) is a
cheap triage signal; the precision claim comes from a blind human sample.

### 3.5 Human-in-the-loop review

| Source | Finding |
|---|---|
| Papadopoulos et al., *We don't need no bounding-boxes* (CVPR 2016), <https://arxiv.org/abs/1602.08405>; *Extreme clicking* (ICCV 2017), <https://arxiv.org/abs/1708.02750> | verifying a proposal is far cheaper than drawing (seconds per verification against about 7 s per extreme-click box and about 35 s for a classic box, as reported); design for yes/no verification |
| Amershi et al., *Power to the People: The Role of Humans in Interactive Machine Learning* (AI Magazine 2014), <https://doi.org/10.1609/aimag.v35i4.2513> | people want rapid, incremental feedback, to steer the model, and to see why it decided; long opaque batch runs frustrate them |
| Parasuraman and Manzey, *Complacency and Bias in Human Use of Automation* (Human Factors 2010), <https://doi.org/10.1177/0018720810376055> | automation bias: people accept wrong suggestions they are shown. A precision audit that shows the proposal measures agreement with the machine, not accuracy; the audit must be blind |
| Monarch, *Human-in-the-Loop Machine Learning* (Manning 2021) | sample a random slice of every review queue for unbiased evaluation; keep the evaluation set out of training |

### 3.6 Long-running, controllable jobs on Celery

| Source | What it says |
|---|---|
| Celery tasks guide, "Granularity", <https://docs.celeryq.dev/en/stable/userguide/tasks.html#granularity> | "In general it is better to split the problem up into many small tasks rather than have a few long running tasks." Also: route long-running tasks to dedicated workers; tasks should be idempotent |
| Celery workers guide, "Revoking tasks", <https://docs.celeryq.dev/en/stable/userguide/workers.html#revoking-tasks> | `terminate=True` kills the process and "you must never call this programmatically"; revokes are held in worker memory. Celery has no pause primitive: pause must be cooperative |
| Celery `acks_late` (same tasks guide) and the Redis transport `visibility_timeout` | a late-acked message not acknowledged within the visibility timeout is redelivered; 02b sets 6 h |
| OpenAI-compatible `usage` object, <https://platform.openai.com/docs/api-reference/chat/object>; vLLM OpenAI-compatible server, <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html> | each non-streamed chat completion returns `usage.prompt_tokens` and `usage.completion_tokens`; a streamed one only with `stream_options: {"include_usage": true}` |

### 3.7 Libraries

| Need | Library | Decision |
|---|---|---|
| AHC and k-means splits | scikit-learn (already used by `refine_members`, `splitter.py`) | keep |
| Uncertainty, diversity, margin scores | NumPy; scikit-learn `LogisticRegression` for a linear probe on stored vectors | a few lines each; no active-learning framework |
| Active-learning frameworks (modAL, <https://github.com/modAL-python/modAL>; Baal, <https://github.com/baal-org/baal>) | both are classification-oriented, own their training loop and pool in memory | ❌ not adopted: our pool lives in OpenSearch, our labeler is a VLM, our trainer is Ultralytics; adopting one would mean adapting our data to it for about 20 lines of scoring each |
| Label-issue ranking | cleanlab | 🟡 research (AL3); product only after it beats the existing mistakenness score |
| Wilson intervals, bootstrap | statsmodels `proportion_confint(method='wilson')`, SciPy `bootstrap` (research); `src/domain/loop_stats.wilson_lower_bound` (product, stdlib) | keep both |
| Review-time study | Label Studio (Apache-2.0), local in the research venv | research only (AL2) |
| VLM serving | vLLM (`vllm serve`, OpenAI-compatible) | already the `vlm` service; the research venv runs it directly |

---

## 4. Candidate designs

### 4.1 Orchestration

| Design | Verdict | Reason |
|---|---|---|
| One `jobs` row per loop run; the handler runs in **time slices** (yield at a node boundary after `slice_s`, re-queue), the durable loop state in its own tables, pause and cancel cooperative through the job row | ✅ | Celery granularity guidance; fairness on the one `assist_loop` consumer (G-2); every slice re-reads state, so a crash loses at most the in-flight requests |
| One Celery task per node, joined with a chord | ❌ | thousands of tasks per run, a chord needs a result backend (D-49 has none), state split between broker and database |
| A `jobs` row per iteration | 🟡 | cleaner once retraining enters the loop (section 7.4); not needed for the B2 port |
| Pause by `revoke` or by stopping the worker | ❌ | revoke is for queued messages; `terminate=True` is forbidden by the Celery docs and would lose in-flight answers and reservations |
| Keep `events.jsonl` and `budget.json` on a shared volume | ❌ | B2 deletes file protocols (D-41); multi-host (D-46) |

### 4.2 Budgets

| Design | Verdict | Reason |
|---|---|---|
| Postgres rows per `(run, pool, unit)` with one conditional `UPDATE ... RETURNING` per reservation, plus an append-only ledger keyed by the upstream request id | ✅ | atomic across processes; survives a crash; the ledger makes "every unsettled reservation counts as spent" a query; fenced by the job's token |
| The in-process `threading.Lock` of `budget.py` | ❌ | correct only for one process; the rebuild has no such guarantee during a lease handover |
| Valkey `INCRBY` counters | ❌ | a second store for control data; up to about 1 s of loss on crash (D-24) could over-spend |
| Tokens estimated locally with a tokenizer | ❌ | the endpoint knows; read `usage`; estimate only for the reservation |

### 4.3 Selection strategies

| Strategy | Verdict | Evidence |
|---|---|---|
| `cluster_sampled` (the #286 tree: sample, exact gate, propagate or split) | ✅ keep as the default | built and tested; precision unproven against people; yield poor on small clusters |
| `random` | ✅ as a measurement baseline and as the blind audit's draw | Munjal 2022, Lüth 2023 |
| `cluster_margin` (Citovsky 2021): lowest-margin items from a probe, round-robin over existing clusters | 🟡 prototype in AL3 | published at million scale; uses the partition we already have |
| `uncertainty` (probe entropy or margin) | 🟡 AL3 | needs a probe; the existing probe needs a trained detector, a linear probe on stored vectors does not |
| `diversity` (k-center core-set) | 🟡 AL3 | exists as a selection job; core-set is known to degrade in high dimension |
| `detector_class_roots` with the hold-back guard | 🟡 gated on Z5 (section 6.8) | Z4: 2.2x to 2.85x fewer decisions per item than IVF, open-set precision unread |
| BADGE, MI-AOD, learned selection policies | ❌ | need detector-head changes or per-sample gradients; no measured need |

### 4.4 Proposers

| Proposer | Verdict | Note |
|---|---|---|
| VLM classify through the active endpoint and prompt pack | ✅ | as today |
| Second VLM as audit (second opinion) | ✅ | as today; external endpoints only with the three-part consent (`OP_VLM_EXTERNAL_POLICY`, endpoint `allow_external`, per-run acknowledgement) |
| Detector class (`detector_class_name`) and a linear probe on stored vectors as cheap proposers; VLM only where they disagree | 🟡 AL4 | agreement triage; the probe trains in seconds on CPU |
| Zero-shot class prior from the text tower of the embedding model | 🟡 AL4 | nearly free; precision unknown |
| Open-vocabulary box proposers (OWLv2, Grounding DINO, SAM 3) | 🟡 region stage, J12 | boxes are the region stage's target |
| A hosted VLM as the default labeler | ❌ | privacy; only with explicit consent |

### 4.5 Writes, acceptance and review

| Design | Verdict | Reason |
|---|---|---|
| Propagated labels are a machine kind (`class_source: assist_propagated`, `class_validated: false`), excluded from the training export until a person accepts them | ✅ | #286 Q1/Q6 decision, kept |
| Machine labels marked validated after a gate | ❌ | would lock machine decisions against every later fix and impersonate a person |
| Calibrated per-class auto-accept thresholds from a human sample (selective classification) | 🟡 after AL2 | needs a human-verified calibration set per class |
| Propagation and rollback as label operations | ✅ | one undo mechanism (CQ3), not two |
| Review in verify mode: a grid of items with the same proposed class, accept all or correct a few, keyboard-driven | ✅ | verification is cheap (Papadopoulos 2016); keymaps exist |
| A blind audit queue (no proposal shown) for the precision number | ✅ | automation bias (Parasuraman and Manzey 2010) |
| Embedding Label Studio or CVAT as the product's review UI | ❌ | Cropwright is the frontend (vision); research use only |

### 4.6 Events

| Design | Verdict |
|---|---|
| `events` rows plus the Valkey wake-up, as designed in 02b section 6 | ✅ |
| A loop-specific stream or Valkey Streams | ❌ (02b section 6.1) |

---

## 5. Seams the v0.7.0 rebuild must include

Every item names its owning plan file and lane. "B2 creates" means the B2 lane writes it; "later"
means the value or column arrives with the feature that uses it, never as an unused value (no dead
code).

### 5.1 Loop tables in PostgreSQL

Owner: [02a_postgres_schema.md](02a_postgres_schema.md), new section 5.1 "Auto-label loop
tables"; written by lane **1A** in `0001_control_plane` (the schema is not built yet, so a second
revision is not needed); used by lane **4D**. RLS: `assist_loop_runs` carries `project_id`; every
other loop table reaches the project through `run_id` with the `EXISTS` policy of 02a section 5.
Item ids are OpenSearch document ids (`text`); class identity is the normalized name, never an id.

```sql
CREATE TABLE assist_loop_runs (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),     -- the loop_id on the wire
  project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id             uuid NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE RESTRICT,  -- status, pause, cancel live on the job
  preset_config_id   uuid NULL REFERENCES config_documents(id) ON DELETE SET NULL,
  preset_revision    integer NULL,
  request            jsonb NOT NULL,                        -- normalized AssistLoopRequest, validated on write and read
  strategy           text NOT NULL CHECK (strategy IN ('cluster_sampled')),   -- later: detector_class_roots, cluster_margin, uncertainty, diversity, sweep
  seed               bigint NOT NULL,
  pinned             jsonb NOT NULL,                        -- {vlm:{config_id,revision,model}, prompt_pack:{config_id,revision}, audit_vlm, partition:{kind,digest,snapshot_at}}
  stop_reason        text NULL CHECK (stop_reason IN ('done','budget_crops','budget_requests','budget_tokens','budget_time','vlm_failed')),
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  created_by_user_id uuid NULL,
  CHECK ((preset_config_id IS NULL) = (preset_revision IS NULL))
);
CREATE INDEX ix_assist_loop_runs_project ON assist_loop_runs (project_id, created_at DESC);

CREATE TABLE assist_loop_iterations (
  run_id             uuid NOT NULL REFERENCES assist_loop_runs(id) ON DELETE CASCADE,
  ordinal            integer NOT NULL CHECK (ordinal >= 1),
  status             text NOT NULL CHECK (status IN ('open','closed')),          -- later: reviewing, training
  counts             jsonb NOT NULL DEFAULT '{}',          -- the per-level rows the wire serves as `levels`
  started_at         timestamptz NOT NULL DEFAULT now(),
  closed_at          timestamptz NULL,
  PRIMARY KEY (run_id, ordinal),
  CHECK ((status = 'closed') = (closed_at IS NOT NULL))
);

CREATE TABLE assist_loop_nodes (
  run_id             uuid NOT NULL REFERENCES assist_loop_runs(id) ON DELETE CASCADE,
  path               text NOT NULL CHECK (path ~ '^r[0-9]+(\.[0-9]+)*$'),       -- grammar of src/domain/loop_tree.py; Z5-I widens it
  iteration          integer NOT NULL,
  parent_path        text NULL,
  level              smallint NOT NULL CHECK (level >= 0),
  member_count       integer NOT NULL CHECK (member_count >= 0),
  eligible_count     integer NOT NULL CHECK (eligible_count >= 0),
  membership_digest  text NOT NULL,
  snapshot_storage_key text NOT NULL,                      -- sorted member ids + class_state_token per id (storage seam, 02c section 5)
  step               text NOT NULL CHECK (step IN ('open','admitted','asked','gated','propagating','done')),
  decision           text NULL CHECK (decision IN ('pass','split','review_leaf','review_small','stale','unvisited')),
  class_name         text NULL,
  answered           integer NULL,
  agree              integer NULL,
  bound              double precision NULL,
  distribution       jsonb NULL,                           -- [{class_name, votes}] plus new-class proposals
  admitted_crops     integer NULL,                         -- the node's budget claim (the old `admit` event)
  admitted_requests  integer NULL,
  label_op_id        uuid NULL REFERENCES label_operations(id) ON DELETE SET NULL,   -- the propagation (section 5.6)
  items_propagated   integer NOT NULL DEFAULT 0,
  skipped            jsonb NOT NULL DEFAULT '{}',          -- {locked, holdout, excluded, drifted, dissent, token_changed}
  updated_at         timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, path),
  FOREIGN KEY (run_id, iteration) REFERENCES assist_loop_iterations(run_id, ordinal) ON DELETE CASCADE,
  CHECK ((step = 'done') = (decision IS NOT NULL))
);
CREATE INDEX ix_assist_loop_nodes_open ON assist_loop_nodes (run_id) WHERE step <> 'done';
CREATE INDEX ix_assist_loop_nodes_decision ON assist_loop_nodes (run_id, decision, level);

CREATE TABLE assist_loop_selections (       -- what was chosen to be asked, and why
  run_id             uuid NOT NULL REFERENCES assist_loop_runs(id) ON DELETE CASCADE,
  item_id            text NOT NULL,
  purpose            text NOT NULL CHECK (purpose IN ('node_draw','node_topup','redraw','audit','blind_audit')),
  node_path          text NULL,
  iteration          integer NOT NULL,
  score              double precision NULL,                -- selector score where one exists (later strategies)
  draw_rank          integer NOT NULL,                     -- order within the seeded draw: reproducibility
  selected_at        timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, item_id, purpose)
);

CREATE TABLE assist_loop_proposals (        -- what a labeler answered for one item
  run_id             uuid NOT NULL REFERENCES assist_loop_runs(id) ON DELETE CASCADE,
  item_id            text NOT NULL,
  role               text NOT NULL CHECK (role IN ('primary','audit')),
  proposer_kind      text NOT NULL CHECK (proposer_kind IN ('vlm')),           -- later: detector, linear_probe, zero_shot
  proposer_ref       text NOT NULL,                        -- '<config_id>:<revision>' of the endpoint, or a model id later
  request_id         uuid NOT NULL,                        -- the upstream call; joins the budget ledger
  outcome            text NOT NULL CHECK (outcome IN ('answered','unreadable','failed')),
  class_name         text NULL,                            -- normalized registry name
  new_class_name     text NULL,                            -- a proposed name not in the registry (never created by the loop)
  attributes         jsonb NULL,                           -- pack-declared attributes, when asked
  item_write         text NULL CHECK (item_write IN ('written','locked','token_changed','not_attempted')),
  created_at         timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, item_id, role),
  CHECK ((outcome = 'answered') = (class_name IS NOT NULL OR new_class_name IS NOT NULL))
);
CREATE INDEX ix_assist_loop_proposals_request ON assist_loop_proposals (run_id, request_id);

CREATE TABLE assist_loop_reviews (          -- a person's decision on a machine label or proposal
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  run_id             uuid NOT NULL REFERENCES assist_loop_runs(id) ON DELETE CASCADE,
  item_id            text NOT NULL,
  queue              text NOT NULL CHECK (queue IN ('verify','blind_audit')),
  machine_source     text NOT NULL CHECK (machine_source IN ('propagated','direct','proposal_only')),
  machine_class_name text NULL,                            -- what the loop wrote or proposed
  shown_class_name   text NULL,                            -- what the UI showed; NULL in blind_audit by CHECK
  decision           text NOT NULL CHECK (decision IN ('accept','correct','reject','skip')),
  human_class_name   text NULL,
  served_at          timestamptz NOT NULL,                 -- server time the item was handed to this reviewer
  decided_at         timestamptz NOT NULL DEFAULT now(),
  active_ms          integer NULL CHECK (active_ms BETWEEN 0 AND 600000),     -- client-measured focus time
  batch_size         smallint NOT NULL DEFAULT 1 CHECK (batch_size BETWEEN 1 AND 200),  -- items decided by one grid action
  label_op_id        uuid NULL REFERENCES label_operations(id) ON DELETE SET NULL,
  reviewer           text NOT NULL,
  reviewer_user_id   uuid NULL,
  UNIQUE (run_id, item_id, queue),
  CHECK (queue <> 'blind_audit' OR shown_class_name IS NULL),
  CHECK ((decision IN ('accept','correct')) = (human_class_name IS NOT NULL))
);
-- B2 creates this table with the columns above; its routes arrive in v0.8.0 (section 7.3). It is
-- in 0001 so the review data model is fixed before the wire, not retrofitted.

CREATE TABLE assist_loop_budgets (
  run_id             uuid NOT NULL REFERENCES assist_loop_runs(id) ON DELETE CASCADE,
  pool               text NOT NULL CHECK (pool IN ('vlm','audit')),
  unit               text NOT NULL CHECK (unit IN ('crops','requests','prompt_tokens','completion_tokens','wall_s')),
  limit_value        bigint NULL CHECK (limit_value IS NULL OR limit_value >= 0),  -- NULL = uncapped; never NULL for (vlm, crops)
  reserved           bigint NOT NULL DEFAULT 0 CHECK (reserved >= 0),
  spent              bigint NOT NULL DEFAULT 0 CHECK (spent >= 0),
  updated_at         timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, pool, unit),
  CHECK (unit = 'prompt_tokens' OR limit_value IS NULL OR spent + reserved <= limit_value)
);

CREATE TABLE assist_loop_budget_ledger (
  id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id             uuid NOT NULL REFERENCES assist_loop_runs(id) ON DELETE CASCADE,
  request_id         uuid NOT NULL,                        -- one reservation = one upstream call
  pool               text NOT NULL,
  unit               text NOT NULL,
  entry              text NOT NULL CHECK (entry IN ('reserve','settle','release','forfeit')),
  amount             bigint NOT NULL,
  fencing_token      bigint NOT NULL,                      -- the job attempt that wrote it
  at                 timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (run_id, pool, unit) REFERENCES assist_loop_budgets(run_id, pool, unit) ON DELETE CASCADE
);
CREATE INDEX ix_assist_loop_ledger_request ON assist_loop_budget_ledger (run_id, request_id);

CREATE TABLE vlm_usage_hourly (             -- every VLM caller, not only the loop
  project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  endpoint_config_id uuid NOT NULL REFERENCES config_documents(id) ON DELETE CASCADE,
  source             text NOT NULL CHECK (source IN ('assist_loop','assist_run','autolabel','vlm_worker','region_verify','label_batch')),
  hour               timestamptz NOT NULL CHECK (hour = date_trunc('hour', hour)),
  requests           bigint NOT NULL DEFAULT 0,
  crops              bigint NOT NULL DEFAULT 0,
  prompt_tokens      bigint NOT NULL DEFAULT 0,
  completion_tokens  bigint NOT NULL DEFAULT 0,
  usage_missing      bigint NOT NULL DEFAULT 0,            -- answered calls whose response had no usage object
  PRIMARY KEY (project_id, endpoint_config_id, source, hour)
);
```

Why the members snapshot is a blob and not rows: a root can hold 100,000 members (section 6.8,
key roots); one blob per node through the storage seam keeps the table small and matches the
holdout snapshot pattern (`holdout_freezes.snapshot_storage_key`).

Retention ([02a](02a_postgres_schema.md) section 9): `jobs` rows referenced by
`assist_loop_runs` are kept (add the kind to the exception list); a run and its children are kept
while any of its `label_operations` can still be undone; `assist_loop_proposals` and
`assist_loop_selections` of runs finished more than 90 days ago are deleted in batches by the reaper
(the node rows keep the counts); `vlm_usage_hourly` keeps 400 days. J10 confirms.

### 5.2 Job states, columns and transitions

Owner: [02b](02b_jobs_locking_events.md) sections 1, 2.1, 3.1, 3.3 and 4; DDL in
[02a](02a_postgres_schema.md) section 5; lane **1A** (columns, the transition trigger
generated from `src/domain/job_states.py`), lane **3A** (repos, wrapper, reaper).

New `jobs` columns:

| Column | Type | Meaning |
|---|---|---|
| `retry_count` | `integer NOT NULL DEFAULT 0` | incremented only when an attempt ends `interrupted`, `lost` or by redelivery after a crash; the claim checks `retry_count < max_attempts` instead of `attempt < max_attempts` (fixes G-1; `attempt` still numbers `job_attempts` rows) |
| `pause_reason` | `text NULL CHECK (pause_reason IN ('user','audit','budget','project_paused','vlm_unavailable'))` | set with `pause_requested_at` by the route or by the handler; cleared on resume; `CHECK ((status = 'paused') <= (pause_reason IS NOT NULL))` |

New and changed transitions (the table in 02b section 1):

| From | To | Who | Condition |
|---|---|---|---|
| `running` | `queued` | wrapper, fenced | **new**: the handler returned `Yield` at a boundary after `slice_s` (kinds with `slice_s` only); attempt outcome `yielded`; no `retry_count` change; dispatched again at once |
| `queued` | `paused` | pause route | **new**: a queued job can be paused before it runs; **no revoke** (a revoked id stays in worker memory and would skip the resume; the stale message is acked by the claim; program 02b, review RR-1) |
| `running` | `paused` | wrapper, fenced | as today, widened: `pause_requested_at` set by the route or by the handler itself (audit, budget, project paused, VLM unavailable); for every kind whose `pausable` is true |
| `paused` | `queued` | resume route; **reaper** for system reasons | the reaper resumes `project_paused` when `projects.paused_at IS NULL` and `vlm_unavailable` when the VLM is free again (`deployment_settings.vlm_paused_for_training` cleared and the endpoint probe answers); `user`, `audit` and `budget` wait for a person |

`job_attempts.outcome` gains `yielded`. Cancelled stays terminal for every kind: today's loop can be
resumed after cancel; in the rebuild "stop for now" is pause, and cancel means done (one meaning
across all job families). This is a deliberate behaviour change (J4).

Kind table changes ([02b](02b_jobs_locking_events.md) section 2.1, data in
`src/config/job_kinds/trigger.py`, lane **4D**):

| Kind | Field | Value |
|---|---|---|
| `assist_loop` | `pausable` | true (boundary: a node step; in-flight requests finish) |
| `assist_loop` | `slice_s` | 600 |
| `assist_loop` | max queue wait | 24 h (was 1 h; a yielded slice waits behind other projects' slices) |
| `assist_loop` | soft / hard limit | 30 m / 35 m (a slice, not a run; was 24 h) |
| `autolabel` | `pausable` | true (boundary: between stages and between VLM chunks of the sweep), J3 |
| every other kind | `pausable` | false; `slice_s` none |

A test asserts `pausable` kinds and only those accept a pause request (409 `job_not_pausable`
otherwise), and that every kind with `slice_s` has a soft limit above `slice_s` plus the longest
boundary wait (one VLM request timeout).

### 5.3 The control API and its Celery semantics

Owner: [03d](03d_stage4_job_families.md) 4D-2 (routes and service), contracts regenerated in
the same commits; lane **4D**. All routes are project-scoped, go through the one serializer, accept
the optional `Idempotency-Key` on start (D-58), and are added to the cross-project leak tables.

| Action | Route | State effect | Celery |
|---|---|---|---|
| Plan | `POST /curation/projects/{project}/assist/loops` with `dry_run: true` (exists) | none; returns the plan | none |
| Start | same route, `dry_run: false` (exists) | one transaction: `jobs` row `queued` (via `start_job`), `assist_loop_runs`, `assist_loop_budgets` rows, iteration 1; 202; 409 `assist_loop_busy` from `uq_jobs_one_active` | `send_task` after commit (02b section 2.2) |
| Pause | `{P}/assist/loops/{loop_id}/pause` (new, POST) | `queued` -> `paused` at once; `running` -> `pause_requested_at = now()`, `pause_reason = 'user'`, answer 202 with `control.state = 'pausing'`; `paused` -> 200 no-op; `cancelling` -> 409 `job_cancelling`; terminal -> 409 `job_finished` | none: pause is cooperative; a queued job's stale message is acked by the claim (no `revoke`, D-71) |
| Resume | `POST /curation/projects/{project}/assist/loops/{loop_id}/resume` (exists) | `paused` or `interrupted` -> `queued`, `pause_reason = NULL`; 409 `resume_blocked` with the reason when the project is paused or the VLM is unavailable | dispatch |
| Cancel | `POST /curation/projects/{project}/assist/loops/{loop_id}/cancel` (exists) | 02b section 3.3 exactly | no `revoke` (D-71): cancel is a database write and the claim acks the stale message; never terminate |
| Edit budget | `{P}/assist/loops/{loop_id}/budget` (new, PATCH) | body `{limits: [{pool, unit, limit_value}]}`; allowed in `queued`, `running`, `paused`; a limit below `spent + reserved` answers 409 `budget_below_spent`; emits `assist_loop.budget`; never resumes by itself | none |
| Rollback | `POST /curation/projects/{project}/assist/loops/{loop_id}/rollback` (exists) | becomes a label-operation undo over the run's propagation ops (section 5.6); dry run first | the undo kind of 02b (`label_undo`) |

Handler semantics (in `src/services/curation/assist_loop/driver.py`, run by `op.assist_loop.run`):

1. **Checkpoints are the node steps.** Each step (admit, draw, ask chunk, gate, propagate) commits
   its rows before the next starts. A resumed or next slice rebuilds the open-node queue from
   `assist_loop_nodes WHERE step <> 'done'` and the answers from `assist_loop_proposals`; no file
   replay.
2. **Pause latency.** Before every budget reservation and before admitting a node, the handler
   reads `ctx.paused()` (set by the heartbeat within 15 s, or by the throttled progress write within
   about 1 s). On pause it stops reserving; in-flight requests finish (bounded by the VLM request
   timeout), their answers and settlements are written, open nodes stay open; the wrapper writes
   `paused`. Target: `paused` within one request timeout plus 1 s of the request (measured in the
   4D live test).
3. **Cancel** stops the same way but cancels in-flight requests (`asyncio` task cancel); their
   reservations are forfeited (counted spent).
4. **Slices.** After `slice_s`, at the next node boundary, the handler returns `Yield`; the wrapper
   writes `queued` and dispatches. Other projects' slices on the same queue interleave in FIFO order.
5. **System pauses.** A failing audit batch, a budget that reaches a limit with `on_budget: pause`
   (J5), the project being paused (`projects.paused_at`), or the VLM being taken for training make
   the handler request its own pause with that reason.
6. **One owner.** The exclusive group `assist_loop` and the fenced claim guarantee one running
   slice per project; `may_write()` guards OpenSearch writes; every loop-table write carries the
   fencing token check through `src/db/repos/assist_loop.py`.

### 5.4 Event types

Owner: [02b](02b_jobs_locking_events.md) section 6.2 (producer list) and lane **3B**
(the feed); the constants in a new `src/domain/assist_loop_events.py` (pure), lane **4D**. Each
payload is a snapshot (no deltas), carries `loop_id`, and stays under the 7,000-byte cap. The loop
tables, not these rows, are the record.

| Type | When | Payload |
|---|---|---|
| `job.state` (existing) | every state change | adds `pause_reason`, `stop_reason`, `control` (section 5.8) |
| `job.progress` (existing) | throttled 1 per second | `{done: crops_spent, total: max_vlm_crops, stage: 'iteration:<n>', counters: {nodes_open, nodes_done, items_propagated, items_direct, left_for_review}}` |
| `assist_loop.nodes` | at most 1 per second, batching up to 40 decided nodes | `{loop_id, nodes: [{path, level, decision, class_name, answered, agree, bound, eligible, items_propagated}]}` |
| `assist_loop.budget` | when any pool crosses 50, 80, 95 or 100 % of a limit, and after a budget edit | `{loop_id, budgets: [{pool, unit, limit_value, reserved, spent}]}` |
| `assist_loop.audit` | after each audit batch | `{loop_id, sampled, agreed, lower_bound, floor, status}` |
| `assist_loop.iteration` | iteration opened or closed | `{loop_id, ordinal, status, counts}` |
| `assist_loop.warning` | a condition a person should see, at most once per minute per code | `{loop_id, code: 'vlm_error_rate' | 'usage_missing' | 'stale_nodes' | 'drift', detail}` |
| `assist_loop.review_queue` (v0.8.0, with the review routes) | when the queue sizes change, throttled | `{loop_id, pending_verify, pending_blind_audit}` |

Cropwright subscribes through `GET /curation/projects/{project}/events` (the SSE route of 02b
section 6.3) and filters by `loop_id`; it re-reads `GET /curation/projects/{project}/assist/loops/{loop_id}`
on a `reset` event.

### 5.5 Budget accounting

Owner: lane **4D** (`assist_loop/budget.py` rewritten on `src/db/repos/assist_loop.py`), plus
`src/services/labeling/vlm_client.py` for `usage` capture (4D's file list gains it; J2).

- **Reserve** (one statement per unit, all units of one call in one transaction):
  `UPDATE assist_loop_budgets SET reserved = reserved + :n, updated_at = now() WHERE run_id = :r AND
  pool = :p AND unit = :u AND (limit_value IS NULL OR spent + reserved + :n <= limit_value)
  RETURNING reserved, spent, limit_value`; zero rows refuses the call (`budget_exhausted`), and the
  ledger gets `reserve` rows with the job's fencing token. The units of one call are crops (images
  in the call), requests (1), completion tokens (`max_tokens` of the call: an exact upper bound) and
  prompt tokens (an estimate: the last 20 calls' mean per image times images plus the prompt
  length, plus 20 %).
- **Settle** after the response: move the reservation to `spent` with the actual amounts from
  `usage` (completion never exceeds the reservation; prompt may, by the estimate error, which is why
  the `prompt_tokens` row is exempt from the CHECK and the overshoot is reported). No `usage` in the
  response: settle at the reserved amount and count `usage_missing` (fail closed: the budget is
  charged the upper bound).
- **Forfeit** on resume or slice start: every `reserve` without `settle` or `release` for the run
  is moved to spent (`forfeit` rows); a crash wastes at most `vlm_concurrency x images_per_call`
  crops, never exceeds a limit.
- **Wall time** is charged per slice from `started_at` to the yield or stop.
- **Usage everywhere**: one function `record_vlm_usage(session, project_id, endpoint, source,
  crops, usage)` upserts `vlm_usage_hourly` for every VLM call site (the loop, assist runs, the
  `autolabel` sweep, the VLM worker, region verification, the label-batch route). The per-project
  daily cap (`max_crops_per_day`) then reads this table instead of counting OpenSearch documents
  (v0.8.0, J9).
- **Cost** for remote endpoints is not stored: the wire computes it from tokens and the endpoint's
  configured price per million tokens if the endpoint document carries one (J9).

### 5.6 Lock-rule integration

Owner: unchanged module `src/clients/occ_locks.py`; tests in lane **4D**; label-operation kinds in
lane **2D** ([03b](03b_stage2_control_data.md)).

| Point | Rule |
|---|---|
| Selection | eligible = not `is_locked_class`, not `class_validated`, not `test_holdout`, not excluded, at snapshot time (query level, cheap) |
| Per-item VLM write | through `ClassWriteGuard` and the OCC skip-on-conflict bulk; re-checked per document at write time; the result is stored in `assist_loop_proposals.item_write` |
| Propagation | one `label_operations` row per propagated node (`kind = 'assist_loop_propagate'`), members with their prior class state; the merge returns nothing for a locked or token-changed document; `assist_loop_nodes.label_op_id` links it |
| Rollback | the label-op undo over the run's ops: restores an item only while it is exactly as the loop left it, never touches the holdout (CQ3 sections 5-6); replaces `assist_loop/rollback.py` |
| Review writes (v0.8.0) | an accept or a correction is a human label written as a label operation (`kind = 'assist_loop_review'`), so it is locked afterwards and undoable |
| Holdout | holdout items are never selected, never sent to any endpoint (no test-set contamination, no external exposure) and never in a loop review queue |
| Export | `assist_propagated` items stay out of the training export until a person accepts them (export readiness reads `class_validated`); J6 asks about an opt-in |
| Fencing | `may_write()` before every OpenSearch write; a slice that lost its lease writes nothing |

The per-class lock-rule test of #286 (`tests/curation/test_assist_loop_lock_rule.py`) is carried
over unchanged in intent: one case per locked kind, each never written, through the new path.

### 5.7 Config kinds

Owner: [02a](02a_postgres_schema.md) section 4; lane **2C**
([03b](03b_stage2_control_data.md)).

| Change | Value |
|---|---|
| `config_documents.kind` | add `assist_loop_preset` (project-scoped, revisioned) |
| `config_activations.axis` | add `assist_loop_preset` with the pair `('assist_loop_preset','assist_loop_preset')`: the project's default preset |
| Body model | `AssistLoopPresetBody` in `src/schemas/curation/assist_loop.py` (`extra='forbid'`): `strategy`, `gate` (theta, alpha, max_dissent), `recursion` (max_depth, branching, min_node_size), `budget` defaults (max_vlm_crops required, requests, tokens, wall), `on_budget` (`'complete' | 'pause'`), `audit` (endpoint name, rate, min_per_batch, max_crops, floor, on_fail), `vlm_concurrency`, `node_concurrency` |
| Run pinning | a run stores `preset_config_id` and `preset_revision`; a request field overrides a preset field; the normalized request is stored whole |

Prompt packs and VLM endpoints stay their existing kinds.

### 5.8 The UI-facing wire

Owner: `src/schemas/curation/assist_loop.py` and `src/services/curation/wire.py`, lane **4D**;
Cropwright types in lane **4E** (and the v0.8.0 frontend lane). The backend decides, the frontend
renders (no client-side state machine).

| Model | Fields |
|---|---|
| `AssistLoopControl` (new) | `state: Literal['queued','running','pausing','paused','cancelling','completed','completed_with_errors','failed','cancelled','interrupted']` (`pausing` = running with a pause requested); `can_pause`, `can_resume`, `can_cancel`, `can_edit_budget`, `can_rollback: bool`; `resume_blocked_reason: Literal['project_paused','vlm_unavailable'] | None` |
| `AssistLoopState` (existing, changed) | adds `control: AssistLoopControl`, `pause_reason: Literal['user','audit','budget','project_paused','vlm_unavailable'] | None`, `iteration: int`, `slices: int`; `budget.pools[]` gains `prompt_tokens`, `completion_tokens` (limit, reserved, spent) and `usage_missing`; `status` keeps the job status vocabulary (with `queued`) |
| `AssistLoopBudgetEdit` (new request) | `limits: list[{pool: Literal['vlm','audit'], unit: Literal['crops','requests','prompt_tokens','completion_tokens','wall_s'], limit_value: int | None}]`, 1-10 entries |
| `AssistLoopMetrics` (new, v0.8.0; computed in `src/domain/assist_loop_metrics.py`) | `labels_per_hour`, `items_labeled` (propagated + direct), `vlm_crops_per_labeled_item`, `precision_vs_human: {n, agreed, estimate, wilson_low, wilson_high} | None` (blind audit only), `review_minutes_per_100_labels: float | None`, `audit_agreement` |
| SSE frames | the types of section 5.4; `id` is the opaque `<xid>-<id>` cursor |

---

## 6. Prototype protocol

All prototypes run after the v0.7.0 tag (MQ-9) unless the owner answers J1 otherwise. None changes
product code; each writes its registration (arms, bars, data pins) and commits it before any score.

### 6.1 Code location: tracked `scripts/research/auto_label/`

✅ **Recommended: tracked**, in `scripts/research/auto_label/`, with an exact-pinned
`scripts/research/auto_label/requirements.txt` (every line `==`) and a `pip freeze` lock beside it.
Reasons: [docs/research/CLAUDE.md](../../research/CLAUDE.md) requires a reproduce command that works
from a clean checkout for every published number, which a gitignored script cannot give; the
clustering research already lives in `scripts/research/`; the naming-leak and docs checks then
cover the code. Set P is reached only through a path argument or an environment variable the
operator sets, never written in the code or the output; every output goes to
`artifacts_local/auto_label/` (gitignored). Unit tests for the metric functions (pure, CPU) go to
`tests/research_scripts/test_auto_label_metrics.py`.

❌ Gitignored scripts: faster to write, but numbers that cannot be reproduced from the tree break
the research rules and the white paper.

Note: the existing `scripts/research/requirements.txt` has unpinned lines; the new file is pinned
from the start (owner decision D-44, exact pins).

Layout (files created by the AL0 lane, not by this plan):

| File | Role |
|---|---|
| `scripts/research/auto_label/crops.py` | build a pinned crop list from a project export or a public dataset with ground truth (IoU 0.5 matching), seeded |
| `scripts/research/auto_label/vlm_rate.py` | AL0 driver against a local OpenAI-compatible endpoint; records `usage` |
| `scripts/research/auto_label/answer_cache.py` | AL1: ask once per crop, cache answers keyed by (crop sha256, model, pack revision) as Parquet |
| `scripts/research/auto_label/replay.py` | AL3: run selectors against cached answers and ground truth; no GPU |
| `scripts/research/auto_label/selectors.py` | random, cluster_sampled (imports the product's pure `loop_stats` and `loop_tree`), cluster_margin, uncertainty, k-center |
| `scripts/research/auto_label/metrics.py` | label rate, yield, precision with Wilson intervals, area under the labels-per-cost curve |
| `scripts/research/auto_label/review_study.py` | AL2: build Label Studio tasks and read back `lead_time` |

Product code may not import research code; research code may import the product's pure modules
(`src/domain/`, prompt rendering and reply parsing in `src/services/labeling/`) so the prototype
asks exactly what the product asks.

### 6.2 The venv

A separate venv outside `.venv` (which holds CPU torch and `faiss-cpu`):
`python3.12 -m venv artifacts_local/venvs/auto_label`, then
`artifacts_local/venvs/auto_label/bin/pip install -r requirements-test.txt -r scripts/research/auto_label/requirements.txt`.
Shared packages use the versions of [01_package_pins.md](01_package_pins.md) (torch 2.15.0
with torchvision 0.29.2, NumPy 2.5.4, the product's Ultralytics pin of the day, the product's
transformers pin); research-only packages (vLLM, Label Studio, statsmodels, cleanlab, pyarrow) are
pinned `==` to their current release on the day of the AL0 commit, and the lock file records the
resolver result. No Docker build; vLLM runs as `vllm serve <model> --host 127.0.0.1` from this
venv on the agreed GPU.

### 6.3 Datasets

| Set | Licence | Use | Ground truth |
|---|---|---|---|
| COCO 2017 val and a train subset | images Flickr terms, annotations CC BY 4.0 | AL0-AL5; the #286 project size (2,500 images) and a 10,000-image project for the cost ratio | instance labels matched to detector crops at IoU 0.5 |
| Open Images V7 (validation pins already in the tree; the train split for Z5) | images CC BY 2.0, annotations CC BY 4.0 | AL1, AL3, AL6 | box labels; box `Source` (manual or machine-verified) recorded |
| LVIS v1 (COCO images, 1,203 classes, federated) | CC BY 4.0 annotations | AL1 rare and common classes: VLM confusion on long-tail names | scored only on classes verified for that image (`neg_category_ids`, `not_exhaustive_category_ids`) |
| Set P (private, about 21 MP photos, 86 owner classes) | private | AL0 rate on large sources, AL1 precision, AL2 if the owner reviews | owner annotations; aggregates only, in `artifacts_local/` |

Pins (manifests with SHA-256, seeds) go to `scripts/datasets/manifests/` for public sets only.

### 6.4 Prototypes, metrics, repetitions and stop rules

| ID | Question | Data and arms | Metrics | Repetition | GPU | Stop rule |
|---|---|---|---|---|---|---|
| **AL0** label rate (#286 S0, #466) | how fast does the VLM label, and what does a crop cost in tokens | 500 COCO crops and 500 set-P crops; the two `tested` catalog models; tiered, not factorial: images per call 4 at concurrency 4, 8, 16, 32, then 1 and 8 at the best concurrency | crops/s, requests/s, p50/p95 latency, prompt and completion tokens per crop (from `usage`), unreadable rate, GPU utilization (`nvidia-smi dmon`) | best cell 3 times; others once | about 3 h on one card | stop climbing concurrency when doubling it gains under 5 %; stop a model whose unreadable rate exceeds 10 % |
| **AL1** VLM precision and noise model | how right is the VLM per class, against ground truth | every detector crop of the 2,500-image COCO project (about 15,700), the 8,000-image Open Images validation pin (about 15,000 labelled crops), 2,000 LVIS crops over 100 classes, 2,000 set-P crops; default model, one prompt-pack revision | per-class precision and recall with Wilson 95 % intervals, macro averages, the 20 most frequent confusion pairs, `__new__` rate; a 300-crop repeat for self-consistency | one pass (cached), the 300-crop repeat | about 1.5 h at 10 crops/s | if COCO top-1 accuracy is below 0.70, record it and drop the propagation arm of AL3 (no gate can reach 0.95 precision on top of it) |
| **AL2** human review cost and precision (needs a person, J8) | what a person spends per label, by mode; what the true precision is | Label Studio locally; public COCO and Open Images crops only; two 25-minute sessions, counterbalanced: A label from scratch (pick the class), B verify proposals (accept or correct, a grid per proposed class); disjoint item sets with the same class mix; then a 200-item blind audit of AL3's machine labels | review minutes per 100 labels per mode (Label Studio `lead_time`); human-vs-ground-truth agreement (label noise); human-vs-machine precision with Wilson intervals | once per mode; one reviewer, stated as a limit | none | stop after the two sessions; a session interrupted for more than 5 minutes is discarded and repeated once |
| **AL3** selection strategies by replay (CPU) | which strategy labels the most items at a fixed precision and cost | AL1 caches as the labeler (every ask is a real cached answer); arms `random`, `cluster_sampled` (product statistics), `cluster_margin` and `uncertainty` (linear probe on stored vectors, retrained per round), `diversity` (k-center); budgets of 1,000, 3,000 and 10,000 VLM crops | items labeled, precision against ground truth (Wilson low), coverage, items left for review, area under the labels-per-crop curve | 5 seeds per arm | none (CPU, 8 threads, `nice`) | registered bar: a strategy replaces `cluster_sampled` as the default only if it labels at least 1.5x the items at precision Wilson-low at least 0.93 at the 3,000 budget, on COCO and Open Images, in every seed; a strategy that does not beat `random` by 10 % is dropped |
| **AL4** cheap proposers and agreement triage | how much VLM work can a detector, a linear probe or a zero-shot text prior remove | AL1 crops; proposers: `detector_class_name`, linear probe, zero-shot prior from the embedding model's text tower; triage rule "accept when two agree, ask the VLM otherwise" | share of items needing the VLM; precision of the agree set against ground truth; crops saved per 1,000 items | 3 seeds (probe splits) | under 0.5 h (text encoding) | a proposer whose agree-set precision Wilson-low is below 0.95 is not used for acceptance, only for ordering |
| **AL5** teacher-student and promotion (#84) | do propagated labels help or hurt a trained detector | COCO vehicle classes; train the product's YOLO family at the product pin on (a) ground-truth-reviewed labels only and (b) the same plus the loop's propagated labels; evaluate on a frozen ground-truth holdout disjoint from both; #84's wheel loop as an optional second row | mAP50-95 and per-class AP, delta (b) - (a) | 2 seeds | about 6 h | if (b) is more than 1 mAP point below (a), propagated labels stay out of the export by default (confirms #286 Q6); no further arms |
| **AL6** Z5, detector-class roots (#398) | section 6.8 | | | | one research window (6 h) plus 2 CPU days | section 6.8 |

Order: AL0, then AL1 (its caches feed AL3 and AL4), then AL3 and AL4 on CPU while AL2 is scheduled
with the person, then AL5; AL6 runs in its own window when the GPU is free. GPU total about 17 h.

### 6.5 GPU etiquette

- GPUs 0 and 2 only; never GPU 1. One GPU job at a time on the host: every GPU command runs under
  `flock "$SCRATCH/locks/gpu<N>.lock"` for the card it uses (`gpu0.lock` or `gpu2.lock`; 00b X-14).
- Before each GPU block: `uptime` (a 1-minute load above 16 marks the run noisy and it is repeated)
  and `nvidia-smi`; ask the peer sessions for the card and wait for the reply; stop at once when a
  peer asks.
- vLLM with an explicit `--gpu-memory-utilization` agreed with whoever shares the card.
- No compose stack is needed; if one is used, it runs under its own `-p` name, never the default.
- Time boxes per prototype as in section 6.4; an overrun is reported, not silently extended.

### 6.6 Results recording

- One note per prototype in `docs/research/` (`auto_label_rate.md`, `auto_label_vlm_precision.md`,
  `auto_label_review_cost.md`, `auto_label_strategies.md`, `auto_label_teacher_student.md`), each
  with Setup, Method, Results, Findings, Open issues, Reproduce, and the registration committed
  before the results.
- Raw per-run JSON, logs and GPU samples in `artifacts_local/auto_label/<id>/`; the small files
  that back a published number in `docs/benchmarks/auto_label_<id>/` with `SHA256SUMS`. Never prune
  either folder.
- Set P: numbers stay in `artifacts_local/` by default; publishing an aggregate line (no names, no
  paths) needs the owner's yes per note (J11).
- Each note updates its issue (#466, #398, #84) and the research tables in
  `docs/research/README.md` and `docs/design/README.md`.

### 6.7 What the prototypes cannot show

A replay labeler hides latency effects and VLM drift over a long run; one reviewer is not a
population; dataset ground truth has its own noise (AL2 measures it); class-level only (no boxes,
no make and model).

### 6.8 Z5: detector-class roots with the hold-back guard (absorbed from the #398 plan)

**Two stages; the second only after the first passes.** Z5 is research (a note); Z5-I is product
(section 7.6), only on a Z5 PASS, a passing visual check and the owner's yes. **Stop rule:** FAIL,
or INCONCLUSIVE again, means IVF roots stay, nothing ships, #398 closes with the note.

**Headline correction.** The issue's "at least 1,200 open-set crops" was sized from the unguarded
`key` arm (19.9 open-set items propagated per run on 395 held-out open-set crops). The guarded arm
propagated 9.2 per run on the same crops (0.0233 per crop), so reading its bar B-c (at least 30
propagated per run) needs about 1,290 held-out, about 2,470 total open-set crops at Z4's share.
Also: about 2.2x fewer decisions is the guarded arm (0.834 / 0.377); the unguarded arm is 2.85x;
the guard costs 29 % more effort than `key`. The 95th percentile is of each node's sampled members'
cosine distances to their own centroid, not project-wide.

**Set (registered before download).** Open Images V7 **train** split at Z4's composition scaled
about 4.4x:

| Group | Z4 images | Z5 images |
|---|---|---|
| `openset` (one of the 12 open-set animals) | 644 | 2,850 |
| `animal` (largest box cat, dog, horse, sheep or cow) | 1,500 | 6,600 |
| `vehicle` (largest box car, truck, bus, motorcycle, bicycle or train) | 2,500 | 11,000 |
| `rest` | 3,356 | 14,750 |
| total | 8,000 | 35,200 |

Projected: about 63,700 labelled crops, about 3,300 open-set, about 1,700 held out, so `key_guard`
should propagate about 40 per run. **Count before downloading**: the fetcher streams the train box
CSV, applies Z4's filters (CC BY 2.0, no stored rotation, not group-of, not depiction, area at
least 0.5 %), writes the counts into the registration, and stops INCONCLUSIVE-by-registration if the
projected held-out open-set count is under the registered minimum (1,300). ❌ an open-set-dense set
(it changes the risk being measured: open-set crops stop being a minority inside big pure roots);
🟡 an open-set-dense stress row on the calibration half only, unbarred.

**Fetcher work (E-0, docs and research scripts only).** Move the Open Images URL tables out of
`scripts/datasets/fetch_openimages_plates.py` into `scripts/datasets/openimages_urls.py` with a
`train` entry (locations from the official download page,
<https://storage.googleapis.com/openimages/web/download_v7.html>), update every importer, keep no
re-export; stream the multi-GB train CSV (`csv.DictReader` over the open file) for every split, one
path; check the CSV headers before use (fail closed on a missing `License`, `Rotation`,
`OriginalURL`, `IsGroupOf`, `IsDepiction`, coordinates); record the box `Source` distribution;
`--exclude-manifest` for every earlier pin including Z4's, file name and SHA-256 disjointness;
commit a COCO-to-Open-Images class-name map as data and have `build_labels.py` and the selection
read it (a test: no name both detectable and open-set). Red tests: a fixture CSV streamed in chunks
gives the in-memory selection; a missing `License` header raises; `--split train` is accepted.

**Registration** (`docs/research/cluster_config_z5.md`, committed before scoring): split, filters,
quotas, a new seed, pre-download counts, disjointness; a throwaway project on a research stack
under its own `-p`, `/health` ready first, detector and encoder ids from the items; split by source
image 50/50 with a new seed (fold 1 calibrates, every bar read once on the held-out half); arms
`key`, `key_guard`, `ivf_loop`, `ivf_gate` (and `within_key` only as an unbarred row, J19); simulator
pinned to the loop commit (theta 0.9, alpha 0.05 split over two looks, max dissent 1, branching 4,
max depth 3, min node 40, guard percentile 95, seeds 0-19); bars per J18: option A (Z4 unchanged:
U2 >= 0.95; U1 <= 0.5 x the cheaper IVF arm; open-set U2 >= 0.90, INCONCLUSIVE below a mean of 30
open-set propagated per run) or option B (A plus B-d: wrong open-set propagations per 1,000 pool
items under a registered limit, bootstrap 95 % upper bound); adoption: the cheaper passing of `key`
and `key_guard`, ties to `key`, else IVF stays; unbarred rows: 10 % noisy labeler, registry-closed
reading, per-key purity, held-back counts and how many would have been wrong, per-class exposure.

**Run.** Preflight (load, peers, `-p`, GPU 0 or 2); fetch with counts; ingest with
`scripts/curation/ingest_upload.py --project <slug> --force-rehash` (time box 6 h); export vectors
read-only (`scripts/research/cluster_diag/fetch_vectors.py`), labels at IoU 0.5; fit IVF with the
production configuration; simulate every arm with the existing simulator
(`scripts/research/cluster_diag/loop_sim.py`, its `held_back` at `GUARD_PERCENTILE = 95.0`; Z4 bars
in `z4_bars.py`, driver `z4_run.py`) on CPU (8 threads, `nice`); read the bars once; visual check
only if a machine bar passes; publish backing JSON in `docs/benchmarks/cluster_config_z5/` with
`SHA256SUMS`; delete the throwaway project; update #398 and #346.

**Visual check** (owner rule: VLM check plus a human re-judge). Contact sheets from
`src/services/labeling/contact_sheet.py`; at most 20 cluster sheets, half targeted (largest passing
roots of keys that hold open-set crops, and held-back sets), half a seeded uniform control. Judge
agreement first on 40 planted calibration-half sheets (half clean, half contaminated): pair kappa
>= 0.7 and flag precision >= 0.8, else the judge only ranks sheets for a person (J26). About 10 %
planted controls; 10 % re-rendered in a second order for position bias; a blind seeded 5 % human
re-judge (at least 20); judge-vs-human kappa below 0.7 suspends the judge. Purity with a Wilson
interval beside the ground-truth U2; a half-width above 0.10 is "insufficient sample". Open Images
animal crops are public: a hosted judge is allowed; a sheet flagged for faces goes to the local VLM
or a person only.

| Outcome | Then |
|---|---|
| counts below the registered minimum | INCONCLUSIVE by registration; close #398; IVF stays |
| an arm FAILS and none PASSES | FAIL; close #398 |
| none FAILS, some INCONCLUSIVE, none PASSES | INCONCLUSIVE again; the line stops; close #398 |
| an arm PASSES, the visual check fails or is insufficient | not adopted; close #398 |
| an arm PASSES and the visual check passes | owner decision on Z5-I |

Time box: one research window (6 GPU hours) plus two CPU days. Z5 must run on the stack whose
vectors the product makes (after B5c "crops at model size"), or first pass a 1,000-crop cosine
parity check (>= 0.999).

---

## 7. Production design

### 7.1 The loop as an assistant, not a batch job

```
 Cropwright                      API (stateless)                    Celery queue assist_loop
 ---------                       ---------------                    ------------------------
 plan / start / pause / resume   routes -> start_job / pause_job    op.assist_loop.run (one slice)
 edit budget / cancel / rollback        -> jobs + assist_loop_*       select -> reserve -> ask ->
 review grid (v0.8.0)            review routes -> label ops            gate -> propagate (label op)
 event log, metrics  <-- SSE ---- events rows + Valkey wake-up  <---   audit -> checkpoint -> yield
                                                                     reads: OpenSearch items, vectors
                                                                     writes: OCC item writes, PG rows
```

Lifecycle of one run: plan (dry run, free) -> start -> iteration 1: pin the partition (node
snapshots), best-first node steps in slices, audit batches -> the iteration closes when no node is
open or a stop rule fires -> the run completes, or pauses for a person (audit, budget) -> review in
Cropwright (v0.8.0): verify queue ordered by the strategy, blind audit queue random -> metrics.

### 7.2 Module layout after B2 (lane 4D)

| Path | Content |
|---|---|
| `src/domain/loop_stats.py`, `src/domain/loop_tree.py` | kept |
| `src/domain/assist_loop_budget.py` (new, pure) | reservation units for a call, prompt-token estimate, forfeit arithmetic |
| `src/domain/assist_loop_events.py` (new, pure) | event type names and payload builders (section 5.4) |
| `src/db/models/assist_loop.py`, `src/db/repos/assist_loop.py` (new) | every SQL statement of section 5; fenced writes |
| `src/services/curation/assist_loop/` | `driver.py`, `node_step.py`, `evaluate.py`, `propagate.py`, `splitter.py`, `audit.py`, `partition.py`, `plan.py` kept; `store.py` and `rollback.py` deleted (repo and label-op undo replace them); `budget.py` rewritten on the repo; `job.py`, `start.py`, `runtime.py` become the `start_job` caller and the task handler |
| `src/tasks/trigger.py` | `op.assist_loop.run` registered with `register_job('assist_loop', handler)` |
| `src/services/labeling/vlm_client.py` | returns `usage` with every response; `record_vlm_usage` called by each VLM call site |
| `src/schemas/curation/assist_loop.py`, `src/routers/curation/assist_loop.py` | section 5.3 and 5.8 |

### 7.3 Review in the loop (v0.8.0, #466 remainder)

- `{P}/assist/loops/{loop_id}/review` (new, GET): paged items for `queue=verify|blind_audit`;
  verify is grouped by proposed class and ordered by the run's strategy priority (largest
  propagated nodes first, then lowest agreement), with a seeded random 5 % mixed in for unbiased
  measurement; blind audit is a seeded uniform sample of the run's machine labels (propagated and
  direct), at least 30 and about 1 % of written labels, never showing the machine class. The server
  stamps `served_at`.
- `{P}/assist/loops/{loop_id}/review` (new, POST): 1-200 decisions `{item_id, queue, decision,
  class_name, served_at, active_ms}`; accepted and corrected items are written as one label
  operation; locked items are refused per item (`item_locked`), never overwritten.
- `{P}/assist/loops/{loop_id}/metrics` (new, GET): `AssistLoopMetrics`.
- Review minutes per 100 labels = `100 x sum(min(active_ms, 120 s)) / labels produced` over the
  window, with `decided_at - served_at` as the fallback when `active_ms` is absent; reported with
  the number of reviewed items.
- The existing accuracy audit (`/curation/projects/{project}/audit/queue`) stays the project-wide
  audit; the loop's blind audit is per run. One shared allocation function in `src/domain/audit_math.py`.

### 7.4 Retraining and promotion (v0.8.0+, #84)

- An iteration may end with a `train` child job (`jobs.parent_job_id`), on the existing `gpu_train`
  queue, using only human-validated labels by default (J6). The loop run pauses with
  `pause_reason = 'user'` and a `waiting_train` progress stage while the child runs; Cropwright shows
  the child's progress. No automatic resume across a training boundary in the first release.
- A trained model becomes the next iteration's proposer (`proposer_kind = 'detector'`) only after
  a bake-off against the current proposer on the **frozen holdout** passes the bake-off profile's
  bar, and a person clicks "use as proposer"; it is never promoted to the live detector by the loop.
  Promotion keeps its own job and `promotions` rows ([03e](03e_stage5_gpu_queues.md) 5D);
  class identity by name end to end (`models.labels`).

### 7.5 Strategies and proposers (v0.8.0, only those that passed AL3 and AL4)

A `Selector` protocol (`select(ctx, iteration) -> list[Selection]`) and a `Proposer` protocol
(`propose(items, budget) -> list[Proposal]`) in `assist_loop/`, with `cluster_sampled` as the first
implementation of each. A new value enters `assist_loop_runs.strategy` and
`assist_loop_proposals.proposer_kind` CHECKs by migration only with its implementation and tests.
`sweep` (label every eligible item directly, the `autolabel` job's VLM stage) joins the loop as a
strategy if J7 says yes, so one control surface covers every whole-project VLM run.

### 7.6 Z5-I (only on a Z5 PASS and the owner's yes)

- `DetectorClassPartition` behind the existing `PartitionReader` seam; one factory
  `partition_reader(opensearch, scope)` used by both the API plan and the task (no second
  construction site); `LoopScope.partition: Literal['clusters','detector_class']` plus
  `detector_class_names: list[str] | None` (normalized, mutually exclusive with `cluster_ids`, 422
  otherwise).
- Root keys `k<first 12 hex of sha256(normalized name)>` with `root_name` and `root_kind` on every
  node; path grammar `(r\d+|k[0-9a-f]{12})((?:\.\d+)*)`; the `assist_loop_nodes.path` CHECK widened
  in the same migration; ordering compares `(kind, int or str)` so `r10002` still sorts before
  `r10010`; `survey` refuses 409 `partition_key_collision` on a hash collision. ❌ names in paths
  (dots, slashes, non-ASCII); ❌ ordinals (`k3` is a class index on the wire).
- `survey`: composite aggregation over `detector_class_name` with `missing_bucket` (the `none` key,
  J22) on the same pool as IVF roots (candidate clusters, J22); `root_of`: the key each id has now,
  so a re-detect reads as drift and the node goes `stale`. Start refuses `reprocess_running` while a
  reprocess job is active (a courtesy check; drift at write time is the guard).
- Vectors at key scale: one paged reader into a preallocated `float32` array replaces the
  `mget`-into-lists `splitter.fetch_embeddings`; under the existing two-split semaphore; peak memory
  under 2 GB on a 100,000-member synthetic root (runtime only; quality claims never on synthetic
  vectors).
- Hold-back guard in the propagate step, gated by `LoopGate.hold_back: bool`: sampled = members
  with a usable answer (failed answers excluded); centroid of the sampled unit vectors (zero norm:
  hold back nothing, log `guard: degenerate`); limit = `numpy.percentile(1 - V @ c, 95)` (default
  linear method); hold back unasked eligible members beyond the limit, and every member without a
  vector (fail closed). The held-back ids are written (a `held_back` column set on the node, or an
  `assist_loop_selections` purpose `held_back`) **before** the propagation, and a resume reads them,
  never recomputes them. `held_back` joins the skip reasons; held-back members are never written,
  so the undo is unaffected. They get a review surface (`set=held_back` on the node members route
  and a count in Cropwright, J24). `GUARD_PERCENTILE = 95.0` is a constant, not a request field.
- Tests: parity of the guard with `loop_sim.held_back` on 200 seeded nodes (mutations: percentile
  50, inverted comparison, failed answers counted); no vector held back; held-back ids logged before
  propagation and stable across a re-embed; grammar accept and refuse lists; mixed-length ordering;
  one factory used by both sites; the cross-project leak test; the per-class lock-rule test with key
  roots; one plausibility loop on a calibration-half project with the real VLM in the next live
  window; Cropwright screenshots at desktop and 430 px.

---

## 8. Landing lanes

| When | Lane | Plan file | Content |
|---|---|---|---|
| B2 stage 1 | **1A** `b2/db-core` | [03a](03a_stage1_foundation.md) | the tables of 5.1, the `jobs` columns and transitions of 5.2 in `0001_control_plane`; `tests/db/test_assist_loop_schema.py` (CHECKs, cascades, RLS through `run_id`, the budget CHECK refuses an over-reserve) |
| B2 stage 2 | **2C** `b2/config-pg` | [03b](03b_stage2_control_data.md) | `assist_loop_preset` kind and axis (5.7) |
| B2 stage 2 | **2D** `b2/records-pg` | [03b](03b_stage2_control_data.md) | label-operation kinds `assist_loop_propagate` and `assist_loop_review` |
| B2 stage 3 | **3A** `b2/job-runtime` | [03c](03c_stage3_job_runtime.md) | `Yield` and the `running -> queued` transition, `queued -> paused`, `retry_count` in the claim, `pause_reason`, `pausable` and `slice_s` in `register_job`, the reaper's system-pause resume rule; red tests: three pause/resume cycles then a fourth resume claims (red on today's claim); a yielded job re-dispatches and keeps its attempt budget |
| B2 stage 3 | **3B** `b2/events-pg` | [03c](03c_stage3_job_runtime.md) | nothing new beyond the producer list; the loop types ride the generic feed |
| B2 stage 4 | **4D** `b2/trigger-workers` | [03d](03d_stage4_job_families.md) | the port: loop on the tables (no `jobs.params` budget), pause and budget routes, events, `usage` capture and `record_vlm_usage`, slices, the lock-rule and budget tests carried over (randomized 200-seed budget test against the SQL reservation, kill after each step then resume, pause latency in the live test) |
| B2 stage 4 | **4E** `b2/frontend-queued` | [03d](03d_stage4_job_families.md) | Cropwright: pause button and paused state with reason, budget edit dialog, event log fed by SSE, `control` flags drive every button |
| after the tag | `research/auto-label` | this file, section 6 | AL0-AL5 (one lane per prototype, registration commits first) |
| after the tag | `research/z5` | this file, section 6.8 | AL6 |
| v0.8.0 | `feat/loop-review` | this file, section 7.3 | review and metrics routes, Cropwright review grid and metrics panel |
| v0.8.0 | `feat/loop-strategies` | this file, section 7.5 | only the strategies and proposers that passed AL3/AL4; `sweep` per J7 |
| v0.8.0+ | `feat/loop-z5i` | this file, section 7.6 | only on a Z5 PASS |
| v0.8.0+ | `feat/loop-retrain` | this file, section 7.4; #84 | train child, bake-off gate, proposer swap |

Lanes follow the house model: one worktree and branch each, targeted red-then-green tests and
static checks only in the lane, one full sweep per wave on the integration branch.

---

## 9. Risks

| # | Risk | How it looks done while wrong | Guard |
|---|---|---|---|
| R1 | Wrong labels at scale (a shared VLM confusion passes the gate) | high gate pass rate, high audit agreement with a second model of the same family | blind human audit with a Wilson interval; validated-member veto; machine kind excluded from export; one-call undo |
| R2 | Automation bias in the precision number | reviewers accept what they are shown; "precision 0.98" | the blind audit queue never shows the machine class (CHECK in `assist_loop_reviews`) |
| R3 | Attempts exhausted by pauses (G-1) | tests pause once and pass | the three-cycle red test in 3A |
| R4 | Starvation on the single consumer (G-2) | one-project tests pass | slices; a two-project test where the second loop's first node starts within one slice |
| R5 | Budget over-spend across a lease handover | single-process tests pass | SQL reservation with the CHECK; ledger fenced by token; randomized crash test |
| R6 | Token budget silently uncapped | endpoint returns no `usage`, the counter stays 0 | settle at the reserved upper bound and count `usage_missing`; warning event |
| R7 | Pause that does not pause (long in-flight batches) | pause button turns grey, spend continues | check before every reservation; latency measured in the live test |
| R8 | Confirmation bias when retraining on machine labels | mAP on a holdout that itself contains machine labels | the frozen holdout is human-validated only; AL5 decides the export default |
| R9 | Strategy gains that are noise | one seed, no random baseline | 5 seeds, random arm, pre-registered bars (Lüth 2023) |
| R10 | Proposal tables grow without bound | fine on a 2,500-image test | retention of 5.1; counts kept on nodes |
| R11 | External data exposure through audit or a remote primary | consent checkbox ticked once | the three-part consent per run; holdout never sent; faces never to a hosted judge |
| R12 | Set P exposure | a path in a log line committed to the tree | path only by argument; outputs in `artifacts_local/`; naming scan; aggregates need the owner's yes |
| R13 | Z5 traps (sizing from the wrong arm, a working guard hiding its own bar, open-set-dense sets, reading the held-out half twice, recomputing the guard on resume, path ordering, ordinal keys, two construction sites, the `none` key, memory at key scale, a courtesy check mistaken for safety, vectors of another pipeline, registry-closed projects, detector mix, blast radius of a huge root) | each as described in section 6.8 and 7.6 | each guard named there |

---

## 10. Owner questions

| J | Question | Recommendation |
|---|---|---|
| J1 | Run AL0 (label rate, about 3 GPU hours) before the tag, since it sizes every budget default, or keep all prototypes after the tag (MQ-9)? | after the tag, as decided; the B2 port keeps today's defaults |
| J2 | Put the seams of section 5 into the B2 plan now (tables in `0001_control_plane`, `usage` capture in lane 4D, which gains `vlm_client.py`)? | yes: adding them after B2 means a second migration and a second port of the loop |
| J3 | Make `autolabel` pausable at stage and chunk boundaries too, or keep pause for the loop only? Assist runs stay cancel-only | yes for `autolabel`; assist runs no (short, 5,000-item cap) |
| J4 | Cancel becomes terminal for the loop (today a cancelled loop can be resumed); "stop for now" is pause | yes |
| J5 | Default when a budget limit is reached: complete the run (today) or pause it so the budget can be raised and resumed? | pause (`on_budget: pause`), with the paused run visible as holding the project's loop slot |
| J6 | Propagated labels in the training export: never until a person accepts them (today), or an opt-in flag that records the machine share? | never by default; revisit after AL5 |
| J7 | Fold the `autolabel` job's VLM sweep into the loop as `strategy: sweep` (one control surface), in v0.8.0? | yes, in v0.8.0, after the B2 port |
| J8 | AL2 needs about one hour of a person's time (two 25-minute sessions plus a 200-item blind audit). Who reviews, and is Label Studio locally acceptable as the study tool? | the owner or a delegate; Label Studio in the research venv |
| J9 | Read the daily VLM cap from `vlm_usage_hourly` instead of an OpenSearch count, and add an optional price per million tokens to the endpoint document for cost reporting? | yes to both, v0.8.0 |
| J10 | Retention: proposals and selections of runs finished over 90 days ago deleted, `vlm_usage_hourly` kept 400 days? | yes |
| J11 | May set-P aggregate numbers (no names, no paths) appear in the public research notes? | only per note, with the owner's yes |
| J12 | Box targets (open-vocabulary proposals reviewed in the loop) or keep boxes in the region stage? | region stage; revisit after #84 |
| J13 | Slice length 600 s and the 30 m / 35 m slice limits? | yes; tune after the 4D live test |
| J14 | Acceptance for the first product claim: precision Wilson-low at least 0.93 against the blind human audit, and at least 0.95 point estimate (the #286 Q7 bar), on a public 10,000-image project? | yes; the 1:20 crop ratio stays a goal until AL3 shows a strategy that reaches it |
| J15 | May the loop start a training child job by itself at the end of an iteration (never promote), or only on a click? | only on a click in the first release |
| J16 | Milestones: #466 and #398 stay v0.8.0; the AL notes are the research deliverables after the tag | yes |
| J17 | (was #398 J1) Z5 set size: about 35,200 train images at Z4's composition, registered minimum 1,300 held-out open-set crops; the pin stores ids and hashes only if 17 MB in git is too much | 35,200; ids and hashes only |
| J18 | (#398 J2) Bars: Z4 unchanged (A), or Z4 plus an exposure bar B-d (B) whose limit is set before scoring (for example at most 1 wrong open-set label per 1,000 pool items, upper bound) | A with the J17 size; B only if the owner prefers a bar the guard cannot make unreadable |
| J19 | (#398 J3) Keep `within_key` in Z5 | unbarred row only; it was killed in Z4 |
| J20 | (#398 J4) Root-key grammar `k<12 hex of sha256(name)>` plus `root_name` on the wire | yes |
| J21 | (#398 J5) After adoption, `detector_class` and `hold_back=true` become the defaults, `clusters` stays selectable | yes |
| J22 | (#398 J6) Pool and the `none` key: exactly the IVF loop's pool (candidate clusters), `none` included | yes |
| J23 | (#398 J7) A cap on very large roots | no cap first; require the audit when a root above 20,000 eligible members would be visited |
| J24 | (#398 J8) A review surface for held-back members | yes: `set=held_back` plus a Cropwright count and link |
| J25 | (#398 J9) Detector mix: key by normalized name only, report the detector set per key | yes |
| J26 | (#398 J10) Judge for the visual check | hosted judge only after it passes planted-sheet agreement on Open Images animals; else the local VLM ranks and a person decides |

---

## 11. Required changes to other plan files (for the integration agent)

| File | Change |
|---|---|
| [v070_rebuild/02a_postgres_schema.md.md](02a_postgres_schema.md) | add section 5.1 with the DDL of section 5.1 here; add `retry_count` and `pause_reason` to `jobs` with their CHECKs; `job_attempts.outcome` gains `yielded`; section 4: `config_documents.kind` and `config_activations.axis` gain `assist_loop_preset` (and the axis-kind pair); section 9: the retention rows of 5.1 and the `jobs` keep-list gains `assist_loop_runs` |
| [v070_rebuild/02b_jobs_locking_events.md.md](02b_jobs_locking_events.md) | section 1: the transitions of 5.2 (`running -> queued` yield, `queued -> paused`, widened `running -> paused`, reaper `paused -> queued` for system reasons); the claim uses `retry_count < max_attempts`; section 2.1: `assist_loop` slice 600 s, soft/hard 30 m/35 m, max queue wait 24 h, `pausable`; `autolabel` `pausable` (J3); section 3.1: `ctx.paused()` read before every reservation, `Yield` return; section 3.3: pause route semantics of 5.3, cancel terminal for every kind; section 4: the reaper's system-pause resume rule; section 6.2: the loop event types of 5.4 in the producer list |
| [v070_rebuild/03a_stage1_foundation.md.md](03a_stage1_foundation.md) | lane 1A owns the loop tables in `0001_control_plane` and `tests/db/test_assist_loop_schema.py` |
| [v070_rebuild/03c_stage3_job_runtime.md.md](03c_stage3_job_runtime.md) | lane 3A: `Yield`, `retry_count`, `pause_reason`, `pausable`, `slice_s`, the reaper rule, and the red tests named in section 8 |
| [v070_rebuild/03b_stage2_control_data.md.md](03b_stage2_control_data.md) | lane 2C: the `assist_loop_preset` kind; lane 2D: label-operation kinds `assist_loop_propagate`, `assist_loop_review` |
| [v070_rebuild/03d_stage4_job_families.md.md](03d_stage4_job_families.md) | 4D-2: replace "`budget.json` (into `jobs.params` and `checkpoint`)" and "events are `events` rows" with "the loop record moves to the `assist_loop_*` tables ([09e_research_auto_label_assist.md](09e_research_auto_label_assist.md) section 5); `events` rows are the UI feed only"; add the pause and budget routes, `usage` capture (`src/services/labeling/vlm_client.py` and `record_vlm_usage` added to 4D's owned files), slices, `rollback.py` and `store.py` deleted in favour of the label-op undo and the repo; 4D red tests: three pause cycles, a two-project slice interleave, the SQL budget under a randomized crash test; 4E: pause, budget edit, event log, `control` flags |
| [v070_rebuild/03f_stage6_scheduler_scaleout.md.md](03f_stage6_scheduler_scaleout.md) | 6D doc sync: `auto_label_loop_plan.md` header says its storage, events, budget and control sections are superseded by this plan for the rebuild |
| [v070_rebuild/reference/traceability.md.md](reference/traceability.md) | #466 and #398 rows point here; #84 row gains "loop side: section 7.4 here" |
| [v070_rebuild/reference/prior_plans_reconciliation.md.md](reference/prior_plans_reconciliation.md) | `auto_label_loop_plan.md`: KEEP-REFERENCE, partly superseded by this plan (storage, events, budgets, control); add a row: `issue_plans/issue-398-plan.md` absorbed into this plan and deleted |
| [v070_rebuild/reference/github_actions.md.md](reference/github_actions.md) | a comment on #466 and #398 linking this plan; #466 scope: review queue and metrics added (section 7.3) |
| [09b_research_clustering.md](09b_research_clustering.md) | RQ-1 / E-1: point to section 6.8 here for the executable Z5 protocol (registration, set size, stop rule, visual check) |
| [README.md](../README.md) | done in this commit: the `issue-398-plan.md` row removed, a row for this plan added |
| `CLAUDE.md` | nothing now; after 4D merges, the services table and the curation rules mention that the loop's record is in PostgreSQL |
