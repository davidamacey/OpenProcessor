# Visual judge: harness and the first agreement run (E0, Haiku subagents only)

Measured 2026-10-29 on branch `feat/cluster-w8-judge` (harness commit `5a332a23`). CPU only; the
judge was a Haiku model reached through the sheet-file backend (below). The local-VLM half of E0
was **not run** (see Open issues). Design: `docs/design/cluster_search_makemodel_plan.md`,
sections 7 and W8.

## What was built

| Piece | File |
|---|---|
| Contact-sheet builder: seeded selection, letterboxed grid, cluster / pair / search layouts, manifest, rebuild check | `src/services/labeling/contact_sheet.py` |
| Sample builders and the judge runner (CLI) | `scripts/research/visual_judge.py` |
| Labelled public crop pools (COCO, Open Images vehicles, LFW faces) | `scripts/research/judge_pool.py` |
| Synthetic labelled clusters and pairs, truth from the cells shown, random/targeted split | `scripts/research/judge_cases.py` |
| Prompt, strict verdict schema, hosted-model privacy refusal | `scripts/research/judge_protocol.py` |
| Backends: local VLM and sheet-file | `scripts/research/judge_backends.py` |
| Write-ahead call budget, verdict cache, JSONL records | `scripts/research/judge_run.py` |
| Wilson / cluster-bootstrap intervals, Cohen's and Fleiss' kappa | `scripts/research/judge_stats.py` |
| Per-judge metrics, calibration, planted controls, disagreements | `scripts/research/judge_agreement.py` |

Rules the code enforces (each has a test in `tests/test_contact_sheet.py` or `tests/research/`):
the same inputs give byte-identical sheets and a manifest that rebuilds to the same `png_sha256`
(a changed crop refuses); a sheet whose kind is `face`, `identity_text` or `plate` is refused for
any hosted backend before a call and before a budget reservation; the call budget is written to
disk before each call and never exceeded; verdicts are cached by `(png_sha256, model,
prompt_version)`; invalid replies are recorded, not dropped; flagged cells map back to item ids
through the manifest; half of a real-cluster budget is a uniform random draw and the other half
targeted, recorded as `sample` in the truth file so they are never pooled.

## The two judge backends

`local`: any local OpenAI-compatible VLM, by default this stack's `vlm` service at
`http://127.0.0.1:4612/v1` (`./openprocessor vlm use <id>` and `vlm apply` select the catalog
model). A non-local endpoint is refused through the repo's URL policy.

`sheet-file`: no API key in the process. A hosted judge that can view image files (a Haiku subagent
started with the Agent tool, `model: haiku`) acts as the judge:

1. `visual_judge.py run --backend sheet-file --exchange-dir X --model <label> ...` writes
   `X/pending/<id>.png` and `X/pending/<id>.prompt.txt` and reserves the call budget.
2. Give each subagent a list of ids and tell it to view each PNG, follow its prompt file and write
   only the JSON reply to `X/verdicts/<id>.json`. Keep the truth files outside `X` and tell it not
   to look elsewhere.
3. Run the same command again: it reads the verdicts, validates them against the schema and
   appends records to the run JSONL. The reservation is idempotent, so collecting costs no budget.

## Method

Public pools of labelled crops (`judge_pool.py`): the COCO crops of the `cocotest` project (boxes
read from the API read-only, labels from the COCO annotations, pixels from the local COCO copy;
447 crops, 11 classes) and the local Open Images vehicle set (873 crops; car, truck, van,
motorcycle, bus used). Synthetic clusters of 24 members: 45 % clean, the rest with 1, 2, 3, 5 or 8
contaminants from another class (half drawn from a confusable group such as car/truck). Pair
sheets hold 12 pairs, half same class. Truth comes from the cells actually drawn. 10 % of cluster
sheets are re-rendered in a second cell order. Haiku sample: 55 cluster sheets and 12 pair
sheets per set (inside the per-run cap of the plan; larger needs a yes), 50 calibration sheets
(10 clusters at five layouts) on the vehicle set. Question level: "object class" and "vehicle type".

## Results

Raw data: `artifacts_local/bench/cluster_search/w8/{e0_coco,e0_oi,cal_oi}/` (`judge/*.jsonl`,
`report_haiku.json`, `truth.jsonl`, `manifests/`, `sheets/`); pools in `pool_coco`,
`pool_oi_vehicles`, `pool_lfw`.

| | COCO crops | Open Images vehicles |
|---|---|---|
| Cluster sheets judged | 55 | 55 |
| Same/different accuracy (Wilson 95 %) | 0.855 (0.738 to 0.924) | 0.764 (0.637 to 0.856) |
| Kappa with truth | 0.697 | 0.298 |
| "Different" precision / recall | 0.853 / 0.906 | 0.745 / 1.000 |
| False-"same" rate (contaminated sheet accepted) | 0.094 | 0.000 |
| False-"different" rate (clean sheet rejected) | 0.217 | 0.765 |
| Odd-cell flag precision / recall | 0.827 / 0.838 | 0.454 / 0.979 |
| Majority name matches the label | 0.945 | 1.000 |
| Purity: truth vs judge (bootstrap 95 % over sheets) | 0.888 vs 0.886 (0.854 to 0.917) | 0.890 vs 0.763 (0.724 to 0.798) |
| Pair rows judged / unsure | 144 / 6.9 % | 144 / 10.4 % |
| Pair accuracy on decided rows (Wilson 95 %) | 0.910 (0.850 to 0.948) | 0.806 (0.730 to 0.865) |
| Pair kappa / false-"same" | 0.821 / 0.030 | 0.611 / 0.015 |
| Re-ordered twins: verdict flips | 0 of 2 | 0 of 3 |

Against the proposed bars (pair kappa at least 0.7, flag precision at least 0.8): COCO crops meet
both (0.82, 0.83); the vehicle set fails both (0.61, 0.45).

Layout calibration (vehicle set, 10 clusters x 5 layouts, Haiku): the judge answered
"different" for all 50 sheets (27 medium, 20 high, 3 low confidence), so kappa with truth is 0 at
every layout and the rule "smallest layout within 0.05 of the best" selects the smallest (3 x 3)
vacuously. This run does not choose a default layout. Flag overlap between layouts for members
shown in both was 0.82 to 1.00 against the 6 x 4 at 192 px sheet.

## Findings

- On COCO classes Haiku is a usable judge on contact sheets: pair kappa 0.82, flag precision
  0.83, purity estimate within 0.002 of truth. These are 55 sheets per set, so intervals are wide.
- On Open Images vehicle labels it over-flags: it rejects 13 of 17 clean sheets and flags cells
  the labels call the same type. The labels (car, van, truck, bus, with overlapping meanings) are
  not crisply visual, so part of this is label noise, not judge error; the set is a poor
  stand-in for make/model until labels at that level exist. The false-"same" rate stays near zero,
  so the judge is conservative: its "same" is trustworthy, its "different" needs a person.
- The calibration needs a set where the judge says "same" at all; rerun it on COCO crops.
- The order-effect sample (5 twins) is too small to say anything.

## Open issues

- **Local-VLM judging (E0 for the local catalog models) was not run.** No VLM service was up. Starting
  one (a vLLM container pinned to GPU 0 or 2, models already cached under `cache/vllm`) was
  declined by this session's permission layer, so it needs an explicit go-ahead. Needed: one
  container per tested catalog entry (`examples/vlm/catalog.tsv`, budgets 17 GB and 23 GB), then for each of the
  prepared sample directories `visual_judge.py run --backend local --endpoint http://127.0.0.1:<port>/v1
  --model <served name>`. Sample directories already exist for COCO, vehicles and (local only)
  LFW faces: `e0_*` hold 200 cluster and 42 pair sheets each, `cal_*` 30 clusters at 5 layouts.
- Haiku was run on 184 sheets, not the plan's full allowance (100 cluster and 25 pair sheets per
  kind, plus 150 calibration sheets); no face or plate sheet was sent.
- Sheet-file runs record no token counts (the budget reserves the PNG estimate at width x height
  / 750); the six subagents together used about 580k tokens including their own context.
- The Message Batches client and `messages.count_tokens` were not built: no API key was available.
- The 5 % human re-judge and the 20-item spot check are for a person: the per-sheet
  disagreement lists are in `report_haiku.json` under `disagreements`.
- The 3,000-item public set is not materialized as a project; the COCO pool is the 447 usable crops
  of the 200-image `cocotest` sample.

## Reproduce

```bash
PY=.venv/bin/python
$PY scripts/bench/cluster_eval_set.py build --api http://127.0.0.1:5603 --project cocotest \
    --coco-gt <coco_gt.json> --out-dir artifacts_local/bench/cluster_search/w8/cocotest_set
$PY scripts/research/judge_pool.py coco --api http://127.0.0.1:5603 --project cocotest \
    --labels artifacts_local/bench/cluster_search/w8/cocotest_set/labels.jsonl \
    --images-dir <local COCO images> --out artifacts_local/bench/cluster_search/w8/pool_coco
$PY scripts/research/visual_judge.py make-e0 --pool .../pool_coco/pool.jsonl --kind object \
    --level "object class" --out .../e0_coco --seed 0
$PY scripts/research/visual_judge.py verify --sheets .../e0_coco --pool .../pool_coco/pool.jsonl
$PY scripts/research/visual_judge.py run --sheets .../e0_coco --backend sheet-file \
    --model haiku-subagent --exchange-dir <scratch>/exchange/e0_coco --prefix cl --limit 55 \
    --max-calls 70 --max-input-tokens 200000 --run-id haiku-e0_coco-cl
# ... judge the pending sheets, then run the same command again, then:
$PY scripts/research/judge_agreement.py report --sheets .../e0_coco \
    --runs .../e0_coco/judge/haiku-e0_coco-cl.jsonl .../e0_coco/judge/haiku-e0_coco-pr.jsonl
```
