# OpenProcessor: finish the visual-judge calibration, the blind 5 % human re-judge and the label check (issue #406)

Status: **PLAN ONLY, not implemented.** Written 2026-11-04 against branch `int/b1-correctness`
at `81c4e4ef`. Nothing in the repository was changed by this pass.

Builds on: `docs/design/research_clustering_plan.md` (RQ-8, experiment E-10, bars in 4.4 and
4.6, stop rules in 7.1), `docs/research/visual_judge_e0.md` (the E0 result this issue finishes),
`docs/research/fine_grained_benchmark_results.md` (FG-0, the label check, inconclusive) and
`docs/design/cluster_search_makemodel_plan.md` sections 7 and 14 (the judge protocol and owner
decision O3). #399 was consolidated into this issue on 2026-10-31.

Reader: a fresh implementation agent with no memory of this session. Every structural claim
below is cited `path:line` against the commit above. Re-check the line numbers before editing;
the branch moves.

---

## 0. TL;DR

**What I recommend.**

1. **Fix the two code-level causes of the vacuous layout calibration before spending any judge
   call.** The calibration builder draws clean clusters with probability 1/5, not 1/2
   (`scripts/research/visual_judge.py:135`), and the layout rule has no guard against a judge
   that gives one answer to every sheet (`scripts/research/judge_agreement.py:209-212`,
   `:257-266`), so "all different" returns kappa 0 at every layout and the rule picks the
   smallest one. Make calibration sheets exactly half clean, and make the rule return "no
   recommendation" with a reason when the sample or the verdicts are degenerate.
2. **Build the blind human re-judge as a small, separate path with opaque sheet names.** Today a
   person would have to read the judge's exchange directory, whose file names carry the sheet id
   (`rc-ctl003`, `cl0012-r`, `cal004-3x3-160`), which tells the reader which sheets are planted
   controls, re-ordered twins, or layouts of one case. The human gets a fresh directory of
   aliased PNGs and a CSV to fill in; a `collect` step validates every row through the existing
   verdict schema (`scripts/research/judge_protocol.py:329-337`).
3. **Run the label check (FG-0) with the existing tooling** (`scripts/research/fine_grained/fg0.py`,
   `fg0_run.py`), filled by a person instead of Haiku. It needs one new flag (who reviewed), not
   new code.
4. **Spend the judge budget where it can decide something**: Haiku on a balanced COCO-crop
   calibration plus a top-up of the COCO E0 sample and a test-retest slice, inside the
   300-sheet cap the research plan proposes (OQ-3); local VLMs (one rerun of the truncated
   catalog entry, optionally one larger catalog entry) on every kind, in the research GPU
   window RW-3. Vehicles at "vehicle type" level get **no new Haiku spend** until the human pass
   says whether the earlier failure was the judge or the labels.
5. **Record the result in a successor note** `docs/research/visual_judge_calibration.md`, with
   the bars registered in a docs commit *before* any new verdict is read, and backing JSON in a
   new `docs/benchmarks/visual_judge_calibration/` folder with a `SHA256SUMS` file. Close #406
   with it.

**What I am NOT deciding** (all in section 10, judgment calls): the human review tool (J1),
whether to build the Message Batches client (J2, needs an API key the owner may or may not
provide), the calibration design (J3), the human sample size per kind (J4), who re-judges (J5),
the vehicle-type fallback (J6), a third local model (J7), whole-image sheets (J8), the GPU and
window (J9), where results are written (J10), the Haiku model pin (J11).

**Stop rule** (section 9): one balanced calibration per kind, at most one prompt revision, one
local-model rerun plus at most one more local model. A kind that fails its bars is "rank only"
for that kind, a person carries the check, and the line ends. No prompt-tuning loops.

**Compute and people.** CPU: about 2 h (sheet builds, scoring). Hosted judge: at most about 300
sheets (Haiku). GPU: about 2 h on one A6000 in RW-3 for local VLMs. Person: about 3 to 4 h
(blind re-judge plus FG-0), splittable into sessions.

---

## 1. Context: why the obvious approach is wrong here

The obvious approach is "rerun `make-calibration` with Haiku on more clusters and have the owner
look at the disagreement list". Each half of that is wrong today:

- **Rerunning the calibration as built repeats the vacuous result.** The calibration builder draws
  `n_bad` from `(0, 2, 3, 5, 8)` (`scripts/research/visual_judge.py:135`), so about 20 % of
  calibration clusters are clean. The E0 builder uses 45 % clean (`visual_judge.py:47`, `:70-71`);
  the calibration builder does not use that constant. On a set where the judge over-flags (Open
  Images vehicles: false-"different" rate 0.765, `docs/research/visual_judge_e0.md:72`), a 1-in-5
  clean sample plus a judge biased toward "different" gives "different" for every sheet. Cohen's
  kappa is then 0 at every layout (`scripts/research/judge_stats.py:54-66` returns 0, not None,
  because truth still varies), and `recommend_layout` (`judge_agreement.py:209-212`) picks the
  smallest layout within 0.05 of a best of 0. That is exactly what E0 recorded
  (`visual_judge_e0.md:84-88`, `:121-122`).
- **The disagreement list unblinds the person.** `report_haiku.json` holds `disagreements`
  (`judge_agreement.py:329-356`): sheets where the judge and the truth differ, with the judge's
  answer. A person who reads it before re-judging is no longer blind to the judge. The 20-item
  spot check mentioned in `visual_judge_e0.md:165-166` and the blind 5 % re-judge in
  `research_clustering_plan.md` 4.4 step 6 are two different things and must run in that order:
  blind re-judge first, spot check after.
- **The exchange directory leaks roles.** `SheetFileBackend.judge` writes
  `pending/<sheet_id>.png` (`scripts/research/judge_backends.py:111-120`). Sheet ids encode the
  role: planted controls are `rc-ctl{k}` (`visual_judge.py:231`), re-ordered twins `cl{k}-r`
  (`:86-87`), calibration sheets `cal{k}-{cols}x{rows}-{thumb}` (`:139`). Any judge that reads
  file names (a person, or a hosted subagent handed a list of ids) can tell controls from real
  sheets and can tell that five sheets are the same case. Planted controls are only controls when
  the judge cannot spot them.
- **n = 30 per layout cannot resolve the 0.05 tolerance of the layout rule.** At n = 30 binary
  verdicts and kappa near 0.7, the standard error of kappa is roughly 0.1 to 0.15, so "within
  0.05 of the best" is noise. The rule needs a paired comparison over cases (the same case at
  several layouts) and a floor at the RQ-8 bar, or it must say "not distinguishable" and keep the
  current default (`DEFAULT_GRID = (6, 4)`, `DEFAULT_THUMB_PX = 160`,
  `src/services/labeling/contact_sheet.py:36-37`).

---

## 2. What is there today (read from the code at `81c4e4ef`)

### 2.1 Sheet builder (product code, pure)

| Fact | Where |
|---|---|
| Builder version `cs1`, default grid 6 x 4 at 160 px, gutter 4 px, at most 8 prototypes, at most 12 pairs | `src/services/labeling/contact_sheet.py:34-39` |
| Bitmap digits and letters, no font file; PNG without timestamps | `contact_sheet.py:46-60`, `:193-195` |
| Member selection: prototypes, then outliers (3 if at least 12 cells, 2 if at least 6), then near members, then seeded random fill; roles recorded, never drawn | `contact_sheet.py:118-152` |
| Letterboxed thumbnails on grey | `contact_sheet.py:102-107`, `:173-179` |
| Manifest with `png_sha256`; `rebuild_sheet` refuses a changed crop or a differing PNG | `contact_sheet.py:211-239`, `:343-361` |
| Pair sheet: up to 12 rows, sides swapped at random | `contact_sheet.py:276-307` |

### 2.2 Research judge harness (`scripts/research/`)

| Fact | Where |
|---|---|
| CLI: `make-e0`, `make-calibration`, `make-clusters`, `make-search`, `run`, `verify` | `scripts/research/visual_judge.py:1-18`, `:352-408` |
| E0 clean share 0.45, contamination counts (1, 2, 3, 5, 8), 10 % controls of 8 contaminants | `visual_judge.py:47-50` |
| Calibration layouts: 3x3@160, 4x4@160, 6x4@160, 6x4@128, 6x4@192 | `visual_judge.py:51` |
| **Calibration clean share is 1/5** (`n_bad=rng.choice((0, 2, 3, 5, 8))`) | `visual_judge.py:135` |
| Calibration sheet ids encode case and layout | `visual_judge.py:139` |
| Planted control sheet ids encode the role (`rc-ctl{k}`) | `visual_judge.py:218-237` |
| Re-ordered twin ids end in `-r` | `visual_judge.py:85-87` |
| Synthetic clusters: contaminants get distances 0.5 to 1.0, members 0 to 0.5, first 3 members are prototypes | `scripts/research/judge_cases.py:94-122`, `:30` |
| Confusable groups (car/truck/bus/train/van, bicycle/motorcycle, cat/dog, horse/cow/sheep, cup/bottle/bowl) | `judge_cases.py:23-29` |
| Truth derived from the cells drawn | `judge_cases.py:150-162` |
| Half-random, half-targeted split for real clusters, reported apart | `judge_cases.py:165-181` |
| Prompt `p1`; strict JSON verdict schema; hosted refusal for `face`, `identity_text`, `plate` kinds | `scripts/research/judge_protocol.py:203-204`, `:236-239`, `:250-276`, `:296-337` |
| Token estimate width x height / 750 from the PNG header | `judge_protocol.py:242-247` |
| Local VLM backend refuses a non-local endpoint; temperature 0; records truncation | `scripts/research/judge_backends.py:34-97` |
| Sheet-file backend writes `pending/<sheet_id>.png` and reads `verdicts/<sheet_id>.json` | `judge_backends.py:100-120` |
| Write-ahead budget, cache by `(png_sha256, model, prompt_version)`, privacy check before reservation | `scripts/research/judge_run.py:41-60`, `:208-234` |
| Requests are the sorted manifests, cut by `--limit` | `scripts/research/judge_sheetdir.py:178-187` |
| Wilson interval, sheet-level bootstrap, Cohen's and Fleiss' kappa (own implementations, unit-tested) | `scripts/research/judge_stats.py:17-87` |
| Agreement report: cluster, pair, judge-vs-judge, layout calibration, controls, order effect, disagreements | `scripts/research/judge_agreement.py:79-381` |
| Layout rule: smallest `(cells, thumb_px)` within 0.05 kappa of the best; no degeneracy guard | `judge_agreement.py:41`, `:209-212`, `:257-267` |
| Planted-control batch check: false accept and false reject at most 0.2 | `judge_agreement.py:42`, `:298-315` |
| Pools: COCO (via the API of a project, read only), Open Images vehicles, LFW faces; each row records `source` | `scripts/research/judge_pool.py:18-96` |
| Research tests | `tests/test_contact_sheet.py` (18 tests), `tests/research/test_judge_*.py` and `tests/research/test_visual_judge_cli.py` (46 tests); one of them pins the current layout rule: `tests/research/test_judge_agreement.py:148` |

Not present anywhere: a human-review path, an `anthropic` SDK client, Message Batches, or
`count_tokens`. The research requirements files (`scripts/research/requirements.txt`,
`scripts/research/fine_grained_requirements.txt`) do not list `anthropic`.

### 2.3 The label check (FG-0)

| Fact | Where |
|---|---|
| One sheet per claimed model (50 sheets), the review-sample images plus one same-make wrong model (when the make has another model) and one other-make image; the reviewer is told the claimed make and model | `scripts/research/fine_grained/fg0.py:1-8`, `:46-65`, `:68-79` |
| Answers yes / no / unsure per cell for "model" and "make"; strict JSON | `fg0.py:95-121` |
| Bars: model-level accuracy at least 0.95 on decided rows; the reviewer is "readable" only if it rejects at least 0.80 of the same-make plants | `fg0.py:29-30`, `:129-174` |
| Runner: `prepare` writes `exchange/pending/fg0_NN.png` and prompts, truth outside; `collect` scores | `scripts/research/fine_grained/fg0_run.py:17-77` |
| `collect` hard-codes the reviewer as machine review | `fg0_run.py:68-69` |
| Registered bar: "checked by a person" | `docs/research/fine_grained_benchmark_plan.md:104` |
| Result so far: machine review only, INCONCLUSIVE; decided-row model accuracy 0.971, unsure 29.6 %, the machine reviewer accepted a quarter of same-make plants | `docs/research/fine_grained_benchmark_results.md:58-87` |
| Data: Commons set C, 2,943 images, per-file licence recorded (public, redistributable) | `scripts/datasets/manifests/commons_set_c.json`; `fine_grained_benchmark_plan.md:25` |

### 2.4 Raw E0 outputs exist locally (read only, gitignored)

`artifacts_local/bench/cluster_search/w8/` in the main checkout holds `e0_coco`, `e0_oi`, `e0_lfw`,
`cal_coco`, `cal_oi`, the pools (`pool_coco`, `pool_lfw`, and the vehicle pool), the per-judge
JSONL runs, the budgets and the local-VLM verdict caches. So every rebuild in this plan is CPU
only and needs no stack: `verify` rebuilds from `pool.jsonl` crops (`visual_judge.py:273-291`).
Do not move or prune these directories; they back the E0 note.

### 2.5 Measured E0 numbers this plan starts from

From `docs/research/visual_judge_e0.md:65-122`:

| | Haiku, COCO crops | Haiku, OI vehicles | Local 4B, COCO | Local 4B, vehicles | Local 4B, faces |
|---|---|---|---|---|---|
| Cluster sheets | 55 | 55 | 210 | 217 | 220 |
| Kappa with truth (same/different) | 0.697 | 0.298 | 0.400 | -0.009 | 0.519 |
| Flag precision | 0.827 | 0.454 | 0.167 | 0.110 | 0.298 |
| Pair kappa | 0.821 | 0.611 | 0.031 | 0.000 | 0.051 |
| Bars (pair kappa >= 0.7, flag precision >= 0.8) | pass | fail | fail | fail | fail |

The second local catalog model has no scored run (its replies were truncated at 600 output
tokens; the default is now 2,048, `visual_judge.py:406`; `visual_judge_e0.md:124-134`).

---

## 3. Requirement 1: a non-vacuous layout calibration

### 3.1 Balanced sheets

**Recommended.**

- `make-calibration` gets `--clean-fraction` (default 0.5) and draws clean or contaminated by a
  seeded **permutation of exact counts**, not by a coin per case: with `n` clusters, exactly
  `round(n * clean_fraction)` are clean. Contaminated cases draw `n_bad` from the E0 counts
  `BAD_COUNTS` (`visual_judge.py:48`), so the contamination mix is one constant for E0 and
  calibration.
- Keep the half-hard, half-easy contaminant draw (`judge_cases.py:113`) and record `bad_kind` in
  the truth row (calibration truth rows do not carry it today, `visual_judge.py:150-160`).
- Record `clean_fraction` in `spec.json` (`_spec`, `visual_judge.py:54-63`).

**Rejected: keep the 1/5 draw and add more clusters.** More sheets at the same imbalance give a
tighter interval around a degenerate answer.

### 3.2 A layout rule that can say "not distinguishable"

**Recommended** changes to `judge_agreement.calibration` and `recommend_layout`:

1. **Sample guard.** If the truth clean share across calibration cases is outside [0.4, 0.6],
   return `recommended: null, reason: "unbalanced sample"`.
2. **Verdict guard.** If at any layout the judge's "same" share is 0 or 1, that layout's kappa is
   reported but marked `degenerate`; if every layout is degenerate, return
   `recommended: null, reason: "judge answered one way at every layout"`.
3. **Floor.** A layout is eligible only if its kappa with truth reaches the RQ-8 bar (0.7, the
   same number as `research_clustering_plan.md` 4.6, row RQ-8). No eligible layout means
   `recommended: null, reason: "no layout reaches 0.7"`.
4. **Paired comparison instead of a fixed 0.05 tolerance.** For each eligible layout L against
   the best layout B, compute the per-case accuracy difference (same case, both layouts) and its
   bootstrap 95 % interval over cases. L is "not worse" when the interval's lower bound is above
   -0.05. The recommendation is the smallest not-worse layout. Use the existing sheet-level
   bootstrap helper (`judge_stats.cluster_bootstrap`) or `scipy.stats.bootstrap`
   ([SciPy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html))
   with `paired=True`; do not write a third bootstrap.
5. **No recommendation keeps the current default.** The note then says "6 x 4 at 160 px stays; the
   calibration could not separate layouts at n = ...", which is a result, not a failure.

**Rejected: lower the floor until something is eligible.** That re-creates the vacuous pick.

**Rejected: choose the layout by token cost alone.** A 3 x 3 sheet costs fewer tokens per call
but shows 9 members instead of 24; cost per judged member is about the same (width x height / 750
scales with cell count). Cost is reported beside the rule, not used as the rule.

### 3.3 Calibration design per judge (see J3)

Recommended (counts in sheets, one call each):

| Judge | Kind | Cases | Layouts | Sheets | Why |
|---|---|---|---|---|---|
| Haiku | COCO crops (object class) | 50 (25 clean) | 3: 3x3@160, 4x4@160, 6x4@160 | 150 | the only kind where Haiku said "same" at all and passed the E0 bars; three grids, one thumb size, so 50 cases fit the cap |
| Local VLMs | COCO crops | 50 | all 5 of `CALIBRATION_LAYOUTS` | 250 per model | local calls cost GPU time only |
| Local VLMs | OI vehicles (vehicle type) | 50 | all 5 | 250 per model | only after the human adjudication (section 5.3) says the labels are usable at this level; otherwise skipped |
| Local VLMs | LFW faces (same person) | 30 | 3 grids | 90 per model | faces never go to a hosted model (`judge_protocol.py:204`) |

Thumbnail size (128 / 160 / 192) is calibrated with the local VLMs only; for Haiku it stays 160
unless the local runs show a size effect larger than the paired interval.

### 3.4 A confound the note must state

Synthetic contaminants always have the largest distances (`judge_cases.py:115-120`), and the
selector fills the outlier slots first (`contact_sheet.py:125`, `:142-144`). A 3 x 3 sheet has 2
outlier slots, a 6 x 4 sheet 3. So on small grids nearly every shown contaminant is an outlier
slot, and the share of contaminated cells per sheet differs by layout. Report the shown
contamination share per layout next to the kappa; do not read a layout effect that is explained by
it. Real clusters do not guarantee that contaminants are the farthest members, so the layout
chosen here transfers to real clusters only in direction.

---

## 4. Requirement 2: the blind human re-judge

### 4.1 What "blind" must mean, concretely

The person sees: a PNG with numbered cells and the same question the model saw (the prompt text
from `render_prompt`, `judge_protocol.py:250-276`). The person never sees, before finishing:
the sheet id, the case id, the layout name, the sample role (random, targeted, control, twin),
the truth, any judge's verdict, or the disagreement list.

### 4.2 Design

**Recommended:** a new module `scripts/research/judge_human.py` (a new concern; `visual_judge.py`
is 420 lines and already holds five subcommands), with three subcommands:

```text
judge_human.py draw     --sheets DIR [--sheets DIR ...] --per-kind N --targeted M --controls K \
                        --seed S --out HUMAN_DIR
judge_human.py export   --plan HUMAN_DIR/plan.json --exchange EXCH_DIR
judge_human.py collect  --plan HUMAN_DIR/plan.json --exchange EXCH_DIR --rater RATER_ID
```

- `draw` (run before any new judge verdict is read, so the random half does not depend on
  verdicts): per kind, a seeded uniform draw of `N` sheets from the judged sheet set, plus `M`
  targeted sheets, plus `K` planted controls. The targeted half is the adjudication sample of
  5.3 and is drawn after the judges ran (a second `draw --targeted-only` call). Writes
  `plan.json` (outside the exchange directory): sheet id, alias, sample role, kind, source
  directory.
- **Aliases.** `alias = sha256(f'{salt}:{sheet_id}')[:12]` with a per-plan random salt stored only
  in `plan.json`. The exported order is shuffled by the seed, and the person sees sheets of all
  roles interleaved.
- `export` writes `EXCH_DIR/<alias>.png`, `EXCH_DIR/<alias>.prompt.txt` and one
  `EXCH_DIR/answers.csv` with one row per alias and empty answer columns. Nothing else goes into
  `EXCH_DIR`. Re-ordered twins and several layouts of one case are never both in one plan (one
  member per case), so a person cannot recognise a repeat.
- `collect` maps aliases back, converts each CSV row into the JSON verdict shape and validates it
  with the existing `parse_verdict` (`judge_protocol.py:329-337`); an invalid row is recorded as
  invalid, not dropped (the harness rule, `visual_judge_e0.md:26-27`). It appends records to
  `HUMAN_DIR/judge/human-<rater>.jsonl` in the same record shape `run_requests` writes
  (`judge_run.py:180-191`), with `model = "human:<rater>"` and `backend = "human"`, so
  `judge_agreement.py report --runs` reads human runs exactly like model runs.

CSV columns:

| Layout | Columns |
|---|---|
| cluster | `alias, same (y/n), odd_cells (space-separated numbers), name, confidence (high/medium/low), seconds` |
| pair | `alias, row1 ... row12 (yes/no/unsure), seconds` |

`seconds` is optional and gives the person-time per sheet for the cost table.

**Rejected: let the person work in the judge's exchange directory.** File names leak roles
(section 1) and the directory may already contain the judge's verdict files
(`judge_backends.py:111-114`).

**Rejected: a web UI in Cropwright for this issue.** The review screens of
`cluster_search_makemodel_plan.md` 8.4 belong to the product (project-scoped, backend-served
records) and are much larger than a research re-judge; building them to collect about 150
answers inverts the cost. Recorded as a later option, not part of #406.

J1 lists Label Studio and a static HTML form as the alternatives to the CSV.

### 4.3 Sample sizes (see J4)

The registered rule is "a seeded 5 % of verdicts, at least 20 per run"
(`research_clustering_plan.md` 4.4 step 6). On E0's Haiku samples (55 cluster sheets per kind),
5 % is 3 sheets, so the floor of 20 decides. Kappa from 20 binary answers has an interval roughly
0.3 wide, so a point estimate of 0.7 says little.

Recommended per kind:

| Kind | Random (for judge-vs-human kappa) | Targeted (adjudication, reported apart) | Planted controls | Who sees it |
|---|---|---|---|---|
| COCO crops, cluster sheets | 40 | 10 | 4 (2 same, 2 different) | owner or delegate |
| COCO crops, pair sheets | 10 sheets (120 rows) | 0 | 0 | owner or delegate |
| OI vehicles, cluster sheets | 30 | 20 (judge said different, truth said same) | 4 | owner or delegate |
| LFW faces, cluster sheets | 30 | 0 | 4 | local only; owner or delegate on this machine |

About 150 sheets. At roughly 30 to 60 s per cluster sheet and 90 s per pair sheet, about 2 h of
person time. Report kappa with a bootstrap interval; a kind whose lower bound is below 0.5 is
INCONCLUSIVE regardless of the point value.

### 4.4 What the human pass measures

| Comparison | Bar or use | Source of the bar |
|---|---|---|
| Human vs each judge, random half, per kind | kappa >= 0.7 (RQ-8) | `research_clustering_plan.md` 4.6 |
| Human vs truth, synthetic sheets (truth exists) | reported; below 0.7 means the labels are not crisply judgeable at that level (section 5.3) | new, registered in R0 |
| Human on planted controls | false accept and false reject at most 0.2, else the human batch is redone after a short rest (same `CONTROL_MAX_ERROR`, `judge_agreement.py:42`) | existing constant |
| Human cell flags vs truth | flag precision and recall, as for judges | `judge_agreement.py:113-116` |

### 4.5 Test-retest for the hosted judge

The current API gives no temperature control (the Messages API rejects `temperature` from API
version `2026-10-21`; checked in the Claude API reference bundled with Claude Code on
2026-11-04), and the sheet-file backend runs through subagents. So Haiku's verdict on one sheet may vary
between calls. Recommended: 25 COCO sheets sent a second time in a fresh context (new aliases,
bypassing the verdict cache with a distinct run id and model label suffix `-retest`), reporting
self-agreement kappa. A self-agreement below 0.8 is stated next to every Haiku number.

---

## 5. Requirement 3: judges per kind

### 5.1 Hosted judge (Haiku), within the cap

Recommended allocation of at most 300 sheets (OQ-3 of `research_clustering_plan.md` 7.3):

| Use | Sheets |
|---|---|
| Balanced COCO calibration (3.3) | 150 |
| COCO E0 top-up to the plan's full allowance (45 more cluster sheets, 13 more pair sheets) | 58 |
| Test-retest (4.5) | 25 |
| Reserve: OI vehicles at a corrected level, only if 5.3 says the labels are usable | 67 |

Same seed and pools as E0 (`visual_judge_e0.md:170-188`), new sheet directories
(`e10_coco_cal`, `e10_coco_topup`), never overwriting `e0_*`.

**Subagent assignment rule (for the sheet-file path).** Each subagent gets at most one sheet per
case (never two layouts of one case, never a sheet and its twin) and only aliased file names. The
E0 note does not record how ids were split over its six subagents; if one subagent saw several
layouts of a case, the E0 flag-overlap and order-effect numbers are not independent. State that
limit in the new note.

### 5.2 Local VLM judges (RW-3, GPU)

| Run | Sets | Notes |
|---|---|---|
| Rerun of the second catalog entry that E0 could not score (the catalog's default entry) at 2,048 output tokens | e0_coco, e0_oi, e0_lfw, the new balanced calibration sets | the truncation cause and fix are in `visual_judge_e0.md:124-134` |
| Qwen3-VL-4B on the new balanced calibration sets | e10_coco_cal, e10_oi_cal (if run), e10_lfw_cal | E0 sets already judged; caches are reused (`judge_run.py:210-215`) |
| Optional (J7): one larger catalog entry, `qwen3-vl-8b` (Apache-2.0, about 26 GB budget, `examples/vlm/catalog.tsv`) | same sets | only if both small models fail the COCO bars |
| Smaller pair sheets (6 pairs per sheet) for local judges | e0_coco pairs | E0 suggested the tall 2 x 12 sheet may not suit a small model (`visual_judge_e0.md:150-151`); add `--pairs-per-sheet` to `make-e0` (`judge_cases.pair_rows` already takes `n_pairs`, `judge_cases.py:125-131`) |

Serving: the repo's `vlm` service under its own compose project (`-p op-rs-rw3`), selected with
`./openprocessor vlm use <id>` and `vlm apply` (`visual_judge_e0.md:32-34`), on the GPU agreed
with the peer sessions (J9). Serialize with the shared GPU lock (`docs/research/CLAUDE.md`, GPU
etiquette). Stop the VLM after the run.

### 5.3 Vehicles: judge error or label noise?

E0 found Haiku rejects 13 of 17 clean vehicle sheets and suggested label overlap (car, van, truck,
bus) as a cause (`visual_judge_e0.md:140-144`). The human pass decides it:

- The 20 targeted vehicle sheets (truth "same", judge "different") are judged blind by the
  person, mixed into the random sample.
- If the person also calls most of them "different" (human-vs-truth kappa below 0.7 on vehicles),
  the "vehicle type" labels are not a usable truth at that level. Then: no more Haiku spend on
  vehicle type; vehicle-type judging is "rank only"; make-level judging on set C stays as FG-7 has
  it (make PASS, model FAIL, `fine_grained_benchmark_results.md:254`).
- If the person agrees with the truth, the judge is wrong on vehicles, and the reserve of 5.1 may
  be spent on one balanced vehicle calibration with a revised prompt (prompt `p2`, registered in
  R0 before the run, see 9). One revision only.

### 5.4 Privacy constraint for every new sheet set

Face, identity-text and plate sheets never go to a hosted judge (`judge_protocol.py:204`,
`:236-239`, checked before any budget reservation at `judge_run.py:216`). For every new sheet
directory, the privacy kind must follow the **provenance of the pool** (`source` in `pool.jsonl`,
written by `judge_pool.py`), not only the operator's `--kind` text: building a sheet set whose
pool rows come from a face or plate source under a non-private kind must be refused, and a test
must show it. The human pass on faces happens on this machine only, with the exchange directory
under `artifacts_local/` (gitignored).

---

## 6. Requirement 4: the label check (FG-0) by a person

**Recommended.**

- Reuse `fg0_run.py prepare` unchanged for the sheets and truth. It already keeps roles outside
  the exchange directory (`fg0_run.py:17-38`). Sheet ids there are `fg0_01` to `fg0_50`
  (`fg0.py:64`) and carry no role, so aliasing is not needed; plants are inside each sheet and the
  person is not told which cells they are.
- Add `--reviewer` to `collect` (`machine:<label>` or `person:<rater>`), replacing the hard-coded
  string at `fg0_run.py:68-69`. No default: a missing reviewer is an error, so a machine run is
  never recorded as a person's.
- The person fills a CSV (one row per cell: `sheet, cell, model (yes/no/unsure), make
  (yes/no/unsure)`); a converter writes `exchange/verdicts/fg0_NN.json` in the schema
  `fg0.parse_verdict` validates (`fg0.py:95-121`). Same CSV-first choice as J1.
- Scoring is unchanged (`fg0.score`, `fg0.py:129-174`): PASS needs model-level decided accuracy
  >= 0.95 **and** the person must reject at least 0.80 of the same-make plants (readability).
  The strict reading (unsure counts as not confirmed) is reported beside it.
- The person may consult public reference photos of a model (the question is about labels, not
  about the person's memory); "unsure" is the honest answer for interiors, badges and detail
  shots (about 30 % of the sample, `fine_grained_benchmark_results.md:80-83`).
- Person time: 50 sheets of 6 to 8 cells, about 2 to 3 min each, about 2 h.

What FG-0 decides: whether the model-level numbers of wave WB stop being "provisional"
(`fine_grained_benchmark_results.md:85-87`). It does not reopen the encoder decision; WB2 already
replaced set C for that (`docs/research/wb2_standard_benchmarks.md` section 9.4).

---

## 7. Requirement 5: hosted-judge client (see J2)

Two paths, both documented:

| Path | What it is | Cost and control | Status |
|---|---|---|---|
| Sheet-file (exists) | A Haiku subagent started with the Agent tool views pending PNGs and writes JSON (`visual_judge_e0.md:36-45`) | No API key; tokens include the subagent's own context (about 580k for 184 sheets, `visual_judge_e0.md:162-163`); the model is whatever the harness maps "haiku" to, so the run must record the actual model id it reports | works |
| Message Batches (planned in W8, not built) | `client.messages.batches.create(requests=[{custom_id, params}])`, poll `processing_status` until `ended`, read results keyed by `custom_id` (results arrive in any order); 50 % of the standard price; strict JSON through structured outputs (`output_config.format`); token budget checked with `messages.count_tokens` before submit | needs `ANTHROPIC_API_KEY` or an `ant auth login` profile, never logged; model pinned as `claude-haiku-5-5` ($0.10 / $0.50 per million input / output tokens up to 100k-token prompts, per the Claude API reference bundled with Claude Code, checked 2026-11-04) | not built |

Docs: [Message Batches](https://platform.claude.com/docs/en/build-with-claude/batch-processing),
[vision and image tokens](https://platform.claude.com/docs/en/build-with-claude/vision),
[token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting),
[structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
Re-read them at execution; this note does not pin request shapes beyond the four calls named.

Recommended: build the batch backend only if the owner provides a key (J2). It goes in
`judge_backends.py` as `BatchBackend` (hosted = True, so the privacy refusal applies), keyed by
alias as `custom_id`, recording `input_tokens`, `output_tokens` and the request id; the budget
reserves from `count_tokens` instead of the pixel estimate. The `anthropic` SDK goes into
`scripts/research/requirements.txt` (research venv only), never into the product requirements
(owner decision O4: no product-side hosted verifier, `cluster_search_makemodel_plan.md:1360-1362`).

Either way, no Celery task is involved: the judge is research tooling, run by an operator from
the research venv. Heavy work in this issue is the local VLM run, which uses the existing `vlm`
service, not an API request handler.

---

## 8. Requirement 6: pre-registration and the write-up

- **R0 (docs only, before any new verdict is read):** a registration section at the top of
  `docs/research/visual_judge_calibration.md` with: the bars of 4.4 and 4.6 above, the layout
  rule of 3.2, the sample sizes of 3.3 and 4.3, the vehicle decision rule of 5.3, the prompt
  version (`p1`, and the exact `p2` text if 5.3 allows a revision), the stop rule of section 9.
  Committed as `docs(research): register the judge calibration bars (#406)`.
- **Results** appended below the registration in a later commit, never editing the registered
  part (the WB2 pattern, `docs/research/wb2_standard_benchmarks.md:3-6`).
- **Backing data:** the small report JSONs (`report_*.json`, human run JSONL without image paths,
  calibration tables) in `docs/benchmarks/visual_judge_calibration/` with `SHA256SUMS`; sheets,
  crops and face material stay in `artifacts_local/` (faces never leave it).
- **Pointers:** a one-line "superseded for layout and human agreement by ..." note at the end of
  `visual_judge_e0.md`'s "Open issues" section; rows in `docs/research/README.md` and the research
  table of `docs/design/README.md` (`docs/research/CLAUDE.md`, "Adding a note", step 3);
  `fine_grained_benchmark_results.md` gets an appended "FG-0 by a person" section with the
  verdict (append, not rewrite). Update `docs/WHITEPAPER.md` lines that say the re-judge was not
  done (`docs/WHITEPAPER.md:482`, `:556`, `:566`) once results exist.
- **Docs check:** `.venv/bin/python scripts/docs/check_docs_vs_code.py` on the changed files; the
  naming-leak scan applies (the default catalog VLM entry is named in tracked files only where
  the allowlist already allows it; in the note, call it "the catalog's default entry").

---

## 9. Stop rules

| Line | Stop when | Then |
|---|---|---|
| Layout | the balanced calibration returns "not distinguishable" or no layout reaches 0.7 | 6 x 4 at 160 px stays the default; no second calibration |
| Haiku, per kind | pair kappa < 0.7 or flag precision < 0.8 on the topped-up sample, or judge-vs-human kappa < 0.7 | Haiku ranks sheets for a person on that kind; no prompt tuning beyond the one registered revision |
| Vehicles | human-vs-truth kappa < 0.7 at vehicle-type level | labels at that level are declared unjudgeable; no further vehicle-type spend |
| Local VLM | the rerun of the second catalog entry and (if J7) one larger entry both fail the COCO bars | local judges are rank-only on crops; faces keep the local judge as a ranking aid (it never accepted a wrong identity, `visual_judge_e0.md:147-149`) and a person decides |
| FG-0 | the person's verdict is PASS, FAIL or INCONCLUSIVE (unreadable) | recorded; no second person pass unless the owner asks |
| Issue | all of the above recorded | close #406 with the note; the research plan's RQ-8 row points to it |

---

## 10. Judgment calls

| ID | Question | Options | Recommendation and trade-off |
|---|---|---|---|
| J1 | How does a person enter answers? | (a) CSV beside aliased PNGs, any image viewer; (b) a self-contained static HTML form (inline PNGs, radio buttons, cell checkboxes, a download of the answers) generated by the script; (c) [Label Studio](https://labelstud.io/guide/) (Apache-2.0, maintained) with an image + choices + text template | **(a)**. Zero UI code, stdlib `csv`, the same pattern FG-0 already uses (`review_sample.csv`). (b) is friendlier for 150 sheets but is custom UI code. (c) is the maintained library, but needs a server, an account and an import/export adapter to the verdict schema, and its cell-number grid is a free-text field anyway; worth it only if human review becomes routine |
| J2 | Hosted path | (a) keep the sheet-file subagents; (b) build the Message Batches backend | **(a) unless the owner provides an API key**, then (b). (b) pins the model id, records real token counts, halves the price and removes the subagent-context overhead; (a) needs no key but cannot pin the model or count tokens |
| J3 | Calibration design for Haiku | (a) 3 grids x 50 cases (150 sheets); (b) 5 layouts x 30 cases (150 sheets, as E0) | **(a)**. 30 cases cannot separate layouts; three grids at one thumb size answer the question that matters (how many cells); thumb size goes to the local judges |
| J4 | Human sample size | (a) registered 5 %, at least 20 per run; (b) section 4.3 (about 150 sheets) | **(b)**. 20 answers give a kappa interval about 0.3 wide. Cost: about 2 h of person time instead of about 40 min |
| J5 | Who re-judges | owner, or a named delegate | owner's call; the delegate must not have seen the E0 disagreement lists; faces only on this machine |
| J6 | Vehicle fallback if the labels are the problem | (a) rank-only at vehicle type, stop; (b) build a cleaner vehicle truth (person-checked subset) | **(a)**. (b) is a labelling project, not judge calibration; make level is already covered by FG-7 |
| J7 | Third local model | (a) stop after the rerun; (b) add `qwen3-vl-8b` | **(b) only if both small models fail COCO**; costs about 1 more GPU hour and 26 GB on one card |
| J8 | Whole-image sheets (the research plan lists "whole images" as a kind for E-10) | (a) defer to #436 / RQ-10; (b) build a whole-image pool now | **(a)**. No whole-image pool or truth definition exists (`judge_pool.py` has COCO crops, OI vehicles, LFW only); RQ-10 is priority 4 |
| J9 | GPU and window for the local runs | GPU 0 or GPU 2, in RW-3 after the v0.7.0 tag (`research_clustering_plan.md` 5.1, 5.2); research windows are outside the program's three GPU windows (`docs/design/program/00_index.md` 8.1) | owner's call with the peer sessions; never GPU 1 |
| J10 | Where results go | (a) successor note `visual_judge_calibration.md`; (b) append to `visual_judge_e0.md` | **(a)**. Research notes are dated snapshots (`docs/research/CLAUDE.md`); E0 keeps a pointer |
| J11 | Haiku model | current `claude-haiku-5-5` for the batch path; for subagents, whatever the harness maps "haiku" to | record the id in every run header either way; a mixed-model sample is reported per model |

Owner time and budget questions already open in the research plan stay there: OQ-3 (Haiku
budget), OQ-7 (owner time). This plan assumes the recommended answers (yes, about 300 sheets;
yes, owner time).

---

## 11. File-by-file change list

### 11.1 Research harness (`scripts/research/`)

| File | Change |
|---|---|
| `visual_judge.py` | `make-calibration`: `--clean-fraction` (default 0.5), exact-count seeded permutation, contamination from `BAD_COUNTS`, `bad_kind` in truth rows, `clean_fraction` in the spec; `--layouts` to pick a subset of `CALIBRATION_LAYOUTS` (for J3). `make-e0`: `--pairs-per-sheet` (default 12). All `make-*`: refuse a pool whose `source` is a face or plate source unless `--kind` is a private kind (5.4) |
| `judge_agreement.py` | `calibration`: sample guard, verdict guard, floor, paired comparison, `recommended` may be null with `reason`; shown-contamination share per layout (3.4). New `human_agreement` block: human vs each judge on common sheets, per kind and per sample role, kappa with bootstrap interval; human vs truth where truth exists; adjudication table for targeted sheets (did the human side with truth or judge) |
| `judge_human.py` (new) | `draw`, `export`, `collect` (section 4.2); aliasing; CSV to verdict conversion through `parse_verdict`; records in the run JSONL shape |
| `judge_sheetdir.py` | helper to load several sheet directories for `draw` (manifests and truth keyed by `(dir, sheet_id)`, so equal ids in two directories cannot collide) |
| `judge_backends.py` | `SheetFileBackend` writes aliased names from a caller-supplied alias map instead of `sheet_id` (keeps the role leak out of hosted runs too). Optional `BatchBackend` (J2) |
| `judge_protocol.py` | `PROMPT_VERSION` stays `p1`; bump to `p2` only with the registered revision of 5.3 (the cache key includes it, `judge_run.py:210`) |
| `judge_stats.py` | kappa bootstrap helper (resampling sheets) if `scipy.stats.bootstrap` is not used directly; no new kappa implementation |
| `fine_grained/fg0_run.py` | required `--reviewer`; CSV converter subcommand `from-csv` |
| `requirements.txt` | `anthropic` only if J2 (b) |

### 11.2 Tests (`tests/research/`)

| File | New or changed test |
|---|---|
| `test_visual_judge_cli.py` | calibration clean share is exactly 0.5 (within one case) for n = 10, 30, 50; calibration truth rows carry `bad_kind`; `--pairs-per-sheet 6` gives 6-row pair sheets; the provenance rule of 5.4 (a private-source pool is refused under a non-private kind) |
| `test_judge_agreement.py` | replace the test at `:148` with: all-"different" verdicts over a 20 % clean sample return `recommended: null` and the reason; an unbalanced truth sample returns `unbalanced sample`; a layout below 0.7 is not eligible; the paired rule picks the smallest not-worse layout on a constructed case where the fixed-0.05 rule would pick a different one |
| `test_judge_human.py` (new) | exported file names contain no substring of any sheet id, case id, `ctl`, `cal`, `-r` or layout string; the exchange directory holds only PNG, prompt and `answers.csv`; no case appears twice in one plan; `collect` maps aliases back, rejects an inconsistent row (`same = y` with odd cells) as invalid and keeps it; human records load through `judge_agreement.load_runs` |
| `test_fine_grained_data.py` | `fg0_run collect` without `--reviewer` fails; CSV conversion round-trips through `fg0.parse_verdict` |
| `test_judge_protocol.py` | (if J2 b) the batch backend refuses a face sheet before any request object is built, using a fake client |

### 11.3 Docs and data

| File | Change |
|---|---|
| `docs/research/visual_judge_calibration.md` (new) | R0 registration, then results (section 8) |
| `docs/benchmarks/visual_judge_calibration/` (new) | report JSONs, `SHA256SUMS` |
| `docs/research/visual_judge_e0.md` | one appended pointer line under "Open issues" |
| `docs/research/fine_grained_benchmark_results.md` | appended "FG-0 by a person" section |
| `docs/research/README.md`, `docs/design/README.md` | rows for the new note |
| `docs/design/research_clustering_plan.md` | RQ-8 row and E-10 row point to the note (one line each) |
| `docs/WHITEPAPER.md` | update the three "not done" lines after results |

No product code changes: `src/services/labeling/contact_sheet.py` is used as is. No contracts, no
routes, no env vars, no CHANGELOG entry beyond a `docs(research)` line if the project's changelog
rules ask for research notes (check `CHANGELOG.md` conventions at execution).

---

## 12. How we prove it: verification plan

### 12.1 Static and unit (CPU, per lane: targeted tests only)

Each new test in 11.2 is written first and watched fail against the current code:

| Test | Red today because |
|---|---|
| calibration clean share 0.5 | `visual_judge.py:135` gives about 0.2 |
| degenerate calibration returns null | `recommend_layout` returns the smallest layout (`judge_agreement.py:209-212`) |
| aliases leak nothing | there is no alias path; sheet-file names are sheet ids (`judge_backends.py:117-118`) |
| privacy kind follows pool provenance | new rule (section 5.4); the test is written before the check |
| `fg0_run collect` needs a reviewer | the reviewer string is hard-coded (`fg0_run.py:68-69`) |

Commands (lane level, no full suite):

```bash
.venv/bin/python -m pytest tests/research/test_visual_judge_cli.py tests/research/test_judge_agreement.py \
  tests/research/test_judge_human.py tests/research/test_fine_grained_data.py -q --no-cov
.venv/bin/pre-commit run --files <changed files>
```

### 12.2 Empirical replay on existing raw data (CPU, no judge calls)

Before any new judge call, recompute the E0 calibration reports with the new rule from the
existing JSONL runs:

```bash
.venv/bin/python scripts/research/judge_agreement.py report \
  --sheets artifacts_local/bench/cluster_search/w8/cal_oi \
  --runs artifacts_local/bench/cluster_search/w8/cal_oi/judge/*.jsonl
```

Expected: `cal_oi` for Haiku and Qwen3-VL-4B returns `recommended: null` with "judge answered one
way at every layout" (today it returns 3 x 3). For `cal_coco` (Qwen3-VL-4B, kappas 0.52 to 0.76),
the note records what the paired rule says and whether 30 cases separate anything. This is the
red-to-green proof on real data, and it costs nothing.

### 12.3 Rebuild and verify

Every new sheet directory passes `visual_judge.py verify` (byte-identical rebuild) before any
call. Every human plan passes a leak check: list `EXCH_DIR` and assert only aliases.

### 12.4 Judge runs

Per run, the report must show: planted controls `batch_ok` true (else the batch is discarded,
`judge_agreement.py:298-315`); invalid replies counted; budget never exceeded (the write-ahead
budget file); the run header has git sha, seed, model id, prompt version.

### 12.5 Human pass

- Controls inside the human batch pass (4.4).
- The person confirms in writing (one line in the note) that the disagreement lists were not
  opened before `collect` ran.
- `judge_agreement.py report --runs <judge runs> <human run>` produces the human block.

### 12.6 One sweep

Per the program's rule, one offline sweep on the integration branch after the lanes merge:
`.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live' -n 8`, `pre-commit run --all-files`,
the docs check. No live harness run and no program GPU window is used.

---

## 13. Edge cases and traps (the failure register)

1. **A green calibration test with a vacuous fixture.** A test fixture where every judge verdict
   is "different" and the truth is 20 % clean passes today's layout test. The new tests must
   include exactly that fixture and expect null.
2. **Kappa of 0 is not "no information" in the code.** `cohens_kappa` returns None only when
   chance agreement is 1 (`judge_stats.py:63-65`). A judge with one answer against a varying
   truth returns 0.0, which a `> threshold` check treats as a real value. The verdict guard must
   look at the judge's answer share, not at None.
3. **Aliases that are deterministic without a salt** (for example a plain hash of the sheet id)
   can be reversed by hashing the guessable ids (`cl0001` ... `cl0200`). Use a random per-plan
   salt kept outside the exchange directory.
4. **Shuffled order but sorted file names.** If aliases are sorted when listed, the list order
   is the alias order, which is random; fine. If export numbers files `001.png ...` in plan
   order, the order leaks the draw order (random first, targeted second). Shuffle before
   numbering, or use aliases only.
5. **Two layouts of one case, or a twin, in one human plan.** The person would recognise the
   repeat. `draw` keeps one sheet per case; test it.
6. **The person reads `report_haiku.json` first.** Blindness is a procedure as much as code. The
   note records the order of steps and the person's confirmation.
7. **Human records cached as model verdicts.** The verdict cache key is
   `(png_sha256, model, prompt_version)` (`judge_run.py:210`). Human runs must not write into a
   judge's cache directory; give `collect` no cache argument.
8. **`--limit` takes the first N sorted ids** (`judge_sheetdir.py:187`). In E0 this mixed twins
   into the first 55 sheets. For the top-up, select by an explicit id list from a seeded draw,
   not by `--limit`, so the top-up does not re-send already judged sheets (the cache would answer
   them for free, but the counts in the note would be wrong).
9. **Test-retest silently answered from the cache.** A retest with the same model label and
   prompt version hits the cache and reports perfect self-agreement. Use a distinct model label
   suffix for the retest and assert zero cache hits in its run.
10. **Planted controls recognisable by content.** Control sheets with 8 easy contaminants
    (`visual_judge.py:49-50`, `hard_fraction=0.0` at `:226`) are visibly easier than real sheets.
    That is fine for a batch-health check but means control pass rates say little about hard
    sheets; do not cite control accuracy as judge accuracy.
11. **Shown contamination differs by layout** (3.4). A "smaller grid is better" reading may only
    reflect a higher contaminated-cell share.
12. **The vehicle level wording.** "vehicle type" in E0 (`visual_judge_e0.md:57`) is ambiguous
    for van vs car vs truck. If a `p2` prompt is registered, define the level in words (for
    example "the same body type: car, van, truck, bus, motorcycle") and treat the result as a new
    kind, not as the E0 kind.
13. **Privacy kind and pool provenance disagree.** Section 5.4 makes provenance win; test it with a private-source fixture.
14. **A human answer of "same" with odd cells listed.** `_cluster_verdict` rejects it
    (`judge_protocol.py:300-301`); the CSV converter must not "repair" it. It is an invalid row.
15. **Unsure pair rows.** They are excluded from accuracy and reported as a rate
    (`judge_agreement.py:153-179`). The same applies to human pair rows; a person with a high
    unsure rate is not a disagreement.
16. **FG-0 unsure rows.** About 30 % of the set C sample is unusable shots. The decided-row rate
    can pass while the strict rate is low; report both, as the machine run did.
17. **FG-0 reviewer string.** A person run recorded with the old hard-coded machine string would
    make the bar look met by a person when it was not, or the reverse. The flag is required.
18. **Subagents keeping context.** A Haiku subagent given 20 ids keeps the earlier sheets in its
    context. Give each subagent one sheet per case and, for the retest, a fresh subagent.
19. **Model drift during the run.** A hosted model id that changes mid-run splits the sample.
    The run header records the id per call; the report groups by it.
20. **Benchmark data.** Never overwrite `e0_*` or `cal_*` directories; new runs go into new
    directories. Do not prune `artifacts_local/bench/cluster_search/w8/`.
21. **The naming-leak scan.** The default catalog VLM's id contains a word that the domain scan
    bans outside allowlisted files (`scripts/codegen/check_naming_leaks.py:61`,
    `naming_leak_allowlist.txt` Scan B). New research code must take the served model name from
    the CLI (`--model`, as today), never as a literal.

---

## 14. Suggested phasing

| Phase | What | Needs | Gate to the next phase |
|---|---|---|---|
| P0 | R0 registration commit (docs only) | nothing | committed before any new verdict is read |
| P1 | Code: calibration balance and rule, `judge_human.py`, privacy-by-provenance, FG-0 reviewer flag, tests (red then green) | CPU | lane tests green; 12.2 replay shows null on `cal_oi` |
| P2 | Build sheets: `e10_coco_cal`, `e10_coco_topup`, `e10_lfw_cal` (and `e10_oi_cal` only later); human plan `draw` (random halves) | CPU | `verify` clean; leak check clean |
| P3 | Haiku runs (sheet-file, or batch if J2 b) | O3 budget | controls pass; budget not exceeded |
| P4 | Local VLM runs (RW-3) | GPU window agreed with peers | invalid-reply rate recorded; VLM stopped after |
| P5 | Human pass: `draw --targeted-only` (vehicle and COCO disagreements), `export`, person fills CSV, `collect` | owner or delegate, about 2 h | human controls pass |
| P6 | FG-0 by a person | about 2 h | `collect --reviewer person:<id>` |
| P7 | Report, note, benchmarks folder, pointers; one offline sweep; close #406 | CPU | docs check green |

P1 and P2 can run any time (no GPU). P3 and P6 do not need a GPU and may run before the v0.7.0
tag if the owner wants (they touch no program window); P4 waits for a research window (J9).

---

## 15. Relation to other issues

- **#346** (clustering tracker): one results row for E-10 once the note exists.
- **#399** (closed, consolidated here): FG-0 is its remainder.
- **#398, #416, #347, #365, #435, #436**: every quality claim that uses a judge reads the per-kind
  verdict of this issue first (`research_clustering_plan.md` 4.4 step 1).
- **Cropwright review screens** (`cluster_search_makemodel_plan.md` 8.4): not part of #406; if
  human review becomes routine, a separate issue.
