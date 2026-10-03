# Curation scores: validation results

Status: research snapshot (2026-09); measured on a CPU-only workstation against a ~348k-item embedding index (residual pool ~125k); numbers will drift.

Implemented in the public repo: yes. Scorers: `src/services/curation/item_scores/` (`uniqueness.py`, `near_dup.py`, `mistakenness.py`). Status registry and promotion rule (`VALIDATED_SCORERS`, `VIZ_PROJECTION_SHIP_MODE`): `src/services/curation/strategy_registry.py`. Diverse selection: `src/services/curation/selection/kcenter_greedy.py`. Projection overlay: `src/services/curation/embedding_viz.py`. Routes: `POST /scores/compute`, `GET /scores/status`, `GET /scores/coverage`, `POST /select/diverse`, `POST /viz/projection/rebuild`, `GET /viz/projection`. Flags (all default off, see `env.template`): `OP_SCORES_ENABLED`, `OP_SCORES_SHADOW`, `OP_SELECT_DIVERSE_ENABLED`, `OP_VIZ_PROJECTION_ENABLED`. Backfill script: `scripts/curation/backfill_scores.py`. The public `VALIDATED_SCORERS` has since grown to include `uniqueness` as well as `mistakenness`; this document records the state at the time of the first validation pass.

This document records what was measured for each curation-scoring method, why each was or was not adopted, and what remains open. A method that failed here should not be re-proposed without new evidence. Scores are additive metadata and orderings; none of them writes a cluster assignment (see [clustering_methods.md](clustering_methods.md)).

## 0. Data used

| Quantity | Value |
|---|---|
| Total indexed items | 347,837 |
| Residual pool (the pool every scorer runs over) | 124,920 |
| Human-validated items | 388 |
| Frozen holdout items | 0 |
| Distinct cluster buckets in the residual pool | 512 (the IVF default K) |
| Items with any linear-probe prediction field | 0 |
| Items with a label-change history and human source | 122 (92 show a genuine correction) |

Two environment gaps shaped the pass:

1. No persisted IVF centroid file existed on the test host even though the pool carried a completed IVF assignment. Uniqueness requires the centroid file (error if absent); near-duplicate degrades to an unscoped global pass. For validation only, centroid stand-ins were rebuilt in memory by mean-pooling and re-normalizing embeddings within each existing bucket, never written to disk. The same `compute_uniqueness` and `compute_near_dup_groups` math ran. Operators should run one real clustering pass to populate the store before relying on these scores.
2. No probe checkpoint had ever been trained, so no item had probe prediction fields. That blocks the margin-vs-entropy protocol and gives mistakenness no production input.

Writes to the live index were blocked by the sandbox, so every number comes from read-only queries plus in-process math on fetched embeddings. The production backfill had not been run end to end.

## 1. Representativeness

Not validated: the sort clause it depends on did not exist yet. It is a rename of distance-to-centroid, so no new scoring math is proposed. Deferred.

## 2. Uniqueness (k-NN density)

Protocol: Spearman rho between the uniqueness score and minus log label frequency; bar rho >= 0.25. Ran the production scorer over the full 124,920-item pool with k=16, nprobe=12 (the defaults).

90.2% of the pool (112,673 items) has no proposed label, so counting the blank string as a giant class gives a misleading number.

| Population | n | Spearman rho | Verdict |
|---|---:|---:|---|
| Full pool, blank label counted as a class | 124,920 | 0.042 | not the right test |
| Restricted to items with a proposed label | 12,247 | 0.384 (p ~ 0) | pass |

The restricted set has 77 proposed classes, from singletons to a class with 4,106 items. The real gate is an operator A/B (200 top-uniqueness vs 200 random items, blind-labelled, at least 1.5x rate of new-class or unmatched outcomes); it needs a human session and was not run. Verdict: pre-screen pass, A/B not executed, stays `shadow`.

## 3. Mistakenness (confident-learning margin)

Protocol: synthetic 5% label flip; AUROC >= 0.80 and precision@100 >= 0.50.

| n | mislabeled | AUROC | precision@100 |
|---:|---:|---:|---:|
| 500 | 25 | 1.0 | 0.25 (ceiling: only 25 positives exist) |
| 5,000 | 250 | 0.997 | 0.98 |

Both bars pass once the population is large enough for precision@100 to be meaningful; tests should use n=5,000 so they assert both bars. A real-data supplement found 92 genuine human corrections (for example one vehicle type corrected to another), too few for AUROC and blocked by the missing probe predictions. Verdict: adopt; promoted from `shadow` to `experimental`. Caveat for any UI: coverage reads 0% until a probe checkpoint is trained and run (`scripts/curation/run_probe.py`).

## 4. Near-duplicate (item level)

Protocol: threshold sweep {0.95, 0.96, 0.97, 0.98, 0.99}; bar at least 95% true-duplicate precision (50 manually judged pairs per threshold) and at least 99% distinct-label retention after collapsing each group to its most central member. Ran the production grouping over a 20,000-item sample, scoped by the existing bucket assignment.

| Threshold | groups | items in groups | % of sample | class-coverage retention |
|---:|---:|---:|---:|---:|
| 0.95 | 536 | 1,252 | 6.26% | 100.00% |
| 0.96 | 326 | 698 | 3.49% | 100.00% |
| 0.97 | 145 | 300 | 1.50% | 100.00% |
| 0.98 | 48 | 99 | 0.49% | 100.00% |
| 0.99 | 3 | 6 | 0.03% | 100.00% |

Retention passes everywhere (exactly computed). No human was available to judge pairs; as a proxy, paired items shared a bucket 100% of the time (true by construction of bucket scoping) and shared a class 100% except one pair at 0.98 (19/20), though most pairs were unlabeled so the proxy is weak. Verdict: retention passes, precision not executed, stays `shadow`. 0.99 is likely too conservative; 0.97-0.98 is the likely useful range pending a human spot check.

## 5. Uncertainty margin vs entropy

Cannot execute without probe predictions (0 of 347,837 items carried any). Next step: train a probe model, run the probe inference script, then compare true errors captured in the top-k by margin sort vs entropy sort against a small human-labelled holdout (bar: margin >= entropy).

## 6. Diversity (k-center-greedy)

Pre-screen: cohort A k-center-greedy (~5,000), B random (~5,000), C one representative per bucket (~5,000), from a 30,000-item sample; count distinct labels covered. Bar: A/B >= 1.3x.

| Cohort | Distinct labels covered (of 66) |
|---|---:|
| A: k-center-greedy | 57 |
| B: random | 38 |
| C: cluster representatives | 42 |

A/B = 1.50x (pass); C/B = 1.11x. A naive O(n*k) CPU implementation (cosine distance on unit-norm embeddings) took 93.6 s for k=5,000 over 30,000 items. The full gate (a training run plus evaluation on a frozen holdout, mAP50-95 improvement of at least 1.0 point) is a later, multi-hour GPU decision and was not run.

## 7. 2-d projection overlay (visualization only)

Protocol: fit a 2-d UMAP on a ~10k sample; for each point compute the fraction of its 10 nearest neighbours in the 2-d plane sharing its real cluster id. Bar: >= 0.30 ship plain; 0.15-0.30 ship with an "approximate" banner; < 0.15 do not ship. The projection state uses its own cache slot, separate from any clustering reducer.

| Run | n sampled | distinct buckets hit | mean 10-NN purity |
|---|---:|---:|---:|
| 1 (plain scroll) | 10,000 | 66 | 0.897 (discarded) |
| 2 (random-scored sample) | 10,000 | 561 | 0.472 (number of record) |
| 3 (independent random draw) | 10,000 | 554 | 0.468 |

Run 1 was discarded: a single-shard scroll returns insertion order, so the first 10,000 documents covered only 66 of ~590 buckets, and the inflated purity reflected little cluster diversity. Random-scored sampling reaches 554-561 buckets and a stable ~0.47. For run 2: median 0.50, p10 0.0, p90 1.0, and 22.1% of points have zero same-cluster neighbours. Mean purity 0.47 means about 4.7 of 10 neighbours share the cluster, enough for visual browsing and lasso selection but not to decide membership, which stays with IVF. Verdict: adopt, capped at `experimental` until an interactive-performance check exists. A re-measurement below 0.15 flips `VIZ_PROJECTION_SHIP_MODE` to `do_not_ship`, which forces `disabled` regardless of the flag.

## 8. Summary

| Method | Result | Bar | Verdict |
|---|---|---|---|
| Representativeness | n/a | n/a | deferred |
| Uniqueness | rho = 0.384 (labelled subset, n=12,247) | >= 0.25 | pre-screen pass; A/B not run; shadow |
| Mistakenness | AUROC 0.997, precision@100 0.98 (n=5,000) | >= 0.80 / >= 0.50 | pass; experimental |
| Near-duplicate | 100% retention at all 5 thresholds; precision proxy only | >= 99% / >= 95% | retention passes; shadow |
| Uncertainty margin | not executed | margin >= entropy | no probe checkpoint |
| Diversity | 1.50x | >= 1.3x | pre-screen pass |
| 2-d projection | purity 0.472 | >= 0.30 | pass; experimental cap |

A flag can only turn a validated method on; it can never revive a failed one.

## 9. Caveat for probe upgrades

The class-posterior extraction used for probe predictions patches a YOLO11 predictor to skip NMS and read the raw pre-NMS per-anchor tensor (4 + nc channels by anchors), pick one box with the same criterion NMS uses, and softmax its class-score row. A newer NMS-free YOLO generation exports a fixed `[N, 300, 6]` tensor (top-1 class and confidence per slot) with no per-class posterior, so this trick does not carry over. Moving the probe to such a model would require either exporting the one-to-many training head's raw output, or reading logits before final top-1 selection if the implementation retains them.

## 10. Open items

1. Run a real clustering pass to populate the centroid store.
2. Run the score backfill for real against the index.
3. Train a probe checkpoint and run probe inference.
4. Run the uniqueness operator A/B.
5. Manually judge near-duplicate pairs at 0.97 and 0.98.
6. Run the full training A/B for diverse selection.
