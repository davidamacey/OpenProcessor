# 15b: storage reads, every query shape on the record

| Field | Value |
|---|---|
| Status | plan, revision 0 (2026-11-15): the read map of plan 15 section 12, written after the gate A measurement at 1 and 10 million objects (8 clients held). Companion of [15_storage_plan.md](15_storage_plan.md) and [15a_storage_schema.md](15a_storage_schema.md); the cut-over slices that build it are [15c_storage_slices.md](15c_storage_slices.md) |
| Code base read | `int/storage` at `7cadf2968` (the integration code of `int/b4-stateless` plus storage step 1: `src/domain/claim*.py`, `fold*.py`, `results.py`, migration `0010_storage_record`, `src/db/repos/`, `src/services/record/apply.py`) |
| Checklist | the storage inventory of 2026-11-13 (`artifacts_local/speed_1113/storage_inventory.md`, gitignored), section 3: shapes 1 to 80 with file:line. Its paths are relative to `src/services/curation/` unless they start with another directory; its line numbers were re-checked on this tree (same code) |
| Measurement | gate A (`artifacts_local/speed_1113/v2_read_shapes/`, gitignored): `notes.md`, `queries_pg.sql`, `summary_pg.sql`, `raw/time_pg_1m.json`, `raw/time_os_1m.json`, `raw/time_pg_10m.json`, `raw/time_os_10m.partial.json` |
| Decided by the coordinator | page fetches are SQL on the record; exact totals only from summary keys, else a capped count; dashboards read summary tables kept by the writer; cluster placement is the side table; text substring search is a trigram index; the search engine leaves the curation stack (section 1.2 of the inventory stays until phase 3); the fallback for a pending measurement that fails is additive (a search copy without vectors for the failing shapes only) |

## 0. How to read this file

Section 1 lists the measured numbers this file uses and what is still running. Section 2 gives the
SQL of every distinct read pattern (P1 to P15), with real table and column names of `0010`. Section 3
maps all 80 shapes, one row each. Sections 4 to 10 are the designs the rows point at: the list cursor,
the capped-count rule, the summary tables and how `apply()` keeps them, the sort catalogue, the
trigram index, the box-row index and the job readers. Indexes are numbered `X1`..`X25`; their DDL is
in 15a section 2.6 (migration `0011_storage_indexes`), summary tables in section 6 here.

"to be measured" means: no cell of gate A times this query on this schema; the first wave sweep after
the slice that builds it times it on set P and on the 10 million spike tables (section 1.3). Names of
new modules are given where a slice creates them; everything else is a name in the code.

## 1. What gate A measured, and what waits

### 1.1 Cells used (PostgreSQL 18 and OpenSearch 3.9, 8 clients, p50 / p99 ms)

Notation `pg10 s1a_deep` = `raw/time_pg_10m.json`, cell `s1a_deep@8`; `os10` = `raw/time_os_10m.partial.json`;
`pg1`, `os1` the 1 million files. Bars (plan 15 section 6): page shapes 50 / 200, dashboards 200 / 800.

| Shape of the spike | Query (`queries_pg.sql`) | pg10 | os10 | pg1 | os1 |
|---|---|---|---|---|---|
| list page, class, deep keyset | `s1a_deep` | 1.3 / 2 | 32.3 / 52 | 1.5 / 2 | 24.0 / 36 |
| list page, cluster, deep | `s1b_deep` | 1.8 / 3 | 17.2 / 26 | 1.5 / 2 | 19.1 / 27 |
| list page, class + flags, deep | `s1c_deep` | 1.4 / 2 | 34.6 / 99 | 1.3 / 2 | 21.1 / 45 |
| exact total, one class | `s1a_total` | 7.8 / 248 | 2.2 / 3 | 1.0 / 29 | 2.3 / 3 |
| exact total, class + flags | `s1c_total` | 11.7 / 454 | 2.6 / 4 | 1.5 / 53 | 2.0 / 3 |
| exact total, cluster | `s1b_total` | 0.8 / 3 | 2.1 / 3 | 0.5 / 1 | 2.2 / 3 |
| review queue, deep | `s2_deep` | 2.3 / 12 | 34.1 / 115 | 2.0 / 7 | 27.0 / 41 |
| review queue, exact total | `s2_total` | 138.3 / 257 | 2.3 / 3 | 13.6 / 42 | 2.6 / 3 |
| cluster cards, whole project (raw aggregation) | `s3_all` | 3976.9 / 5799 | 1813 / 1832 at 1 client only | 770.1 / 1003 | 505.5 / 602 |
| cluster cards, one detector class | `s3_det` | 2460.5 / 4867 | not run | 411.6 / 983 | 32.9 / 276 |
| one cluster's facets | `s4_facets` | 2.2 / 23 | not run | 1.9 / 11 | 5.7 / 11 |
| dataset statistics (raw) | `s5_stats` | 8721 / 17021 | not run | 825.6 / 1701 | 202.2 / 284 |
| class counts (raw) | `s6_classes` | 1087 / 7967 | not run | 178.2 / 706 | 116.6 / 175 |
| photo page with item facts | `s7_first`, `s7_deep` | 3.1 / 5, 3.7 / 6 | 28.0 / 49, 29.0 / 74 | 3.1 / 5, 3.5 / 5 | 22.9 / 34 |
| box rows page, deep | `s8_deep` | 194.7 / 593 | 52.1 / 88 | 21.0 / 107 | 36.4 / 55 |
| box rows first page + total | `s8_first_total` | 663.6 / 2710 | 105.9 / 239 | 25.2 / 123 | 40.2 / 70 |
| box rows exact total | `s8_total` | 600.2 / 2496 | 2.8 / 4 | 3.9 / 12 | 2.8 / 4 |
| locate, class (exact counts) | `s9a` | 48.0 / 1457 | 367.5 / 1517 | 3.4 / 66 | 5.3 / 7 |
| locate, cluster | `s9b` | 0.9 / 4 | 116.5 / 1478 | 0.6 / 2 | 7.0 / 9 |
| locate, class + flags | `s9c` | 31.9 / 739 | 336.4 / 1614 | 3.1 / 59 | 8.1 / 10 |
| box text substring, sequential scan | `s10_text` | 2244.7 / 9444 | 5.2 / 7 | 385.5 / 502 | 4.8 / 6 |
| map grid | `s11_map` | 18.0 / 171 | not run | 3.0 / 29 | 8.6 / 21 |

Engine settings and index set of the spike: `notes.md` sections 2 and 3. The spike's tables are
`objects`, `boxes`, `images` with text ids; the record has integer keys, `uuid` wire ids, and
placement, detector values and boxes in their own tables, so every spike number is a guide for the
same access path, not a measurement of the record's query.

### 1.2 Not measured yet (the chain is still running)

| Pending | What it decides here | Until then |
|---|---|---|
| `pg_sum`: shapes 3, 5, 6 read from `summary_pg.sql` tables | the dashboard reads of section 6 (rows 10, 11, 15, 16) | the design stands; the fallback is additive (section 1.4) |
| `pg_pl`: placement side table (`pl_*` queries) | the cluster-joined reads (rows 1 with a cluster filter, 2, 8, 10 to 13) | the join form of `pl_*` is used as written |
| `pg_ext`: trigram GIN at 10 M (1 M single client: 89 ms to 0.5 ms, `notes.md` section 4) | section 8 | design stands |
| write-under-read (`w_sum` at 1,000 relabels per second) and the HOT ratio | the cost of summary upkeep per label (section 6.5) and the fillfactors of 15a 2.6 | "to be measured" |
| OpenSearch dashboard shapes at 10 M | only the comparison | nothing waits on it |

### 1.3 How "to be measured" is closed

The coordinator's first wave sweep after a slice runs each new query of that slice through
`EXPLAIN (ANALYZE, BUFFERS)` and the gate A runner (`run.py`, held, 1 and 8 clients) against the 10 M
spike database re-loaded into the record's tables by the slice's seed, and on set P. A query that
misses its bar goes back to the slice's lane as code work; a dashboard that misses it gets the
additive fallback of 1.4.

### 1.4 The fallback, if a pending measurement contradicts a decision

Additive only: a search copy without vectors (plan 15 section 6, third outcome), derived (R7), fed
in commit order from the record, for the failing shapes only. No slice already built is rewritten:
the shape's reader function keeps its signature and gains a second implementation behind it.

## 2. Read patterns

Every statement runs inside `project_tx` (02a 1.4), so row-level security adds the project; the
repositories add `project_id = :p` as well (02a 1.4, RC-9). `:p` is the bound project, `:family` the
class id and every class merged into it (15a section 7, M16), `:serving` the item partition named by
`cluster_activations.serving_partition_id`. All keyset forms follow section 4.

**P1, an item page on an `item_state` sort** (spike `s1a_first`, `s1a_deep`):

```sql
SELECT s.item_key, s.crop_id, s.updated_at
FROM item_state s
WHERE s.project_id = :p
  AND s.class_id = ANY(:family) AND NOT s.excluded AND NOT s.holdout      -- the filter, translated (below)
  AND (s.updated_at, s.crop_id) < (:after_updated_at, :after_crop_id)      -- pages after the first
ORDER BY s.updated_at DESC, s.crop_id DESC
LIMIT :limit_plus_one;
```

Filter translation (today `item_filter_clauses`, `item_filter.py`): class, `class_source`,
`label_source`, validated, excluded, holdout, `area`, `rank`, `import_ids && :ids` are columns of
`item_state`; image-level predicates become `EXISTS (SELECT 1 FROM photos p / photo_capture c WHERE
... = s.photo_key)`; detector class, detector confidence, blur and proposal name become `EXISTS` on
`detections`; cluster (slice 6) `EXISTS` on `cluster_memberships` of `:serving` (or `s.moved_*`, 15a
4.3 layer 5); near-duplicate group (slice 5) `EXISTS` on `near_dup_members` of the current run; box
predicates (slice 4) `EXISTS` on `box_state`; text (slice 4) section 8; embedding state (slice 3) a
vector row or a `vector_gaps` row; projection window (slice 8) `EXISTS` on `item_projections`. The
translation is one module (`src/db/queries/item_filter_sql.py`, new) used by every item reader, so a
filter means the same rows everywhere, as `item_filter_clauses` does today.

**P2, hydrate items for the wire** (today `crops_by_ids`, `crop_listing.py:120`; `mget_crops`,
`curation_opensearch/crops.py:79`): `load_item_records(session, item_keys)` (new,
`src/db/repos/item_records.py`) runs one statement per family, each `WHERE item_key = ANY(:keys)`:
`item_state` joined to `items` and `photos`; `detections`; `class_opinions`; `item_region_results`;
`box_state` with `box_texts`; `probe_predictions`; `item_scores`; `item_projections`;
`assist_answers`; `item_notes`; `text_lines`; `photo_capture`; and, from their slices, the serving
membership and the current near-duplicate group. Class names come from the registry snapshot
(`fold_load.load_registry`). The result keeps the page order. `serialize_item` (`wire.py:288`) takes
the typed record instead of an OpenSearch `_source`. Read-your-writes needs nothing: a committed
write is visible to the next statement (plan 15 section 4.4), so every realtime `get` and `mget`
(shapes 9, 23, 29, 32 to 35, 63, 64, 68, 80) becomes a plain read.

**P3, an exact total from a summary table** (section 6):

```sql
SELECT coalesce(sum(n), 0) FROM sum_items
WHERE project_id = :p AND class_id = ANY(:family) AND NOT excluded AND NOT holdout;
```

**P4, a capped count** (section 5):

```sql
SELECT count(*) FROM (
  SELECT 1 FROM item_state s WHERE s.project_id = :p AND <the page's filter> LIMIT 10001
) t;
```

**P5, locate** (today `list_locate.py:146-158`): (a) the item and its sort values, `SELECT
s.updated_at, s.crop_id FROM items i JOIN item_state s USING (item_key) WHERE i.project_id = :p AND
i.crop_id = :crop AND <filter>` (`uq` key of `items`); absent means `not_found` or `filtered_out` as
today. (b) its rank, a capped count of the rows before it: `SELECT count(*) FROM (SELECT 1 FROM
item_state s WHERE s.project_id = :p AND <filter> AND (s.updated_at, s.crop_id) > (:k, :t) LIMIT
10001) t`. (c) the previous row for `cursor_before`: the same filter, `(s.updated_at, s.crop_id) >
(:k, :t) ORDER BY s.updated_at ASC, s.crop_id ASC LIMIT 1`. `cursor_after` is the item's own keyset
values; neither cursor needs the rank.

**P6, a photo page with card facts** (spike `s7_first`, adapted; the counts come from
`photo_counts`, section 6):

```sql
SELECT p.photo_key, p.image_id, p.source, p.storage_key, p.created_at, c.taken_utc, k.items, k.labeled,
       k.validated, k.unlabeled, k.in_holdout, b.top
FROM (SELECT photo_key FROM photos
      WHERE project_id = :p AND photo_key < :after_photo_key        -- sort indexed_at:desc (section 4)
      ORDER BY photo_key DESC LIMIT :limit_plus_one) pg
JOIN photos p USING (photo_key)
LEFT JOIN photo_capture c USING (photo_key)
JOIN photo_counts k USING (photo_key)
CROSS JOIN LATERAL (
  SELECT jsonb_agg(jsonb_build_array(t.crop_id, t.class_id, t.rank, t.bbox) ORDER BY t.rank, t.crop_id) AS top
  FROM (SELECT crop_id, class_id, rank, bbox FROM item_state s
        WHERE s.photo_key = p.photo_key AND NOT s.excluded ORDER BY s.rank, s.crop_id LIMIT :box_limit) t) b
ORDER BY pg.photo_key DESC;
```

An item-level filter on the photo list is `AND EXISTS (SELECT 1 FROM item_state s WHERE s.photo_key
= pg.photo_key AND <item filter>)`: a semi-join with no cap, so the capped join of today
(`JOIN_MAX_IMAGES`, `image_join.py:132`, `join_truncated`) goes.

**P7, cluster cards from the summaries** (spike `sum_s3_all`, keyed by partition):

```sql
WITH top AS (
  SELECT cluster_uid, sum(n_live) AS n FROM sum_cluster
  WHERE partition_id = :serving GROUP BY cluster_uid HAVING sum(n_live) > 0
  ORDER BY n DESC, cluster_uid LIMIT :max_clusters)
SELECT t.cluster_uid, t.n, a.validated, a.labelled, a.max_updated, a.avg_conf, k.top_classes, d.top_det, r.review, g.groups
FROM top t
CROSS JOIN LATERAL (SELECT sum(validated_live) AS validated, sum(labelled_live) AS labelled,
    max(max_updated_live) AS max_updated, sum(conf_sum_live) / NULLIF(sum(conf_n_live), 0) AS avg_conf
  FROM sum_cluster s WHERE s.partition_id = :serving AND s.cluster_uid = t.cluster_uid) a
CROSS JOIN LATERAL (SELECT jsonb_agg(jsonb_build_array(class_id, c) ORDER BY c DESC, class_id) AS top_classes
  FROM (SELECT class_id, sum(n) AS c FROM sum_cluster_class s
        WHERE s.partition_id = :serving AND s.cluster_uid = t.cluster_uid
        GROUP BY class_id HAVING sum(n) > 0 ORDER BY c DESC, class_id LIMIT 3) x) k
CROSS JOIN LATERAL (SELECT detector_class_name AS top_det FROM sum_cluster s
  WHERE s.partition_id = :serving AND s.cluster_uid = t.cluster_uid AND n_live > 0
  ORDER BY n_live DESC, detector_class_name LIMIT 1) d
CROSS JOIN LATERAL (SELECT jsonb_object_agg(tab, n) AS review FROM (SELECT tab, sum(n) AS n FROM sum_cluster_review s
  WHERE s.partition_id = :serving AND s.cluster_uid = t.cluster_uid GROUP BY tab) y) r
CROSS JOIN LATERAL (SELECT count(*) AS groups FROM (SELECT group_id FROM sum_cluster_dup s
  WHERE s.partition_id = :serving AND s.cluster_uid = t.cluster_uid GROUP BY group_id HAVING sum(n) > 0) z) g
ORDER BY t.n DESC, t.cluster_uid;
```

The detector-class form (spike `sum_s3_det`) adds `AND detector_class_name = :det` to every
subquery. Ties among top classes order by class name in Python after the registry join, as the
terms aggregation orders equal counts by key today.

**P8, a page on a membership sort** (spike `pl_s2_first`): the review queue's cluster-fit order, the
`core_first` order and a cluster's own list order by score.

```sql
SELECT s.item_key, m.score
FROM cluster_memberships m JOIN item_state s ON s.item_key = m.subject_key
WHERE m.partition_id = :serving AND m.score IS NOT NULL
  AND s.project_id = :p AND <tab or list predicate on s>
  AND (m.score, m.subject_key) > (:after_score, :after_key)
ORDER BY m.score, m.subject_key
LIMIT :limit_plus_one;
```

Inside one cluster the same with `m.cluster_uid = :cluster` (index X22).

**P9, box rows** (section 9). **P10, trigram text** (section 8).

**P11, a job stream in key order** (section 10):

```sql
COPY (SELECT v.subject_key, v.embedding FROM vec.v<lock_key>_<space_key> v
      JOIN item_state s ON s.item_key = v.subject_key
      WHERE s.project_id = :p AND <pool filter> ORDER BY v.subject_key) TO STDOUT (FORMAT BINARY);
```

or keyset batches `WHERE <filter> AND s.item_key > :last ORDER BY s.item_key LIMIT 5000`, one short
transaction each.

**P12, a vector search** behind the index interface of plan 15 section 7 and 15a 6.4: a count of the
filtered rows (capped at gate B's upper threshold) picks exact search, an iterative scan or a plain
index scan. Gate B owns the index, the thresholds and the scan limit.

**P13, a scoped aggregation**: `GROUP BY` over the rows a filter or a job scope selects, used only
where no summary key answers and the scope is bounded or the reader is a job.

**P14, a work list**: `item_region_results` rows in a pending status (X15) or `analysis_ledger` rows
of a kind whose `analyser_id <> :current OR state <> 'done'` (X16, 15a m2), in key order.

**P15, the map grid** (spike `s11_map`, with the expression GiST index X14 instead of a stored
point column):

```sql
SELECT floor((c.lon + 180) / 360 * pow(2, :z))::int AS x,
       floor((1 - ln(tan(radians(c.lat)) + 1 / cos(radians(c.lat))) / pi()) / 2 * pow(2, :z))::int AS y,
       count(*) AS n, avg(c.lat) AS clat, avg(c.lon) AS clon
FROM photo_capture c
WHERE c.project_id = :p AND c.lat IS NOT NULL
  AND point(c.lon, c.lat) <@ box(point(:w, :s), point(:e, :n))
GROUP BY 1, 2;
```

The `<@` operator on `point` and `box` is core ([geometric functions](https://www.postgresql.org/docs/18/functions-geometry.html));
a GiST index on the expression `point(lon, lat)` serves it ([GiST](https://www.postgresql.org/docs/18/gist.html),
[expression indexes](https://www.postgresql.org/docs/18/indexes-expressional.html)).

## 3. The 80 shapes

Kind: **page** (SQL page or lookup), **sum** (summary table), **cap** (capped count), **agg** (scoped
aggregation), **stream**, **vector**, **del** (disappears), **kept** (stays on the global
OpenSearch indexes of inventory section 1.2 until phase 3). Slices: 15c (S1a ingest writes, S1b
browse, S2 labelling, S3 vectors, S4 regions, S5 near-duplicates and holdout, S6 clustering, S7
import, combine, export, training, S8 machine writers, S9 deletions and acceptance).

| # | Serves (today) | Slice | Kind | New form | Index or summary | Measured |
|---|---|---|---|---|---|---|
| 1 | `GET /crops`, table view (`routers/curation/crops_list.py:143`, body `crop_listing.py:85`; `table_view.py:134`) | S1b | page | P1 + P2; total P3 when the filter is summary keys, else P4; hidden row 7; sorts section 7 | X1, X2, X4; joined sorts section 7 | `s1a_deep` 1.3 / 2, `s1c_deep` 1.4 / 2 (spike indexes, re-keyed here: to be measured) |
| 2 | `GET /review/{tab}` (`routers/curation/review.py:222`; `review_tab_clauses.py`, `review_sorts.py:48`) | S2 | page | P8 for cluster-fit sorts, P1 on X4 for confidence sorts, section 7 for others; total P4; region text section 8 | X4, X22, X23, X7 | `s2_deep` 2.3 / 12 (column variant); `pl_s2` pending; total was `s2_total` 138 / 257, now P4: to be measured |
| 3 | locate, own row (`list_locate.py:146`; callers `review.py:287`, `crops_list.py:~190`, `routers/curation/images_view.py:62`) | S1b | page | P5a; photos: `photos (project_id, image_id)` | `uq` of `items`, `photos` | to be measured |
| 4 | locate, rank and total (`list_locate.py:153`, `:154`; `review.py:337`) | S1b | cap | P5b capped rank; total P3 or P4 | as row 1 | exact form `s9a` 48 / 1457, `s9c` 31.9 / 739; capped: to be measured |
| 5 | locate, previous cursor (`list_locate.py:158`) | S1b | page | P5c | as row 1 | to be measured |
| 6 | `GET /regions`, training candidates, suspected false positives, cohorts (`region_rows.py:222`; `cohorts.py:33`; `routers/curation/regions.py:255`, `:425`) | S4 | page | P9; `total_rows` P3 on `sum_box` for a state + detector filter, else P4; `total` (items) P4 | X6, X7 | spike `s8_deep` 194.7 / 593 and `s8_total` 600 / 2496; new design to be measured |
| 7 | hidden counts (`hidden_counts.py:33`) | S1b | sum | P3 per hidden kind when the filter is summary keys, else P4 | `sum_items` | to be measured |
| 8 | ordered crop pages `core_first` (`crop_orders.py:74`; `crops_list.py:100`, `:206`) | S6 | page | P8 on `m.score DESC` with the missing-last segment; pool count P4; `outliers` and `diverse` stream the pool's vectors (row 59) | X23 | to be measured |
| 9 | page hydrate by ids (`crop_listing.py:120`) | S1b | page | P2 | `items` `uq`, primary keys | to be measured |
| 10 | `GET /clusters` cards (`cluster_cards.py:339`, body `:99`) | S6 | sum | P7; with an item filter outside summary keys, P13 over the filtered members | `sum_cluster`, `sum_cluster_class`, `sum_cluster_review`, `sum_cluster_dup` | raw `s3_all` 3977 / 5799 (pg1 770 / 1003, os1 506 / 602); `pg_sum` pending |
| 11 | `GET /clusters/representatives` page ids (`clusters.py:197`) | S6 | sum | the `top` step of P7 with `OFFSET :offset` (at most about 20,000 clusters per partition at 10 M, `notes.md` 1) | `sum_cluster` | pending with row 10 |
| 12 | representatives per cluster (`cluster_cards.py:291`; `vlm_scope.py:147`) | S6 | page | one statement, a `LATERAL` per cluster: serving members joined to `item_state`, not excluded, not holdout, `PARTITION_REPS_SORT` (`cluster_representatives.py:19`), `LIMIT :per_cluster` | X22 | to be measured |
| 13 | `GET /clusters/{cluster_uid}/facets` (`cluster_facets.py:115`) | S6 | page | spike `pl_s4_facets` over `cluster_memberships` joined to `item_state` and `detections` | X22 | column variant `s4_facets` 2.2 / 23; `pl_s4` pending |
| 14 | `GET /detections/summary` (`detections_summary.py:91`) | S3 | agg | P13: `detections.proposal_name` by embedding state (a vector row, else `vector_gaps.reason`, else `deferred`) over the filter | none | to be measured (dashboard bar; summary candidate `sum_embedding` if missed, section 6.6) |
| 15 | `GET /stats/dataset` (`stats_dataset.py:244`, body `stats_dataset_query.py:45`; `pipeline_events.py:65`) | S2 | sum | P3 over `sum_items` (source, class source with and without class, validated, holdout), `sum_region` (status, verifier, by class family), `sum_box` (detector, state), `sum_cluster` (distinct clusters and states, slice 6); with an item filter outside keys, P13 | four summaries | raw `s5_stats` 8721 / 17021 (pg1 826 / 1701); `pg_sum` pending |
| 16 | class counts (`class_counts.py:80`; `GET /stats/classes`, `GET /classes`, `workflow_steps.py:18`, `stats_classes.py:95`) | S2 | sum | P3 grouped by class (merge root) and label source over `sum_items` | `sum_items` | raw `s6_classes` 1087 / 7967; `pg_sum` pending |
| 17 | new-class proposals summary (`review_proposals.py:67`) | S8 | agg | `GROUP BY proposed_class` over `class_opinions` with outcome `new_class_pending`, live items; three sample ids per group by a `LATERAL` | X19 | to be measured |
| 18 | raw-label aggregations (`review_terms.py:54`, `:101`; `scripts/curation/cluster_raw_labels.py:159`, `:186`) | S8 | agg | `GROUP BY raw_label` (top `max_terms`) joined to `raw_label_clusters`; the script's page loop becomes keyset on `raw_label` | X18 | to be measured |
| 19 | resolve proposals (`review_resolve.py:53`, `:66`) | S2 | cap | P4 over the proposal predicate, then a keyset walk by `item_key` | X19 | to be measured |
| 20 | kNN over items (`semantic_search_run.py:64`, `:149`, `:256`) | S3 | vector | P12 on the item space; fusion stays in Python | vector table; gate B index | gate B |
| 21 | kNN over box vectors (`semantic_search_box.py:43`) | S3 | vector | P12 on the box space, best box per item | vector table; gate B | gate B |
| 22 | unembedded in scope (`semantic_search_run.py:100`) | S3 | cap | P4 of scope rows with no vector row (`NOT EXISTS`) | vector primary key | to be measured |
| 23 | search source vector (`semantic_search.py:348`; `semantic_search_image.py:110`) | S3 | page | `SELECT embedding FROM vec.<table> WHERE subject_key = :k` | vector primary key | to be measured |
| 24 | `GET /images` (`image_view.py:292`) | S1b | page | P6; sorts `indexed_at` (X9, insert order), `taken_at` (X13), `file_name` (X12), `unlabeled_count` (X24); geo filter P15's predicate; item filters as a semi-join; near-duplicate group filter from S5 | X9, X12, X13, X14, X24 | `s7_first` 3.1 / 5, `s7_deep` 3.7 / 6 (taken sort) |
| 25 | image item facts (`image_counts.py:94`) | S1b | page | `photo_counts` for counts, the `LATERAL` of P6 for boxes | X3, `photo_counts` | inside `s7` above |
| 26 | image join ids (`image_join.py:132`) | S1b | del | replaced by the semi-join inside P1, P6, P7, P13 | none | n/a |
| 27 | item-side image join (`image_join.py:172`, caller `map_cells.py:128`) | S1b | del | replaced by `EXISTS` from `photo_capture` to `item_state` | none | n/a |
| 28 | image counts and readiness (`image_view.py:153`, `:344`; `map_cells.py:128`, `:148`, `:218`; `workflow_steps.py:346`, `:353-373`) | S1b | sum | totals P3 on `sum_photos` (source, located), else P4 on `photos`; readiness is always true and goes | `sum_photos` | to be measured |
| 29 | photo detail, file and thumbnail routes (`image_detail.py:56`, `:66`; `images_view.py:81`; `capture_wire.py:64`; `image_serving.py:532`; `image_item_write.py:95`) | S1b | page | P2 of the photo and of its items by rank | `photos` `uq`, X3 | to be measured |
| 30 | `GET /map` (`map_cells.py:128`, `:148`, `:218`) | S1b | page | P15; extent `min/max(lat, lon)` over the same predicate | X14 | `s11_map` 18.0 / 171 |
| 31 | ingest lookups (`ingest_objects.py:80`; `routers/curation/datasets.py:108`; `ingest_reads.py:51`, `:71`) | S1a | page | (a) `source_identifier = ANY`; (b, c) `storage_key = ANY`; (d) `GROUP BY source, date_trunc('day', created_at)` over 14 days | X11, X12, X10 | to be measured |
| 32 | item get (`crops_list.py:211`, `:256`; `review.py:314`; `item_embed_inline.py:82`; `item_geometry.py:239`, `:244`; `label_undo.py:267`; `clients/occ.py:100`) | S1b | page | P2 of one key; the `occ.py` read-merge-write goes in S9 | `items` `uq` | n/a (one probe) |
| 33 | item mget for writers (`curation_opensearch/crops.py:79` and about 20 callers; `occ_upsert.py:144`, `:172`; `ingest_writer.py:532`; `region_text_search.py:48`; `item_scores/mistakenness.py:129`) | S1b | page | P2; the conditional-write callers go in S9; job callers switch in their slices | as P2 | n/a |
| 34 | mget exists-only (`image_completeness.py:79`, `:151`; `near_duplicates/holdout.py:170`; `ingest_dedup.py:87`; `near_duplicates/core_content.py:41` on the global index) | S1a | page | `photos.photo_keys_by_image_id`, `items.item_keys_by_crop_id`; `image_completeness` goes; `holdout.py:170` switches in S5; `core_content.py:41` stays (global) | `uq` keys | n/a |
| 35 | mget vectors (`near_duplicates/vectors.py:56`; `embedding_spaces/reindex_encoder.py:76`; `reindex_opensearch.py:343`) | S3 | page | `SELECT subject_key, embedding FROM vec.<table> WHERE subject_key = ANY(:keys)` | vector primary key | to be measured |
| 36 | `POST /selection/summary` (`selection_summary.py:71`, `:94`) | S2 | agg | P13 over the selection (crop ids, or the selection filter) | as P1 | to be measured |
| 37 | pool counts and area scan (`selection/select.py:174`; `selection/pool_fetch.py:61`; `item_selection.py:110`) | S3 | stream | P4 for the pool; keyset stream of `(item_key, area)` | X1 | to be measured |
| 38 | scalar filter counts (`cohorts.py:33`; `audit.py:193`, `:226`; `assist/scope.py:103`; `assist/segment.py:61`; `classes_write.py:128`; `classes/projection_sync.py:61`; `review_empty_reason.py:26`, `:34`; `region_stage_control.py:92`; `workflow_steps.py`; `reprocess.py:205`; `export_readiness.py:114`; `training/preflight_checks.py:181`; `vlm_scope.py:231`; `autolabel/vlm_stage.py:186`; `holdout_freeze.py:80`; `dataset_import/runner.py:174`) | S2 | cap | one function `count_items(filter)`: P3 when the filter is summary keys, else P4; callers outside labelling switch in their slices | summaries | to be measured |
| 39 | `GET /quality/readiness` (`dataset_readiness.py:364`, `:386`, `:388`) | S7 | sum | class by holdout from `sum_items` (P3); photo coverage per class P13 | `sum_items` | to be measured |
| 40 | region profile impact (`region_impact.py:93`, `:137`, `:186`) | S4 | agg | P13 `GROUP BY analyser_id, status` over `item_region_results`, box detector counts from `sum_box` | none (an activation preview, not a page) | to be measured |
| 41 | `GET /regions/clusters` (`region_fp_clusters.py:99`) | S6 | agg | P13 over the box partition's members joined to `box_state` and `item_state`; top hits by a `LATERAL` | X22 | to be measured |
| 42 | region drain (`region_drain.py:60`; `ingest.py:376`) | S4 | sum | P3 `GROUP BY region_status` over `sum_region`, class family filter | `sum_region` | to be measured |
| 43 | field coverage (`field_coverage.py:92`) | S2 | cap | offer test `EXISTS (SELECT 1 FROM <result table> WHERE project_id = :p)`; counts P4 | primary keys | to be measured |
| 44 | pipeline health and ops metrics (`autolabel/health_snapshot.py:39`; `ops_metrics_refresh.py:99`) | S8 | sum | P3 where the filter is keys, else P4 | summaries | to be measured |
| 45 | holdout stats, strata scan and census (`holdout_freeze.py:80`, `:179`; `holdout.py:143`, `:211`) | S5 | sum | stats P3 (`sum_items.holdout`); strata scan keyset by `item_key`; census `GROUP BY class_id, source` over `sum_items` | `sum_items` | to be measured |
| 46 | label-op current counts (`label_ops/views.py:34`) | S2 | page | `SELECT class_op_id, count(*) FROM item_state WHERE class_op_id = ANY(:ops) GROUP BY 1` | X5 | to be measured |
| 47 | accept-machine and detector census (`accept_machine.py:133`, `:154`; `detector_adoption.py:124`) | S2 | agg | class census P3 (`sum_items`, machine sources, not validated); detector census P13 `GROUP BY detections.class_name` over eligible items | `sum_items` | to be measured |
| 48 | VLM scope (`vlm_scope.py:121`, `:231`) | S8 | agg | per-cluster sizes from `sum_cluster`; the sample `_SAMPLE_SOURCE` becomes a deterministic hash test on `crop_id` | `sum_cluster`, X22 | to be measured |
| 49 | worker pending poll (`scripts/curation/worker/cascade.py:126`; `scripts/curation/vlm_pending.py:211`) | S4 | page | P14 on `item_region_results` pending, joined to `item_state`, `ORDER BY item_key LIMIT :batch`, in-flight keys excluded; the VLM variant's call site switches in S8 | X15 | to be measured |
| 50 | auto-label selection (`autolabel/selection.py:219`, `:240`, `:252`, `:288`) | S8 | stream | counts P4; the scroll becomes keyset over X1 in ascending order | X1 | to be measured |
| 51 | reprocess targets (`reprocess_targets.py:179`, `:221`; `routers/curation/reprocess.py:64`, `:113`) | S8 | stream | keyset by `item_key` over the target filter; a photo by `image_id` | as P1 | to be measured |
| 52 | region requeue preview and run (`region_requeue.py:317`, `:540`) | S8 | stream | preview P3 (`sum_region`, `sum_box`); run keyset of `item_key` through X6 or `item_region_results` | X6 | to be measured |
| 53 | repair and reclassify scans (`registry_reclassify.py:258`; `repair_scan.py:43`; `region_text_repair.py:213`) | S8 | stream | `registry_reclassify` keyset over X2; the two repairs fixed document defects and become ledger re-runs (P14) | X2, X16 | to be measured |
| 54 | image counts upkeep (`image_counts_backfill.py:36`, `:47`; `image_counts_refresh.py:107`) | S1a | del | `photo_counts` kept by the writer (section 6) | none | n/a |
| 55 | source backfill page (`source_backfill/job.py:67`) | S1a | stream | P14: ledger kind `capture_parser` rows not at the current analyser, keyset by `photo_key` | X16 | to be measured |
| 56 | probe job reads (`probe_predictions.py:209`, `:274`, `:314`, `:397`, `:425`) | S8 | stream | P11 of the pool; counts P4; the weighted top N is `ORDER BY <weights expression> LIMIT :n` over `probe_predictions` joined to `item_state` | X21 | to be measured |
| 57 | item scores scroll (`item_scores/job.py:158`, `:246`, `:253`) | S8 | stream | keyset by `item_key` over `probe_predictions` joined to the live, unvalidated items | primary keys | to be measured |
| 58 | export scroll (`export_support.py:161`; callers `export.py:171`, `export_images.py:166`, `export_negatives.py:53`, `region_eval_data.py:131`, `:164`, `export_single_class_rows.py:230`, `:303`, `:315`) | S7 | stream | section 10: one snapshot, server-side cursor in `photo_key, item_key` order | primary keys | to be measured |
| 59 | vector reads in slices (`clustering/pit_pages.py:53`, `:114`; `clustering/vector_read.py:69`, `:123`; `clustering/region_box_rows.py:119`; `clustering/pool_snapshot.py:60`; `near_duplicates/vectors.py:87`, `:120`; `near_duplicates/chain.py:43`; `scripts/curation/backfill_item_ids.py:83`) | S3 | stream | P11 `COPY ... (FORMAT BINARY)` in `subject_key` order (`vector_stream`, new); call sites switch in S5 (near-duplicates), S6 (clustering, region false-positive rows, outlier and diverse orders); `backfill_item_ids.py` goes in S1a | vector primary key | 69,630 rows per second (plan 15 section 1, not gate A) |
| 60 | clustering pre-counts (`clustering/residual_fetch.py:80`; `clustering/outliers.py:77`) | S6 | agg | exact `count(*)` of the pool inside the job (P13); jobs may take a second | as P1 | to be measured |
| 61 | near-duplicate kNN edges (`near_duplicates/candidates.py:80`, `:113`; `rescore.py:67`, `:89`) | S5 | vector | P12 batched per query vector, `k + 1`; the leak partition's `photo_key = ANY` as the filter | gate B | gate B |
| 62 | near-duplicate kNN on the global index (`near_duplicates/core_candidates.py:52`) | S9 | kept | unchanged | global index | n/a |
| 63 | near-duplicate views and status (`near_duplicates/member_views.py:36`, `:52`; `near_duplicates/status.py:204`) | S5 | page | members of a group by `ix_near_dup_members_group` joined to `item_state`; items without a vector P4 | 02d keys | to be measured |
| 64 | embedding-space re-index (`embedding_spaces/reindex_opensearch.py:155`, `:157`, `:205`, `:288`, `:319`, `:343`) | S3 | stream | work list: subjects with no row in the new space's table, keyset by key; the random sample becomes a hash sample; counts exact in the job | vector primary key | to be measured |
| 65 | box-embedding backfill script (`scripts/curation/backfill_region_embeddings.py:106`) | S3 | del | the re-index of the box space (row 64) | none | n/a |
| 66 | class-merge scroll (`clients/opensearch/scroll.py:37`; caller `classes/projection_sync.py:188`) | S2 | del | the merge job walks X2 by class in keyset chunks (15a section 7, class merge) | X2 | n/a |
| 67 | combine reads (`projects/combine/plan.py:426`, `:427`; `combine/image_copy.py:250`; `combine/holdout.py:34`; `combine/warnings.py:28`; `dataset_import/project_source.py:101`) | S7 | stream | keyset over the source project's photos and items by key, the source project bound for the read | primary keys | to be measured |
| 68 | dataset import reads (`dataset_import/project_source.py:137`, `:159`; `image_plan.py:78`; `undo.py:447`, `:490`; `reconcile.py:78`; `runner.py:174`) | S7 | page | P2 and keyset walks by key; undo reads the operation's claims (`claims.live_claims_of_operation`) | `uq_claims_op_subject` | to be measured |
| 69 | kNN on the global indexes (`clients/opensearch/search.py:199`) | S9 | kept | unchanged | | n/a |
| 70 | OCR search (`clients/opensearch/ocr.py:156`, `:221`) | S9 | kept | unchanged | | n/a |
| 71 | global cluster reads (`clients/opensearch/clusters.py:62`, `:123`, `:253`, `:277`) | S9 | kept | unchanged | | n/a |
| 72 | global duplicate groups (`duplicate_detection/service.py:54`, `:81`, `:96`, `:99`, `:105`) | S9 | kept | unchanged | | n/a |
| 73 | faces and persons (`face_identity/identification.py:37`; `face_records.py:41`; `person_management.py:64`, `:149`) | S9 | kept | unchanged | | n/a |
| 74 | face privacy counts (`face_privacy.py:75`, `:87`) | S9 | kept | unchanged | | n/a |
| 75 | image admin (`services/image_admin.py:22`, `:57`, `:71`, `:87`, `:101`) | S9 | kept | unchanged | | n/a |
| 76 | `GET /viz/projection` (`embedding_viz_read.py:90`, `:162`, `:190`) | S8 | page | points keyset by `item_key` over `item_projections` joined to `item_state` and the serving membership; missing count P4 | `item_projections` key | to be measured |
| 77 | assist-loop partition (`assist_loop/partition.py:298`, `:344`) | S8 | agg | `GROUP BY cluster_uid` over serving members under the filter, keyset on `cluster_uid`; answers by P2 | X22 | to be measured |
| 78 | project housekeeping counts (`projects/stats.py:43`; `projects/delete.py:124`; `projects/clone_validate.py:88`; `ingest_policy.py:111`; `embedding_spaces/capacity.py:61`; `embedding_spaces/reindex.py:312`; `scripts/curation/seed_live_harness.py:535`) | S1a | sum | P3 on `sum_items` and `sum_photos`; delete and clone preflight may count exactly (jobs) | summaries | to be measured |
| 79 | periodic ticks (`periodic/cluster_refresh.py:84`, `:85`, `:92`; `periodic/open_vocab_sweep.py:29`) | S6 | cap | P4 of the pool and pending statuses; "new photos" is `EXISTS (SELECT 1 FROM photos WHERE project_id = :p AND created_at > :since)` on X10 (BRIN); the open-vocabulary sweep's call site switches in S8 (ledger, P14) | X10, X16 | to be measured |
| 80 | ingest dedup (`clients/optional_doc.py:18`; `ingest_dedup.py:69`, `:87`) | S1a | page | photo by `image_id` (`uq`), its items by `photo_key` | `uq` keys, X3 | n/a |

Counts by kind: page 25, stream 12, sum 11, agg 10, cap 6, vector 3, del 5, kept 8 (80). The trigram
index (section 8) is not a shape of its own: it serves the text predicates of rows 1, 2, 6 and 53.

## 4. The list cursor

Today `encode_cursor` (`list_cursor.py:23`) wraps `{"sort", "after", "carry"}` in URL-safe base64,
`decode_cursor` refuses a cursor of another sort (`:31-44`), `with_tiebreak` appends `crop_id`
ascending to every sort (`:54-58`), and `after` holds OpenSearch's hit `sort` values.

**What stays.** The envelope and its three functions (`encode_cursor`, `decode_cursor`,
`cursor_carry`), the refusal of a foreign cursor, `limit + 1` rows to know whether a next page
exists, and the carry of the first page's totals (`list_counts.CarriedCounts`), which gains
`total_capped` (section 5). `page_body` and `split_page` (OpenSearch bodies and hits) are replaced
by `keyset_clause(sort: SortSpec, after)` and `split_rows(rows, sort, limit, carry)` in the same
module.

**What `after` holds.** The keyset values of the last row served, in sort order, tiebreak last, as
JSON scalars that round-trip exactly: a `timestamptz` as an ISO-8601 string with microseconds (an
epoch float would lose them), a `real` as the Python float of the stored value, cast back with
`::real` (the spike's `%(cs)s::real`), a `crop_id` as 32 hex digits (`wire_ids.to_wire`), an
integer key as an integer. A fourth field `seg` is added when a sort has a missing-last segment:
`"v"` (rows with a value) or `"n"` (rows without one).

**One direction per row comparison (M7).** A row constructor comparison `(a, b) < (x, y)` is
evaluated left to right with one operator ([row-wise comparison](https://www.postgresql.org/docs/18/functions-comparisons.html#ROW-WISE-COMPARISON)),
so it cannot mix directions. Today's `desc` sorts carry an ascending `crop_id` tiebreak, which is
mixed. Rule: **the tiebreak takes the direction of the last sort key**, so every single-key sort is
one row comparison, `<` for `desc` and `>` for `asc`, and one B-tree serves both directions, because
an index scanned backward gives the reverse order ([indexes and ORDER BY](https://www.postgresql.org/docs/18/indexes-ordering.html)).
The visible effect: equal sort values now come in descending `crop_id` order under a `desc` sort
(contract diff, 15a section 10 addendum in 15c). A sort with several keys of mixed directions (none
in `CROP_SORT_FIELDS`, `crop_browse.py:17`; review sorts are checked by the S2 lane) uses the expanded
form `k1 < :k1 OR (k1 = :k1 AND (k2, t) > (:k2, :t))`, whose cost is an index range on `k1` with a
filter for the tie group.

**Missing values last, in both directions** (today `'missing': '_last'`, `crop_browse.py:91`). An
index scanned backward puts NULLs first, and a row comparison with a NULL is NULL, so a NULL row
would never be reached. A nullable sort key pages in two segments: segment `v` is `key IS NOT NULL`
in key order (`(key, t) < (:k, :t)`), and when it runs out the same page continues with segment `n`,
`key IS NULL ORDER BY t` in the sort's direction. The cursor's `seg` says where to resume.

**Photo insert order.** `indexed_at` (today the images document's `indexed_at`) is
`photos.created_at`, the `now()` of the ingest transaction, and `photo_key` is an identity value
taken at insert ([identity columns](https://www.postgresql.org/docs/18/ddl-identity-columns.html)).
The two orders differ only between photos of ingest transactions that overlap in time (seconds), so the `indexed_at` sorts page
on `photo_key` alone (X9), and the wire still shows `indexed_at`.

**The tiebreak column.** `crop_id` for sorts on `item_state` columns (15a 2.6 rule, `item_state.crop_id`
copy); `photo_key` for photo sorts (today `image_id`, `image_view.py:360`); `item_key` (as
`m.subject_key`) for sorts whose key lives in a result or membership table, because the index that
gives the order is on that table and holds no `crop_id`. Each `SortSpec` names its tiebreak, and the
cursor is refused across sorts, so a cursor never mixes them.

**Old cursors.** None survive: the data is re-ingested and a cursor of the old shape fails
`decode_cursor`'s checks with 400, as a foreign cursor does today.

## 5. Totals: the capped-count rule

1. **Exact** when every term of the normalised filter is a key column of one summary table of
   section 6: `sum_items` (class family, `class_source`, `label_source`, validated, excluded,
   holdout, photo source), `sum_region` (class family, region status, region validated, verifier),
   `sum_box` (detector, state), `sum_photos` (source, located), the cluster summaries (partition,
   cluster, detector class, review tab) and `photo_counts` (one photo). `exact_total(filter)`
   (new, `src/db/queries/totals.py`) returns the P3 sum, or `None` when the filter is not keys.
2. **Capped** otherwise: P4 with the page's own `FROM` and `WHERE`, `LIMIT 10001`. The total served
   is `min(n, 10000)` and `total_capped` is `n > 10000`. The cap is one constant, `TOTAL_CAP =
   10_000`, in `src/domain/list_totals.py` (new). `LIMIT` stops the scan once that many rows are
   produced ([LIMIT and OFFSET](https://www.postgresql.org/docs/18/queries-limit.html)).
3. **Locate** ranks with the same rule: the count of rows before the item, capped; `index_capped`
   is true when the rank reached the cap (the editor's filmstrip shows "10,000+ before"). The
   cursors of a locate do not depend on the rank (P5).
4. **Hidden counts** (row 7) follow rules 1 and 2 per hidden kind, with a `capped` flag.
5. **Carried**: the first page computes the total once and carries `total` and `total_capped` in
   the cursor, as `CarriedCounts` does today; deeper pages cost their page only.
6. **Cost.** A capped count reads at most 10,001 index entries when an index matches the filter
   (estimate a few milliseconds; to be measured with the first S1b sweep). With no matching index and
   a rare predicate the scan can read far more before it finds 10,001 rows or ends; that is the same
   scan the page itself needs, so a capped count is never slower than its page by more than the rows
   past the page.
7. **Where exactness stays.** Dashboards and class counts read summaries (exact). Job sizing inside
   a job (rows 60, 64, 78) may count exactly: a job is not a page.

Frontend reading of totals (checked with `rg -n 'total' frontend/src/lib`): `pager.svelte.ts:123`,
`:195` and `MatchingItemsView.svelte:80`, `:138` show "N matching items"; `ReviewHeader.svelte:79-82`
shows "#position of total"; `reviewController.svelte.ts:69`, `:77`, `:144` add or subtract one after
a label; `editorController.svelte.ts:169`, `:179` show the filmstrip position. With
`total_capped` true these show "10,000+" and stop adjusting the number by one. The wire change is
in 15c (contract diff addendum).

## 6. Summary tables

### 6.1 DDL (migration `0011_storage_indexes`, except the cluster family in `0016_cluster_cutover`)

Every table has `project_id` with RLS(02a) (15a section 1). Key columns are `NOT NULL` with a
sentinel for "none" (`class_id` `-1`, text `''`), as the spike's `COALESCE(class_source, '')`
(`summary_pg.sql:26`): `ON CONFLICT` infers its arbiter from a unique index on exactly the target
columns ([INSERT](https://www.postgresql.org/docs/18/sql-insert.html)), and with no NULL in a key
the conflict is always found. Counts may reach 0 and stay as rows; readers sum, and the rebuild
removes them. `fillfactor 50` as the spike's (`summary_pg.sql:4`), so the counter updates, which
change no indexed column, stay HOT ([HOT](https://www.postgresql.org/docs/18/storage-hot.html)).

```sql
CREATE TABLE sum_items (                      -- every exact item total, class counts, dataset statistics
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  source text NOT NULL, class_id integer NOT NULL, class_source text NOT NULL, label_source text NOT NULL,
  class_validated boolean NOT NULL, excluded boolean NOT NULL, holdout boolean NOT NULL,
  n bigint NOT NULL,
  PRIMARY KEY (project_id, source, class_id, class_source, label_source, class_validated, excluded, holdout)
) WITH (fillfactor = 50);
CREATE TABLE sum_region (                     -- region drain, dataset region statistics
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  class_id integer NOT NULL, region_status text NOT NULL, region_validated boolean NOT NULL, region_verifier text NOT NULL,
  n bigint NOT NULL,
  PRIMARY KEY (project_id, class_id, region_status, region_validated, region_verifier)
) WITH (fillfactor = 50);
CREATE TABLE sum_box (                        -- box-row totals, box statistics, requeue preview
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  detector text NOT NULL, state text NOT NULL, n bigint NOT NULL,
  PRIMARY KEY (project_id, detector, state)
) WITH (fillfactor = 50);
CREATE TABLE sum_photos (                     -- photo totals
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  source text NOT NULL, located boolean NOT NULL, n bigint NOT NULL,
  PRIMARY KEY (project_id, source, located)
) WITH (fillfactor = 50);
CREATE TABLE photo_counts (                   -- the five numbers of an image card (today counts.*, count_fields.py:30)
  photo_key bigint PRIMARY KEY REFERENCES photos(photo_key) ON DELETE CASCADE,
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  items integer NOT NULL, labeled integer NOT NULL, validated integer NOT NULL,
  unlabeled integer NOT NULL, in_holdout integer NOT NULL
) WITH (fillfactor = 90);                     -- X24 indexes unlabeled, so a label is not HOT here; 90 keeps the new version near
-- slice 6, 0016_cluster_cutover: kept for the partitions cluster_activations names (serving or pending)
CREATE TABLE sum_cluster (
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  partition_id uuid NOT NULL REFERENCES cluster_partitions(id) ON DELETE CASCADE,
  cluster_uid uuid NOT NULL, detector_class_name text NOT NULL,
  n_all integer NOT NULL, n_live integer NOT NULL, labelled_live integer NOT NULL, validated_live integer NOT NULL,
  max_updated_live timestamptz NULL, conf_sum_live double precision NOT NULL, conf_n_live integer NOT NULL,
  PRIMARY KEY (partition_id, cluster_uid, detector_class_name)
) WITH (fillfactor = 50);
CREATE TABLE sum_cluster_class (partition_id uuid NOT NULL REFERENCES cluster_partitions(id) ON DELETE CASCADE,
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  cluster_uid uuid NOT NULL, detector_class_name text NOT NULL, class_id integer NOT NULL, n integer NOT NULL,
  PRIMARY KEY (partition_id, cluster_uid, detector_class_name, class_id)) WITH (fillfactor = 50);
CREATE TABLE sum_cluster_review (partition_id uuid NOT NULL REFERENCES cluster_partitions(id) ON DELETE CASCADE,
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  cluster_uid uuid NOT NULL, detector_class_name text NOT NULL, tab text NOT NULL, n integer NOT NULL,
  PRIMARY KEY (partition_id, cluster_uid, detector_class_name, tab)) WITH (fillfactor = 50);
CREATE TABLE sum_cluster_dup (partition_id uuid NOT NULL REFERENCES cluster_partitions(id) ON DELETE CASCADE,
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  cluster_uid uuid NOT NULL, detector_class_name text NOT NULL, group_id uuid NOT NULL, n integer NOT NULL,
  PRIMARY KEY (partition_id, cluster_uid, detector_class_name, group_id)) WITH (fillfactor = 50);
```

"Live" is not excluded and not holdout, as today's cards (`cluster_cards.py:116`). An item's
`detector_class_name` is the class of the detection the fold uses for machine geometry (15a 3.4
step 6), `''` when none. `sum_cluster_dup.group_id` is the item's group in the current near-duplicate
run (15a 2.3, M19); the distinct count is the number of rows with `n > 0`. `sum_cluster_review.tab`
holds the five `CLUSTER_REVIEW_TABS` (`cluster_cards.py:44-50`). Box partitions (region clusters,
row 41) have no summary.

### 6.2 Who writes them

Two modules write `item_state`: `apply()` (`src/services/record/apply.py`) and the class-merge job
(15a section 7). Both keep the summaries through one function, so a test (`rg` ratchet in
`tests/storage/`) fails when any other module writes `item_state`, `box_state` or a summary table.

`src/domain/tallies.py` (new, pure): `tally_rows(inputs) -> Tallies`, one `Counter` per summary
table from per-item **tally inputs**: `photo_key`, photo `source`, `class_id`, `class_source`,
`label_source`, validated, excluded, holdout, `updated_at`, region status, validated and verifier,
detector class and confidence, the box `(detector, state)` pairs, and from slices 5 and 6 the
memberships of the partitions in `cluster_activations` with their review-tab inputs and the item's
near-duplicate group. Review-tab membership is one predicate per tab in `src/domain/review_tabs.py`
(new), and the review queue's SQL tab predicates are generated from the same definitions; a
property test asserts the SQL and the Python predicate select the same rows on generated data.

`src/db/repos/tallies.py` (new): `tally_inputs(session, item_keys)` (one statement for the items,
one for their boxes, both by `item_key = ANY`), `apply_tally_deltas(session, project, before,
after)`, `rebuild_tallies(session, project, *, partition_id=None)`.

### 6.3 How `apply()` keeps them, statement by statement

Inside the caller's transaction, after `lock_items` (rank 6) and before anything is written:

1. `before = tally_inputs(session, locked)`. Skipped when no input can change a tally key: a
   `TALLY_NEUTRAL_TABLES` set beside `RANK_NEUTRAL_TABLES` (`src/db/repos/results.py:80-87`),
   holding `probe_predictions`, `item_scores`, `item_projections`, `assist_answers`,
   `raw_label_clusters`, `photo_derivatives`, `photo_signals`, `photo_places`, `entity_edges`, with
   the same classification test (`tests/storage/test_apply_offline.py`); claims always count.
2. The existing steps: `append_claims`, `upsert_results`, `load_item_inputs`, `fold_item`,
   `upsert_item_states`, `upsert_box_states`, `delete_states`.
3. `after = tally_inputs(session, locked)`; a vanished item contributes nothing, a new one nothing
   before.
4. `pg_advisory_xact_lock_shared(<tally key of the project>)` (section 6.4), then per summary table
   one `INSERT ... ON CONFLICT (<key>) DO UPDATE SET n = <t>.n + EXCLUDED.n` (and the summed
   measures) of the non-zero deltas `after - before`, rows **sorted by key**. `max_updated_live`
   takes `greatest(old, new)`: it can only grow, so a member that leaves keeps the cluster's newest
   time until the next rebuild (accepted; the card shows a time no older than the truth).
5. `photo_counts` the same way per photo (insert at ingest, delta after).

Ingest calls `apply(..., photo_scope=True)` for new items (none before, all after), so a photo's
items are counted once in the same transaction that stores them; the ingest transaction also adds
`sum_photos` `+1` per new photo, and the capture writer moves a photo between `located` rows when a
position appears or is purged.

**Lock order.** The summary rows and the advisory lock are a new **rank 8** after `faces` (rank 7)
of 15a section 7: every writer takes them last, rows in ascending key order, so writers wait but
never deadlock ("acquire locks on multiple objects in a consistent order",
[explicit locking](https://www.postgresql.org/docs/18/explicit-locking.html)). Under READ COMMITTED
an upsert that finds the row locked waits, then applies its increment to the committed value
([transaction isolation](https://www.postgresql.org/docs/18/transaction-iso.html)); `ON CONFLICT DO
UPDATE` is atomic under concurrency (INSERT page).

**Hot rows.** The `sum_items` row of the class a person is labelling (source, class, `human`, the
label source, validated, live) and the `sum_cluster` row of the cluster being labelled: every label
of that class or cluster updates it, and concurrent labellers of the same class queue on it for the
rest of their transaction. The upsert is the last statement before `COMMIT`, so the hold time is
about one commit. A bulk chunk of 1,000 items folds its deltas into a few rows. `photo_counts` rows
are per photo and not hot. The ceiling per hot row is about one commit per commit latency: to be
measured in the write-under-read run.

### 6.4 Rebuild (R7)

`rebuild_tallies` runs as a job under the job lease (`src/clients/job_lease.py`), in one transaction
per project: `pg_advisory_xact_lock(<tally key>)` (exclusive), `DELETE` the project's rows of each
summary, `INSERT ... SELECT ... GROUP BY` from `item_state`, `photos`, `detections`, `box_state` and
the memberships, using the key expressions of `tally_rows`. Writers take the same key shared only
at step 4, so every order is correct: a writer that read `before` ahead of the rebuild applies its
delta on top of the rebuilt (committed old) state; a writer already holding the shared lock makes
the rebuild wait until it commits, and the rebuild then counts its new state. Advisory locks:
[functions-admin, advisory lock functions](https://www.postgresql.org/docs/18/functions-admin.html#FUNCTIONS-ADVISORY-LOCKS).

The cluster summaries of a partition are built (`rebuild_tallies(..., partition_id=)`) in the
transaction that names it `pending_partition_id` or `serving_partition_id` in `cluster_activations`
(a rollback to a retained partition rebuilds it first), and dropped by `ON DELETE CASCADE` when the
partition is pruned. A near-duplicate run that becomes current rebuilds `sum_cluster_dup` of the
active partitions in its flip transaction.

The derived-store test (`tests/db/test_tallies_rebuild.py`, slice S1a): seed, write through
`apply()` with random claims and results from two sessions, drop every summary row, rebuild,
compare with the incrementally kept rows, and compare both with `GROUP BY` over the base tables.

### 6.5 Cost per label write

Each `apply()` that can change a tally adds two `tally_inputs` reads of the locked items (index
lookups by key) and one upsert per summary table with deltas (at most two rows per table for a
single label). Not measured: the write-under-read run (`w_sum`, 1,000 relabels per second) is
pending, so the cost is **to be measured**; the bar is the label route's own p99 and the ingest
table of plan 13.

### 6.6 Summary candidates not built

`sum_embedding (project, proposal_name, embedding state)` for row 14, kept by ingest, the vector
writers and the gap writer, is built only if row 14 misses the dashboard bar at 10 M.

## 7. Sorts

| Sort id (`crop_browse.py:17`, `review_sorts.py`, `image_view.py:66-80`) | Key, table | Served by |
|---|---|---|
| `updated_at` | `item_state.updated_at` | X1; with a class filter X2 |
| `created_at` | `item_state.created_at` | top-N sort over the filtered set (to be measured) |
| `confidence` | `item_state.class_confidence` | X4 on live unvalidated lists (`check_machine`, `confidence:asc`); top-N otherwise |
| `crop_area_norm`, `crop_rank_in_image`, `validated`, `class_name` (registry order) | `item_state` | top-N sort over the filtered set (to be measured) |
| `detector_confidence` (`to_label` default, `review_sorts.py:342`) | `detections.confidence` | X8 drives the walk with a semi-join to `item_state` (to be measured) |
| `blur_lap_ratio`, `detector_class_name` | `detections` | top-N (to be measured) |
| `cluster_score` | `cluster_memberships.score` | X23 (whole partition), X22 (one cluster) |
| `mistakenness_score`, `uniqueness_score` | `item_scores.score` | X20 |
| `probe_pred_entropy`, `disagreement_entropy_asc` (`review_sorts.py:344`) | `probe_predictions.entropy` | X21 |
| `region_score` (`REGION_SCORE_SORT`, `review_sorts.py:80`) | `item_state.region_max_score` | top-N over the regions queue (a bounded queue) |
| `crop_id:asc` (audit, `AUDIT_SAMPLE_ORDER`) | `item_state.crop_id` | top-N over the audit sample (bounded) |
| images `indexed_at`, `taken_at`, `file_name`, `unlabeled_count` | `photos.photo_key` (insert order, section 4), `photo_capture.taken_utc`, `photos.storage_key`, `photo_counts.unlabeled` | X9, X13, X12, X24 |

A top-N sort keeps only `LIMIT` rows in memory but reads every row of the filtered set; on a whole
10 M project that is a scan of `item_state` (2.9 GB, 15a 2.4). `item_state` holds five B-trees
(15a 2.6), so not every column gets one. Section 11, question 1.

Planner statistics: the flags and the class source are correlated (validated rows are `human` or
`external_label`), which the planner's independence assumption misjudges. 15a 2.6 adds `CREATE
STATISTICS st_item_state_class (dependencies, mcv) ON class_id, class_source, class_validated,
excluded, holdout FROM item_state` ([CREATE STATISTICS](https://www.postgresql.org/docs/18/sql-createstatistics.html),
[extended statistics](https://www.postgresql.org/docs/18/planner-stats.html#PLANNER-STATS-EXTENDED))
and `ALTER COLUMN class_id SET STATISTICS 1000` so the most-common-values list covers a registry of
up to 1,000 classes ([ALTER TABLE](https://www.postgresql.org/docs/18/sql-altertable.html)).

## 8. Text search: the trigram index

**Box text** (today the `text_fold` keyword and its `text_fold.sub` wildcard field,
`region_text_search.py:1-11`; the review tab's case-insensitive wildcard). `box_state.text_fold` is
already folded by `src/domain/region_text.py` (normalised, upper case, confusables mapped), so case
needs no `ILIKE`. Index X7: `CREATE INDEX ix_box_state_text_trgm ON box_state USING gin (text_fold
gin_trgm_ops) WHERE text_fold IS NOT NULL`. pg_trgm's GIN class supports "trigram-based index
searches for `LIKE`, `ILIKE`, `~`, `~*` and `=` queries" ([pg_trgm](https://www.postgresql.org/docs/18/pgtrgm.html)).

| Match (`TextMatch`) | SQL |
|---|---|
| `exact` | `b.text_fold = :key` |
| `fragment` | `b.text_fold LIKE '%' \|\| :frag_escaped \|\| '%'` (`%`, `_`, `\` escaped) |
| `fuzzy` (`FUZZY_FUZZINESS = 1`, transpositions, `region_text_search.py:29-30`) | candidates `b.text_fold % :key` under `SET LOCAL pg_trgm.similarity_threshold = :t` (default 0.3, pg_trgm page), then the Damerau-Levenshtein distance at most 1 in `src/domain/region_text.py` over the candidates; `:t` set by measurement on set P plates |
| `same_text_as` (`same_text_key`, `region_text_search.py:37`) | the box's own `text_fold` by `(item_key, box_id)` (`uq_box_state_item_box`), then `exact` |

Items are `SELECT DISTINCT b.item_key ... WHERE b.project_id = :p AND <match>`, used as an `EXISTS`
predicate inside P1, P8 and P9. "A pattern with no extractable trigrams will degenerate to a
full-index scan" (pg_trgm page): a fragment shorter than three characters is refused with 400
`text_too_short` as a served sentence, which is a new refusal (15c contract addendum). Measured:
sequential scan `s10_text` 2244.7 / 9444 at 10 M; GIN 0.5 ms at 1 M single client (`notes.md` 4);
10 M with GIN pending (`pg_ext`). The GIN index carries no project column; several projects share
it and the project test is a recheck. A large deployment with many projects measures whether
`btree_gin` (a composite `(project_id, text_fold)` GIN) is needed; not now.

**Item text lines** (today `item_text_tokens` with a `prefix` per word, `src/domain/item_text.py:89`,
`:110`): `text_lines.text` of an item, index X25 `gin (lower(text) gin_trgm_ops) WHERE item_key IS
NOT NULL`; a word `w` matches `lower(l.text) ~ ('\m' || :w_escaped)` (word-start regular
expression, supported by the same index). `item_text_tokens` goes (15a 5.2).

## 9. Box rows: the index and the page

Today a page of items with matching boxes sorts by the newest `detected_at` of any box (nested sort,
`routers/curation/regions.py:137-149`) and reads matching boxes through `inner_hits`. The spike
walked objects by `updated_at` with a semi-join to boxes: 194.7 / 593 deep, and the exact total
600 / 2496 (`s8_*`).

Design: index X6 `ix_box_state_rows (project_id, state, detector, item_key DESC)`, and items in
**newest-item-first** order:

```sql
-- the page's items: distinct items with a matching box, newest first
SELECT DISTINCT b.item_key FROM box_state b
WHERE b.project_id = :p AND b.state = :state AND b.detector = :det
  AND b.item_key < :after_item_key
  AND EXISTS (SELECT 1 FROM item_state s WHERE s.item_key = b.item_key AND <item filter>)
ORDER BY b.item_key DESC LIMIT :limit_plus_one;
-- their matching boxes (rows)
SELECT b.* FROM box_state b
WHERE b.item_key = ANY(:keys) AND b.state = :state AND b.detector = :det
ORDER BY b.item_key DESC, b.ordinal;
```

The index delivers `item_key` in order inside the equality prefix, so `DISTINCT` is a unique step
over sorted input that stops after the page's items, reading at most the page's boxes (an item holds
at most 74). Filters on box state or detector alone use the prefix that applies; a filter with
neither walks X1-ordered items with an `EXISTS` on `uq_box_state_item_box`. Inside one box cluster
the order is the box membership score (X22), as today's cluster branch (`regions.py:120-135`).
`total_rows` is exact from `sum_box` for a state and detector filter, else P4; `total` (items) is P4
over the `DISTINCT` form. `rows_truncated` goes (no inner-hits window). The order changes from
newest detected box to newest item (15c contract addendum). To be measured at 10 M against the
spike's 194.7 / 593.

## 10. Job readers: streaming large sets

| Reader | Form | Why |
|---|---|---|
| vectors for clustering, near-duplicates, region false-positive rows, outlier and diverse orders, re-index sampling (rows 8, 59, 61, 64) | P11 `COPY (SELECT ... ORDER BY subject_key) TO STDOUT (FORMAT BINARY)`, read with psycopg's `cursor.copy()` and the pgvector adapter (15a 6.4) | one sequential read; binary avoids parsing 1,024 numbers per row ([COPY](https://www.postgresql.org/docs/18/sql-copy.html)); 1 M vectors in about 14 s (plan 15 section 1) |
| exports (row 58), combine (67), import reconcile (68) | one `REPEATABLE READ READ ONLY` transaction, a server-side cursor (`DECLARE ... NO SCROLL CURSOR`, fetched in batches of 2,000) in `photo_key, item_key` order | the export must see one state of the labels (a consistent snapshot); a server-side cursor holds no result in client memory ([DECLARE](https://www.postgresql.org/docs/18/sql-declare.html), [SET TRANSACTION](https://www.postgresql.org/docs/18/sql-set-transaction.html)) |
| long walks that write as they go: reprocess targets, requeue, reclassify, scores, auto-label, source backfill, class merge (rows 50 to 57, 66) | keyset batches `WHERE <scope> AND key > :last ORDER BY key LIMIT 5000`, one short transaction per batch | a transaction open for minutes holds back the xmin horizon, so vacuum cannot remove the dead versions every label leaves on `item_state` ([routine vacuuming](https://www.postgresql.org/docs/18/routine-vacuuming.html)); these jobs tolerate rows changing between batches |
| several readers of one snapshot (a parallel export) | `pg_export_snapshot()` in the leader, `SET TRANSACTION SNAPSHOT` in each reader ([functions-admin, snapshot synchronization](https://www.postgresql.org/docs/18/functions-admin.html#FUNCTIONS-SNAPSHOT-SYNCHRONIZATION)) | only if one export reader is too slow; not built now |

Bound: a snapshot transaction above about 10 minutes is split into keyset batches, and the job
records the snapshot's start time in its result so a reader knows the export's moment. The pool's
`idle_in_transaction_session_timeout` of 02a stays the guard against a stuck reader.

## 11. Questions for the coordinator

1. **Sorts with no index on large lists.** Seven crop sort ids (section 7) are top-N sorts over the
   filtered set; on a whole 10 M project that reads all of `item_state`, likely seconds, where
   OpenSearch sorts any field from doc values. Evidence: five B-tree budget on `item_state` (15a
   2.4); `crop_browse.py:17` lists 13 sort fields. Recommendation: serve them and measure in the S1b
   sweep; if a whole-project top-N misses 50 / 200 ms, the route answers 400 `sort_needs_filter`
   above a capped count of 100,000 rows, with a served sentence, rather than adding indexes that
   every label write pays for.
2. **Box rows order newest item first** instead of newest detected box (section 9). Evidence: the
   nested `mode: max` sort has no index form; a box's `detected_at` and its item's ingest order
   agree except after a re-run. Recommendation: accept; the wire field set is unchanged.
3. **Worker poll on `item_region_results`** (row 49), not on the ledger that 15a m7 calls the only
   work list. Evidence: the cascade is a per-item state machine with two pending statuses
   (`PENDING_STATUSES`, `src/config/region_state.py:76`), which a per-photo ledger row cannot hold.
   Recommendation: accept; the ledger stays the work list of whole-photo re-runs.
4. **Two encodings of the review tabs** (SQL for the queue, Python for the card counts), kept equal
   by a property test (section 6.2). Evidence: `review_counts` is on every card (`cluster_cards.py:163-166`)
   and a summary cannot run SQL per write. Alternative: drop per-tab counts from the cards (a wire
   change). Recommendation: keep both encodings with the test.
5. **Filtered cluster cards.** Cards with an item filter that is not a summary key (anything but
   the detector class) fall back to P13 over the filtered members; for a broad filter at 10 M that is
   the raw `s3` cost. Recommendation: measure on set P with the filters the cluster screen offers;
   if it misses, offer only summary-key filters on the cards.

## 12. Trace: every shape in exactly one slice

| Slice | Shapes |
|---|---|
| S1a | 31, 34, 54, 55, 78, 80 |
| S1b | 1, 3, 4, 5, 7, 9, 24, 25, 26, 27, 28, 29, 30, 32, 33 |
| S2 | 2, 15, 16, 19, 36, 38, 43, 46, 47, 66 |
| S3 | 14, 20, 21, 22, 23, 35, 37, 59, 64, 65 |
| S4 | 6, 40, 42, 49 |
| S5 | 45, 61, 63 |
| S6 | 8, 10, 11, 12, 13, 41, 60, 79 |
| S7 | 39, 58, 67, 68 |
| S8 | 17, 18, 44, 48, 50, 51, 52, 53, 56, 57, 76, 77 |
| S9 | 62, 69, 70, 71, 72, 73, 74, 75 |

6 + 15 + 10 + 10 + 4 + 3 + 8 + 4 + 12 + 8 = 80, each number once.
