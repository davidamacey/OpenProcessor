# 15b: storage reads, every query shape on the record

| Field | Value |
|---|---|
| Status | plan, revision 1 (2026-11-15): the read map of plan 15 section 12, written after the gate A measurement at 1 and 10 million objects (8 clients held); revision 1 folds in the independent review of 15b and 15c and the coordinator's decisions on it (appendix "Review trace"): the summaries are a delta log (section 6). Companion of [15_storage_plan.md](15_storage_plan.md) and [15a_storage_schema.md](15a_storage_schema.md); the cut-over slices that build it are [15c_storage_slices.md](15c_storage_slices.md) |
| Code base read | `int/storage` at `7cadf2968` (the integration code of `int/b4-stateless` plus storage step 1: `src/domain/claim*.py`, `fold*.py`, `results.py`, migration `0010_storage_record`, `src/db/repos/`, `src/services/record/apply.py`) |
| Checklist | the storage inventory of 2026-11-13 (`artifacts_local/speed_1113/storage_inventory.md`, gitignored), section 3: shapes 1 to 80 with file:line. Its paths are relative to `src/services/curation/` unless they start with another directory; its line numbers were re-checked on this tree (same code) |
| Measurement | gate A (`artifacts_local/speed_1113/v2_read_shapes/`, gitignored): `notes.md`, `queries_pg.sql`, `summary_pg.sql`, `raw/time_pg_1m.json`, `raw/time_os_1m.json`, `raw/time_pg_10m.json`, `raw/time_os_10m.partial.json` |
| Decided by the coordinator | page fetches are SQL on the record; exact totals only from summary keys, else a capped count; dashboards read summary tables kept by the writer; cluster placement is the side table; text substring search is a trigram index; the search engine leaves the curation stack (section 1.2 of the inventory stays until phase 3); the fallback for a pending measurement that fails is additive (a search copy without vectors for the failing shapes only) |
| Revision 2 (2026-11-16) | measurement round 3 folded in (`artifacts_local/speed_1113/m3_storage/results_round3.md`, `notes.md`, cells under `raw/`, gitignored). The 12 statements that had to change to run are corrected where they live (list in section 14.1). Every miss has its cause, options and one chosen change in section 14; round 4 (section 15) measures only what is still open; section 16 says what each Rust record slice of plan 18b takes from this file. The backend that builds this is Rust (plan 18 section 3.2): the statements here are its specification, the Python module names are where today's behaviour lives |

**Revision 2 in one paragraph.** The delta log, the compactor, the rebuild, the capped counts and
the several-class pages met their bars at 10 million items. Changed by the measurement: cluster
cards read one card row per cluster, kept by the compactor (P7, 6.1, 6.5); box text uses a B-tree
for exact, variant and fuzzy matches and a narrow project-bound search function for fragments,
because row security keeps the trigram index from being used (section 8); the sort path reads its
key from `item_state` where a copy is there (section 7); compactor batch 10,000 every 0.2 s, count
budget 50 ms, `item_state` fillfactor 90, GIN `fastupdate` on with a 512 kB pending list. Open until
round 4: the sort threshold and the band where both sort paths are slow, the cards under an item
filter, reads while writers run, and two owner questions (section 14.3).

## 0. How to read this file

Section 1 lists the measured numbers this file uses and what is still running. Section 2 gives the
SQL of every distinct read pattern (P1 to P15), with real table and column names of `0010`. Section 3
maps all 80 shapes, one row each. Sections 4 to 10 are the designs the rows point at: the list cursor,
the capped-count rule, the summary tables and their delta log, the sort catalogue, the trigram index,
the box-row index and the job readers. Section 13 is measurement round 3: the numbers a slice waits
for before it starts (15c "entry conditions"); section 14 its outcome, miss by miss; section 15
round 4; section 16 the map to the Rust slices. Indexes are numbered `X1`..`X33`; their DDL is in 15a
section 2.6 (migration `0011_storage_indexes`), summary tables in section 6 here.

"to be measured" means: no cell of gate A times this query on this schema; either section 13 times it
before the slice starts, or the first wave sweep after the slice times it on set P and on the
10 million spike tables (section 1.3). Names of new modules are given where a slice creates them;
everything else is a name in the code. Slice names are those of 15c revision 1 (S1a-i, S1a-ii, S1b-i,
S1b-ii, S2a, S2b, S3, F, S4, S5, S6, S7, S8a, S8b, S8c, S9).

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

Settings and indexes of the spike: `notes.md` sections 2 and 3. Its tables (`objects`, `boxes`,
`images`, text ids) differ from the record's, so a spike number guides an access path; it does not
measure the record's query.

### 1.2 Not measured yet at revision 1 (round 3 has since measured them: section 14)

| Pending | What it decides here | Until then |
|---|---|---|
| `pg_sum`: shapes 3, 5, 6 read from `summary_pg.sql` tables | the dashboard reads of section 6 (rows 10, 11, 15, 16) | the design stands; section 13 item M10 adds the delta branch; the fallback is additive (section 1.4) |
| `pg_pl`: placement side table (`pl_*` queries) | the cluster-joined reads (rows 1 with a cluster filter, 2, 8, 10 to 13) | the join form of `pl_*` is used as written; section 13 item M10 |
| `pg_ext`: trigram GIN at 10 M (1 M single client: 89 ms to 0.5 ms, `notes.md` section 4) | section 8 | section 13 item M9, before S4 |
| write-under-read (`w_sum`, hot-row upserts) and the HOT ratio | replaced: `w_sum` measured the hot-row design this revision drops; the delta log and the index set are measured by section 13 items M1 to M5 before S1a-i | entry condition of S1a-i |
| OpenSearch dashboard shapes at 10 M | only the comparison | nothing waits on it |

### 1.3 How "to be measured" is closed

Numbers a design depends on are measured before the slice starts (section 13; 15c entry conditions).
The rest: the coordinator's first wave sweep after a slice runs each new query of that slice through
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

**P1, an item page on an `item_state` sort** (spike `s1a_first`, `s1a_deep`). One class, one
validated value (X2 is `(project_id, class_id, class_validated, updated_at DESC, crop_id DESC)`, so
"unvalidated in a mostly validated class" is one index range):

```sql
SELECT s.item_key, s.crop_id, s.updated_at
FROM item_state s
WHERE s.project_id = :p AND s.class_id = :c AND s.class_validated = :v
  AND NOT s.excluded AND NOT s.holdout                                   -- the rest of the filter (below)
  AND (s.updated_at, s.crop_id) < (:after_updated_at, :after_crop_id)    -- pages after the first
ORDER BY s.updated_at DESC, s.crop_id DESC
LIMIT :limit_plus_one;
```

**Several classes or validated values** (a class list, a merged family, no validated term): an `=
ANY` on a column before the sort column yields class order, so the sort would read every row of the
classes ([multicolumn indexes](https://www.postgresql.org/docs/18/indexes-multicolumn.html),
[indexes and ORDER BY](https://www.postgresql.org/docs/18/indexes-ordering.html)). For up to **16
class ids** the reader emits one parenthesised branch per (class, validated value), each the form
above with its own `ORDER BY ... LIMIT :limit_plus_one`, joined by `UNION ALL` under an outer `ORDER
BY updated_at DESC, crop_id DESC LIMIT :limit_plus_one` ([SELECT, UNION clause](https://www.postgresql.org/docs/18/sql-select.html#SQL-UNION)).
A branch reads at most `limit + 1` entries ("the first n rows can be retrieved directly", ordering
page), so the outer step orders at most 32 x 61 rows whether the planner merges (Merge Append) or
sorts; the S1b-i (DB) test asserts with `EXPLAIN` that every branch scans X2. Above 16 ids the
two-path rule of section 7 applies. `:family` drops a merged class once its merge job completed (15a
section 7): it has no rows left.

Filter translation (today `item_filter_clauses`, `item_filter.py`): class, `class_source`,
`label_source`, validated, excluded, holdout, `area`, `rank`, `import_ids && :ids` are columns of
`item_state`; image-level predicates become `EXISTS (SELECT 1 FROM photos p / photo_capture c WHERE
... = s.photo_key)`; detector class, detector confidence, blur and proposal name become `EXISTS` on
`detections`; cluster (S6) `EXISTS` on `cluster_memberships` of `:serving` (or `s.moved_*`, 15a
4.3 layer 5); near-duplicate group (S5) `EXISTS` on `near_dup_members` of the current run; box
predicates (S4) `EXISTS` on `box_state`; text (S4) section 8; embedding state (S3) a vector row or a
`vector_gaps` row; projection window (S8b) `EXISTS` on `item_projections`. The translation is one
package, `src/db/queries/item_filter/` (new): a registry and **one module per family** (`core`,
`photos`, `vectors`, `boxes`, `near_dups`, `clusters`, `projections`, `review`), so lanes of one
wave never edit one file (m8); an unregistered family raises `FilterNotMovedError`. Every item
reader uses it, as every reader uses `item_filter_clauses` today. `tests/storage/test_item_filter_sql.py`
walks `ItemFilter.model_fields` (`item_filter.py:29-91`) and the route's `geo_polygon`
(`_item_filter_params.py:164`) and fails on a field with neither an SQL form nor a
`FilterNotMovedError` entry. Named forms: `exclude_class_names` is `NOT (coalesce(s.class_id, -1) =
ANY(:excluded_family))` (no class passes, as the `must_not` term); `review_status` and `origin`
translate and negate today's clauses value by value (with `review_dismissed_at` for `pending`,
`import_ids` for `origin=detector`); `image_ids` (up to 5,000) is `s.photo_key = ANY(:keys)` after a
lookup by `photos (project_id, image_id)`; `geo_bbox` and `geo_polygon` are `point(c.lon, c.lat) <@
box(...)` and `<@ polygon(:path)`, served by GiST X14 ([geometric functions](https://www.postgresql.org/docs/18/functions-geometry.html),
[GiST operator classes](https://www.postgresql.org/docs/18/gist.html#GIST-BUILTIN-OPCLASSES)).

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

**P3, an exact total from a summary table** (section 6.4: base rows plus the deltas not yet
compacted, one statement):

```sql
SELECT coalesce(sum(n), 0) FROM sum_items_now
WHERE project_id = :p AND class_id = ANY(:family) AND NOT excluded AND NOT holdout;
```

**P4, a capped count** (section 5), with the cap of its caller and the count budget:

```sql
SAVEPOINT capped_count;
SET LOCAL statement_timeout = :count_budget_ms;
SELECT count(*) FROM (
  SELECT 1 FROM item_state s WHERE s.project_id = :p AND <the page's filter> LIMIT :cap_plus_one
) t;
RELEASE SAVEPOINT capped_count;   -- on 57014 (query_canceled): ROLLBACK TO SAVEPOINT, answer total_unknown
SET LOCAL statement_timeout TO DEFAULT;
```

The savepoint keeps the page's transaction usable after a cancelled count ([SAVEPOINT](https://www.postgresql.org/docs/18/sql-savepoint.html),
[statement_timeout](https://www.postgresql.org/docs/18/runtime-config-client.html#GUC-STATEMENT-TIMEOUT)).

**P5, locate** (today `list_locate.py:146-158`): (a) the item and its sort values, `SELECT
s.updated_at, s.crop_id FROM items i JOIN item_state s USING (item_key) WHERE i.project_id = :p AND
i.crop_id = :crop AND <filter>` (`uq` key of `items`); absent means `not_found` or `filtered_out` as
today. (b) its rank, a capped count of the rows before it (P4 with the list cap): `SELECT count(*)
FROM (SELECT 1 FROM item_state s WHERE s.project_id = :p AND <filter> AND (s.updated_at, s.crop_id) >
(:k, :t) LIMIT 10001) t`. (c) the previous row for `cursor_before`: the same filter, `(s.updated_at, s.crop_id) >
(:k, :t) ORDER BY s.updated_at ASC, s.crop_id ASC LIMIT 1`. `cursor_after` is the item's own keyset
values; neither cursor needs the rank.

**P6, a photo page with card facts** (spike `s7_first`, adapted; the counts come from
`photo_counts` plus its pending deltas, section 6):

```sql
SELECT p.photo_key, p.image_id, p.source, p.storage_key, p.created_at, c.taken_local, k.items, k.labeled,
       k.validated, k.unlabeled, k.in_holdout, b.top
FROM (SELECT photos.photo_key FROM photos
      WHERE photos.project_id = :p AND photos.photo_key < :after_photo_key   -- sort indexed_at:desc (section 4)
        AND EXISTS (SELECT 1 FROM item_state s                               -- only with an item filter (m1)
                    WHERE s.photo_key = photos.photo_key AND <item filter>)
      ORDER BY photos.photo_key DESC LIMIT :limit_plus_one) pg
JOIN photos p USING (photo_key)
LEFT JOIN photo_capture c USING (photo_key)
LEFT JOIN LATERAL (SELECT sum(items) AS items, sum(labeled) AS labeled, sum(validated) AS validated,
                          sum(unlabeled) AS unlabeled, sum(in_holdout) AS in_holdout
                   FROM photo_counts_now n WHERE n.photo_key = p.photo_key) k ON true
CROSS JOIN LATERAL (
  SELECT jsonb_agg(jsonb_build_array(t.crop_id, t.class_id, t.rank, t.bbox) ORDER BY t.rank, t.crop_id) AS top
  FROM (SELECT crop_id, class_id, rank, bbox FROM item_state s
        WHERE s.photo_key = p.photo_key AND NOT s.excluded ORDER BY s.rank, s.crop_id LIMIT :box_limit) t) b
ORDER BY pg.photo_key DESC;
```

The item-level filter sits inside the limited subquery, so the page is never cut short after the
`LIMIT`; it is a semi-join with no cap, so the capped join of today (`JOIN_MAX_IMAGES`,
`image_join.py:132`, `join_truncated`) goes. The counts are a `LEFT JOIN`: a photo with no item has
its zero `photo_counts` row from the ingest transaction (section 6.2) and reads `no_items`
(`image_view.py:80`).

**P7, cluster cards from the card rows** (revision 2; section 14.2 "cluster cards" gives the
measured cause). The screen asks for every card at once (`max_clusters: 2000`,
`frontend/src/lib/api_clusters.ts:218`), sorted and filtered by the server by one of eight card
sorts (`GET /curation/projects/{project}/clusters`, parameter `sort`), and shows 24 at a time
(`clusterListController.svelte.ts:46`). So the read is set-based: one row per cluster from
`sum_cluster_card` (6.1, kept by the compactor, 6.5), plus the pending deltas of the counted
cluster kinds grouped per cluster in one pass, then the card sort and the limit:

```sql
WITH pend AS (                      -- pending deltas of kinds 7 and 9 of the serving partition, one pass
  SELECT cluster_uid,
         sum(n) FILTER (WHERE kind = 7) AS n_live, sum(m[1]) FILTER (WHERE kind = 7) AS n_all,
         sum(m[2]) FILTER (WHERE kind = 7) AS labelled_live, sum(m[3]) FILTER (WHERE kind = 7) AS validated_live,
         sum(m[4]) FILTER (WHERE kind = 7) AS conf_n_live, sum(conf_sum) FILTER (WHERE kind = 7) AS conf_sum_live,
         max(max_updated) FILTER (WHERE kind = 7) AS max_updated,
         sum(n) FILTER (WHERE kind = 9 AND tab = :needs_review_tab) AS needs_review
  FROM sum_deltas
  WHERE project_id = :p AND kind IN (7, 9) AND partition_id = :serving
  GROUP BY cluster_uid)
SELECT c.cluster_uid, c.n_live + coalesce(d.n_live, 0) AS n, c.validated_live + coalesce(d.validated_live, 0) AS validated,
       c.labelled_live + coalesce(d.labelled_live, 0) AS labelled,
       greatest(c.max_updated_live, d.max_updated) AS max_updated,
       (c.conf_sum_live + coalesce(d.conf_sum_live, 0)) / NULLIF(c.conf_n_live + coalesce(d.conf_n_live, 0), 0) AS avg_conf,
       c.needs_review + coalesce(d.needs_review, 0) AS needs_review, c.review_n,
       c.top_class_ids, c.top_class_n, c.top_det, c.dup_groups
FROM sum_cluster_card c LEFT JOIN pend d USING (cluster_uid)
WHERE c.partition_id = :serving AND c.n_live + coalesce(d.n_live, 0) > 0
  AND <card filters: min_size, max_size, validation_state, needs_review, flag, dominant_class, detector_class>
ORDER BY <card sort, tiebreak cluster_uid>
LIMIT :max_clusters;
```

The `m[]` positions follow the base columns of `sum_cluster` in 6.1. Counts on the card (size,
validated, labelled, average confidence, the "to review" count) are exact: base row plus pending
deltas at one snapshot (6.4). The derived fields (top three classes, the detector's top class, the
near-duplicate group count, the per-tab review counts and the flags computed from them) and the order
of the `dominant_class` sort are the compacted values: they lag the counts by at most one
compaction, as a sort on a summary value does (6.4, X24). `top_det` is today's
`detector_top_class`, so the `detector_class` card filter is a column test. Ties among top classes
order by class name after the registry join, as the terms aggregation orders equal counts by key
today. `:needs_review_tab` is the counted tab that defines the card's "to review" count (section 14.3
question 1).

With the item filter `detector_class_name` (the detector's own class, one or several names) the
counts are those of the members of that class: the same columns read from `sum_cluster_now`,
`sum_cluster_review_now` and `sum_cluster_class_now` grouped per cluster with `detector_class_name =
ANY(:det)` (index `ix_sum_cluster_det`, 6.1), one statement, no per-card subquery. With any other
item filter the cards are P13 (row 10, section 14.2 "filtered cards").

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

Inside one cluster the same with `m.cluster_uid = :cluster` (index X22, whose key ends `score DESC
NULLS LAST, subject_key DESC`, so a backward scan gives `score ASC NULLS FIRST, subject_key ASC`
and both directions are one row comparison; nulls are the missing-last segment of section 4). A
review queue on a membership sort follows the two-path rule of section 7: its capped count runs on
the review partial index X4, so a nearly drained queue (few rows left) is read from X4 and sorted by
the joined score, and only a large queue walks X23 with the tab filter, where most rows match.
Round 3 timed the walk alone, without that choice, and its slow tail is that case (section 14.2
"review by score"); round 4 times the routed form.

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
where no summary key answers and the scope is bounded or the reader is a job. It is **one
set-based statement** driven by the filter's most selective index, grouped by the output key (for
cards: `GROUP BY m.cluster_uid` over the filtered items joined to the serving memberships), never one
statement or one `LATERAL` per output row: round 3 measured about 30 ms per card that way (`m10`
`p13_all_per_card_r8`, 6,095 ms for 200 cards), and the cost of the set-based form grows with the
rows the filter matches, which round 4 times (section 15, R4-9).

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

`<@` on `point` and `box` is core; a GiST index on `point(lon, lat)` serves it ([GiST](https://www.postgresql.org/docs/18/gist.html),
[expression indexes](https://www.postgresql.org/docs/18/indexes-expressional.html)). The location
policy applies at read as today (M7): under `drop` nothing runs; `:z` is clamped by
`max_precision(policy)` and the box snapped by `snap_box` (`map_cells.py:62-63`, `:184`, `:228`);
under `store_coarse` the centroids are rounded to `COARSE_DECIMALS` in the statement.

## 3. The 80 shapes

Kind: **page** (SQL page or lookup), **sum** (summary table), **cap** (capped count), **agg** (scoped
aggregation), **stream**, **vector**, **del** (disappears), **kept** (stays on the global
OpenSearch indexes of inventory section 1.2 until phase 3). Slices: 15c (S1a-ii ingest writes,
S1b-i object lists, S1b-ii photos and map, S2a labelling acts, S2b review queue and counts, S3
vectors, S4 regions, S5 near-duplicates and holdout, S6 clustering, S7 import, combine, export,
training, S8a VLM, auto-label and assist, S8b score writers, S8c re-runs, S9 deletions and
acceptance; S1a-i, the summary machinery, owns no shape).

| # | Serves (today) | Slice | Kind | New form | Index or summary | Measured |
|---|---|---|---|---|---|---|
| 1 | `GET /crops`, table view (`routers/curation/crops_list.py:143`, body `crop_listing.py:85`; `table_view.py:134`) | S1b-i | page | P1 + P2 (one branch per class and validated value up to 16 classes); total P3 when the filter is summary keys, else P4; hidden row 7; sorts section 7 | X1, X2, X4; joined sorts section 7 | `s1a_deep` 1.3 / 2, `s1c_deep` 1.4 / 2 (spike indexes, re-keyed here: to be measured) |
| 2 | `GET /review/{tab}` (`routers/curation/review.py:222`; `review_tab_clauses.py`, `review_sorts.py:48`) | S2b | page | P8 for cluster-fit sorts, P1 on X4 for confidence sorts, section 7 for others (two-path rule; the capped count runs on the review partial index X4, so a drained queue is a short read); total P3 from `sum_review` for tabs whose predicate reads only `item_state`, else P4; the script tabs and the VLM-confidence sort of section 7; region text section 8 | X4, X22, X23, X7 | `s2_deep` 2.3 / 12 (column variant); `pl_s2` pending; total was `s2_total` 138 / 257, now P4: to be measured |
| 3 | locate, own row (`list_locate.py:146`; callers `review.py:287`, `crops_list.py:~190`, `routers/curation/images_view.py:62`) | S1b-i | page | P5a; photos: `photos (project_id, image_id)` | `uq` of `items`, `photos` | to be measured |
| 4 | locate, rank and total (`list_locate.py:153`, `:154`; `review.py:337`) | S1b-i | cap | P5b capped rank; total P3 or P4 | as row 1 | exact form `s9a` 48 / 1457, `s9c` 31.9 / 739; capped: to be measured |
| 5 | locate, previous cursor (`list_locate.py:158`) | S1b-i | page | P5c | as row 1 | to be measured |
| 6 | `GET /regions`, training candidates, suspected false positives, cohorts (`region_rows.py:222`; `cohorts.py:33`; `routers/curation/regions.py:255`, `:425`) | S4 | page | P9 (box predicate, text included, in both statements); `total_rows` P3 on `sum_box` for a state and/or detector filter, else P4; `total` (items) P4 | X6, X26, X27, X7 | spike `s8_deep` 194.7 / 593 and `s8_total` 600 / 2496; new design to be measured |
| 7 | hidden counts (`hidden_counts.py:33`) | S1b-i | sum | P3 per hidden kind when the filter is summary keys, else P4 | `sum_items` | to be measured |
| 8 | ordered crop pages `core_first` (`crop_orders.py:74`; `crops_list.py:100`, `:206`) | S6 | page | P8 on `m.score DESC` with the missing-last segment; pool count P4; `outliers` and `diverse` stream the pool's vectors (row 59) | X23 | to be measured |
| 9 | page hydrate by ids (`crop_listing.py:120`) | S1b-i | page | P2 | `items` `uq`, primary keys | to be measured |
| 10 | `GET /clusters` cards (`cluster_cards.py:339`, body `:99`) | S6 | sum | P7 over `sum_cluster_card` plus pending deltas (revision 2); detector-class item filter: P7's grouped form; any other item filter: P13, one statement over the filtered members | `sum_cluster_card`, `ix_sum_cluster_det`, `ix_sum_deltas_cluster` (6.1) | revision 1 form: 332.8 / 504.4 (`m2` `b0_p7_r8`), 461.2 / 1356.4 (`m10` `p7_cards_r8`, H); card rows: round 4 R4-2 |
| 11 | `GET /clusters/representatives` page ids (`clusters.py:197`) | S6 | sum | P7 with `OFFSET :offset` (19,402 clusters in the serving partition at 10 M, `m2` `b0_p7_r8` plan) | `sum_cluster_card` | with row 10 |
| 12 | representatives per cluster (`cluster_cards.py:291`; `vlm_scope.py:147`) | S6 | page | one statement, a `LATERAL` per cluster: serving members joined to `item_state`, not excluded, not holdout, `PARTITION_REPS_SORT` (`cluster_representatives.py:19`), `LIMIT :per_cluster` | X22 | to be measured |
| 13 | `GET /clusters/{cluster_uid}/facets` (`cluster_facets.py:115`) | S6 | page | spike `pl_s4_facets` over `cluster_memberships` joined to `item_state` and `detections` | X22 | column variant `s4_facets` 2.2 / 23; `pl_s4` pending |
| 14 | `GET /detections/summary` (`detections_summary.py:91`) | S3 | agg | P13: `detections.proposal_name` by embedding state (a vector row, else `vector_gaps.reason`, else `deferred`) over the filter | none | to be measured (dashboard bar; summary candidate `sum_embedding` if missed, section 6.6) |
| 15 | `GET /stats/dataset` (`stats_dataset.py:244`, body `stats_dataset_query.py:45`; `pipeline_events.py:65`) | S2b | sum | P3 over `sum_items` (source, class source with and without class, validated, holdout), `sum_region` (status, verifier, by class family), `sum_box` (detector, state), `sum_cluster` (distinct clusters and states, S6); with an item filter outside keys, P13 (today's `request_cache=True`, `stats_dataset.py:244`, goes: the summary path needs no cache; the filtered P13 is timed against the dashboard bar, else section 1.4) | four summaries | raw `s5_stats` 8721 / 17021 (pg1 826 / 1701); `pg_sum` pending |
| 16 | class counts (`class_counts.py:80`; `GET /stats/classes`, `GET /classes`, `workflow_steps.py:18`, `stats_classes.py:95`) | S2b | sum | P3 grouped by class (merge root) and label source over `sum_items` | `sum_items` | raw `s6_classes` 1087 / 7967; `pg_sum` pending |
| 17 | new-class proposals summary (`review_proposals.py:67`) | S8a | agg | `GROUP BY proposed_class` over `class_opinions` with outcome `new_class_pending`, live items; three sample ids per group by a `LATERAL` | X19 | to be measured |
| 18 | raw-label aggregations (`review_terms.py:54`, `:101`; `scripts/curation/cluster_raw_labels.py:159`, `:186`) | S8a | agg | `GROUP BY raw_label` (top `max_terms`) joined to `raw_label_clusters`; the script's page loop becomes keyset on `raw_label` | X18 | to be measured |
| 19 | resolve proposals (`review_resolve.py:53`, `:66`) | S2b | cap | P4 over the proposal predicate, then a keyset walk by `item_key` | X19 | to be measured |
| 20 | kNN over items (`semantic_search_run.py:64`, `:149`, `:256`) | S3 | vector | P12 on the item space; fusion stays in Python | vector table; gate B index | gate B |
| 21 | kNN over box vectors (`semantic_search_box.py:43`) | S3 | vector | P12 on the box space, best box per item | vector table; gate B | gate B |
| 22 | unembedded in scope (`semantic_search_run.py:100`) | S3 | cap | P4 of scope rows with no vector row (`NOT EXISTS`) | vector primary key | to be measured |
| 23 | search source vector (`semantic_search.py:348`; `semantic_search_image.py:110`) | S3 | page | `SELECT embedding FROM vec.<table> WHERE subject_key = :k` | vector primary key | to be measured |
| 24 | `GET /images` (`image_view.py:292`) | S1b-ii | page | P6; sorts `indexed_at` (X9, insert order), `taken_at` (X13 on `taken_local`, as today's `capture.taken_local`, `image_view.py:65`), `file_name` (X12), `unlabeled_count` (X24, compacted base value, section 6.4); geo box and polygon filters P1's forms; item filters as a semi-join; near-duplicate group filter from S5 | X9, X12, X13, X14, X24 | `s7_first` 3.1 / 5, `s7_deep` 3.7 / 6 (taken sort) |
| 25 | image item facts (`image_counts.py:94`) | S1b-ii | page | `photo_counts` for counts, the `LATERAL` of P6 for boxes | X3, `photo_counts` | inside `s7` above |
| 26 | image join ids (`image_join.py:132`) | S1b-ii | del | replaced by the semi-join inside P1, P6, P7, P13 | none | n/a |
| 27 | item-side image join (`image_join.py:172`, caller `map_cells.py:128`) | S1b-ii | del | replaced by `EXISTS` from `photo_capture` to `item_state` | none | n/a |
| 28 | image counts and readiness (`image_view.py:153`, `:344`; `map_cells.py:128`, `:148`, `:218`; `workflow_steps.py:346`, `:353-373`) | S1b-ii | sum | totals P3 on `sum_photos` (source, located), else P4 on `photos`; readiness is always true and goes | `sum_photos` | to be measured |
| 29 | photo detail, file and thumbnail routes (`image_detail.py:56`, `:66`; `images_view.py:81`; `capture_wire.py:64`; `image_serving.py:532`; `image_item_write.py:95`) | S1b-ii | page | P2 of the photo and of its items by rank | `photos` `uq`, X3 | to be measured |
| 30 | `GET /map` (`map_cells.py:128`, `:148`, `:218`) | S1b-ii | page | P15 with the location policy at read (precision clamp, `snap_box`, coarse centroids, nothing under `drop`); extent `min/max(lat, lon)` over the same predicate | X14 | `s11_map` 18.0 / 171 |
| 31 | ingest lookups (`ingest_objects.py:80`; `routers/curation/datasets.py:108`; `ingest_reads.py:51`, `:71`) | S1a-ii | page | (a) `source_identifier = ANY`; (b, c) `storage_key = ANY`; (d) `GROUP BY source, date_trunc('day', created_at)` over 14 days | X11, X12, X10 | to be measured |
| 32 | item get (`crops_list.py:211`, `:256`; `review.py:314`; `item_embed_inline.py:82`; `item_geometry.py:239`, `:244`; `label_undo.py:267`; `clients/occ.py:100`) | S1b-i | page | P2 of one key; the `occ.py` read-merge-write goes in S9 | `items` `uq` | n/a (one probe) |
| 33 | item mget for writers (`curation_opensearch/crops.py:79` and about 20 callers; `occ_upsert.py:144`, `:172`; `ingest_writer.py:532`; `region_text_search.py:48`; `item_scores/mistakenness.py:129`) | S1b-i | page | P2; the conditional-write callers go in S9; job callers switch in their slices | as P2 | n/a |
| 34 | mget exists-only (`image_completeness.py:79`, `:151`; `near_duplicates/holdout.py:170`; `ingest_dedup.py:87`; `near_duplicates/core_content.py:41` on the global index) | S1a-ii | page | `photos.photo_keys_by_image_id`, `items.item_keys_by_crop_id`; `image_completeness`'s reads go (the module in S9); `holdout.py:170` switches in S5; `core_content.py:41` stays (global) | `uq` keys | n/a |
| 35 | mget vectors (`near_duplicates/vectors.py:56`; `embedding_spaces/reindex_encoder.py:76`; `reindex_opensearch.py:343`) | S3 | page | `SELECT subject_key, embedding FROM vec.<table> WHERE subject_key = ANY(:keys)` | vector primary key | to be measured |
| 36 | `POST /selection/summary` (`selection_summary.py:71`, `:94`) | S2b | agg | the guard is P4 with cap `MAX_BULK_SELECTION + 1` (20,001; `item_selection.py:35`, checked at `selection_summary.py:129`), then P13 over the selection once it is under the guard | as P1 | to be measured |
| 37 | pool counts and area scan (`selection/select.py:174`; `selection/pool_fetch.py:61`; `item_selection.py:110`) | S3 | stream | P4 for the pool; keyset stream of `(item_key, area)` | X1 | to be measured |
| 38 | scalar filter counts (`cohorts.py:33`; `audit.py:193`, `:226`; `assist/scope.py:103`; `assist/segment.py:61`; `classes_write.py:128`; `classes/projection_sync.py:61`; `review_empty_reason.py:26`, `:34`; `region_stage_control.py:92`; `workflow_steps.py`; `reprocess.py:205`; `export_readiness.py:114`; `training/preflight_checks.py:181`; `vlm_scope.py:231`; `autolabel/vlm_stage.py:186`; `holdout_freeze.py:80`; `dataset_import/runner.py:174`) | S2b | cap | one function `count_items(filter, *, cap)`: P3 when the filter is summary keys, else P4 with the caller's cap (section 5 table: guards pass their limit, jobs count exactly inside the job); callers outside labelling switch in their slices | summaries | to be measured |
| 39 | `GET /quality/readiness` (`dataset_readiness.py:364`, `:386`, `:388`) | S7 | sum | class by holdout from `sum_items` (P3); photo coverage per class P13 | `sum_items` | to be measured |
| 40 | region profile impact (`region_impact.py:93`, `:137`, `:186`) | S4 | agg | P13 `GROUP BY analyser_id, status` over `item_region_results`, box detector counts from `sum_box` | none (an activation preview, not a page) | to be measured |
| 41 | `GET /regions/clusters` (`region_fp_clusters.py:99`) | S6 | agg | P13 over the box partition's members joined to `box_state` and `item_state`; top hits by a `LATERAL` | X22 | to be measured |
| 42 | region drain (`region_drain.py:60`; `ingest.py:376`) | S4 | sum | P3 `GROUP BY region_status` over `sum_region`, class family filter | `sum_region` | to be measured |
| 43 | field coverage (`field_coverage.py:92`) | S2b | cap | offer test `EXISTS (SELECT 1 FROM <result table> WHERE project_id = :p)`; `total` exact (P3, live items of `sum_items`), `n_scored` P4 with `capped` (`scores.ts:17` shows "10,000+ / N") | primary keys | to be measured |
| 44 | pipeline health and ops metrics (`autolabel/health_snapshot.py:39`; `ops_metrics_refresh.py:99`) | S8a | sum | P3 where the filter is keys, else P4 | summaries | to be measured |
| 45 | holdout stats, strata scan and census (`holdout_freeze.py:80`, `:179`; `holdout.py:143`, `:211`) | S5 | sum | stats P3 (`sum_items.holdout`); strata scan keyset by `item_key`; census `GROUP BY class_id, source` over `sum_items` | `sum_items` | to be measured |
| 46 | label-op current counts (`label_ops/views.py:34`) | S2b | page | `SELECT class_op_id, count(*) FROM item_state WHERE class_op_id = ANY(:ops) GROUP BY 1` | X5 | to be measured |
| 47 | accept-machine and detector census (`accept_machine.py:133`, `:154`; `detector_adoption.py:124`) | S2b | agg | class census P3 (`sum_items`, machine sources, not validated); detector census P13 `GROUP BY detections.class_name` over eligible items | `sum_items` | to be measured |
| 48 | VLM scope (`vlm_scope.py:121`, `:231`) | S8a | agg | per-cluster sizes from `sum_cluster`; the sample `_SAMPLE_SOURCE` becomes a deterministic hash test on `crop_id` | `sum_cluster`, X22 | to be measured |
| 49 | worker pending poll (`scripts/curation/worker/cascade.py:126`; `scripts/curation/vlm_pending.py:211`) | S4 | page | P14 on `item_region_results` pending, joined to `item_state`, `ORDER BY item_key LIMIT :batch`, in-flight keys excluded; the VLM variant's call site switches in S8 | X15 | to be measured |
| 50 | auto-label selection (`autolabel/selection.py:219`, `:240`, `:252`, `:288`) | S8a | stream | sizing inside the job by an exact count, progress from the job's own counters (section 5 table); the scroll becomes keyset over X1 in ascending order | X1 | to be measured |
| 51 | reprocess targets (`reprocess_targets.py:179`, `:221`; `routers/curation/reprocess.py:64`, `:113`) | S8c | stream | keyset by `item_key` over the target filter; a photo by `image_id` | as P1 | to be measured |
| 52 | region requeue preview and run (`region_requeue.py:317`, `:540`) | S8c | stream | preview P3 (`sum_region`, `sum_box`); run keyset of `item_key` through X6 or `item_region_results` | X6 | to be measured |
| 53 | repair and reclassify scans (`registry_reclassify.py:258`; `repair_scan.py:43`; `region_text_repair.py:213`) | S8c | stream | `registry_reclassify` keyset over X2; the two repairs fixed document defects and become ledger re-runs (P14) | X2, X16 | to be measured |
| 54 | image counts upkeep (`image_counts_backfill.py:36`, `:47`; `image_counts_refresh.py:107`) | S1a-ii | del | `photo_counts` kept by the writer (section 6) | none | n/a |
| 55 | source backfill page (`source_backfill/job.py:67`) | S1a-ii | stream | P14: ledger kind `capture_parser` rows not at the current analyser, keyset by `photo_key` | X16 | to be measured |
| 56 | probe job reads (`probe_predictions.py:209`, `:274`, `:314`, `:397`, `:425`) | S8b | stream | P11 of the pool; counts P4; the weighted top N is `ORDER BY <weights expression> LIMIT :n` over `probe_predictions` joined to `item_state` | X21 | to be measured |
| 57 | item scores scroll (`item_scores/job.py:158`, `:246`, `:253`) | S8b | stream | keyset by `item_key` over `probe_predictions` joined to the live, unvalidated items | primary keys | to be measured |
| 58 | export scroll (`export_support.py:161`; callers `export.py:171`, `export_images.py:166`, `export_negatives.py:53`, `region_eval_data.py:131`, `:164`, `export_single_class_rows.py:230`, `:303`, `:315`) | S7 | stream | section 10: one snapshot, server-side cursor in `photo_key, item_key` order | primary keys | to be measured |
| 59 | vector reads in slices (`clustering/pit_pages.py:53`, `:114`; `clustering/vector_read.py:69`, `:123`; `clustering/region_box_rows.py:119`; `clustering/pool_snapshot.py:60`; `near_duplicates/vectors.py:87`, `:120`; `near_duplicates/chain.py:43`; `scripts/curation/backfill_item_ids.py:83`) | S3 | stream | P11 `COPY ... (FORMAT BINARY)` in `subject_key` order (`vector_stream`, new); call sites switch in S5 (near-duplicates), S6 (clustering, region false-positive rows, outlier and diverse orders); `backfill_item_ids.py` goes in S1a | vector primary key | 69,630 rows per second (plan 15 section 1, not gate A) |
| 60 | clustering pre-counts (`clustering/residual_fetch.py:80`; `clustering/outliers.py:77`) | S6 | agg | exact `count(*)` of the pool inside the job (P13); jobs may take a second | as P1 | to be measured |
| 61 | near-duplicate kNN edges (`near_duplicates/candidates.py:80`, `:113`; `rescore.py:67`, `:89`) | S5 | vector | P12 batched per query vector, `k + 1`; the leak partition's `photo_key = ANY` as the filter | gate B | gate B |
| 62 | near-duplicate kNN on the global index (`near_duplicates/core_candidates.py:52`) | S9 | kept | unchanged | global index | n/a |
| 63 | near-duplicate views and status (`near_duplicates/member_views.py:36`, `:52`; `near_duplicates/status.py:204`) | S5 | page | members of a group by `ix_near_dup_members_group` joined to `item_state`; items without a vector P4 | 02d keys | to be measured |
| 64 | embedding-space re-index (`embedding_spaces/reindex_opensearch.py:155`, `:157`, `:205`, `:288`, `:319`, `:343`) | S3 | stream | work list: subjects with no row in the new space's table, keyset by key; the random sample becomes a hash sample; counts exact in the job | vector primary key | to be measured |
| 65 | box-embedding backfill script (`scripts/curation/backfill_region_embeddings.py:106`) | S3 | del | the re-index of the box space (row 64) | none | n/a |
| 66 | class-merge scroll (`clients/opensearch/scroll.py:37`; caller `classes/projection_sync.py:188`) | S2a | del | the merge job walks X2 by class in keyset chunks (15a section 7, class merge) | X2 | n/a |
| 67 | combine reads (`projects/combine/plan.py:426`, `:427`; `combine/image_copy.py:250`; `combine/holdout.py:34`; `combine/warnings.py:28`; `dataset_import/project_source.py:101`) | S7 | stream | keyset over the source project's photos and items by key, the source project bound for the read | primary keys | to be measured |
| 68 | dataset import reads (`dataset_import/project_source.py:137`, `:159`; `image_plan.py:78`; `undo.py:447`, `:490`; `reconcile.py:78`; `runner.py:174`) | S7 | page | P2 and keyset walks by key; undo reads the operation's claims (`claims.live_claims_of_operation`) | `uq_claims_op_subject` | to be measured |
| 69 | kNN on the global indexes (`clients/opensearch/search.py:199`) | S9 | kept | unchanged | | n/a |
| 70 | OCR search (`clients/opensearch/ocr.py:156`, `:221`) | S9 | kept | unchanged | | n/a |
| 71 | global cluster reads (`clients/opensearch/clusters.py:62`, `:123`, `:253`, `:277`) | S9 | kept | unchanged | | n/a |
| 72 | global duplicate groups (`duplicate_detection/service.py:54`, `:81`, `:96`, `:99`, `:105`) | S9 | kept | unchanged | | n/a |
| 73 | faces and persons (`face_identity/identification.py:37`; `face_records.py:41`; `person_management.py:64`, `:149`) | S9 | kept | unchanged | | n/a |
| 74 | face privacy counts (`face_privacy.py:75`, `:87`) | S9 | kept | unchanged | | n/a |
| 75 | image admin (`services/image_admin.py:22`, `:57`, `:71`, `:87`, `:101`) | S9 | kept | unchanged | | n/a |
| 76 | `GET /viz/projection` (`embedding_viz_read.py:90`, `:162`, `:190`) | S8b | page | points keyset by `item_key` over `item_projections` joined to `item_state` and the serving membership; missing count P4 | `item_projections` key | to be measured |
| 77 | assist-loop partition (`assist_loop/partition.py:298`, `:344`) | S8a | agg | `GROUP BY cluster_uid` over serving members under the filter, keyset on `cluster_uid`; answers by P2 | X22 | to be measured |
| 78 | project housekeeping counts (`projects/stats.py:43`; `projects/delete.py:124`; `projects/clone_validate.py:88`; `ingest_policy.py:111`; `embedding_spaces/capacity.py:61`; `embedding_spaces/reindex.py:312`; `scripts/curation/seed_live_harness.py:535`) | S1a-ii | sum | P3 on `sum_items` and `sum_photos`; delete and clone preflight may count exactly (jobs) | summaries | to be measured |
| 79 | periodic ticks (`periodic/cluster_refresh.py:84`, `:85`, `:92`; `periodic/open_vocab_sweep.py:29`) | S6 | cap | P4 of the pool and pending statuses; "new photos" is `EXISTS (SELECT 1 FROM photos WHERE project_id = :p AND created_at > :since)` on X10 (BRIN); the open-vocabulary sweep's call site switches in S8 (ledger, P14) | X10, X16 | to be measured |
| 80 | ingest dedup (`clients/optional_doc.py:18`; `ingest_dedup.py:69`, `:87`) | S1a-ii | page | photo by `image_id` (`uq`), its items by `photo_key` | `uq` keys, X3 | n/a |

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
`"v"` (rows with a value) or `"n"` (rows without one). The reader binds each decoded value with its
column's type (a `crop_id` as `uuid`, a time as `timestamptz`), never as text: a row constructor
compared against text parameters fails to type-check (round 3 had to add `::uuid`, section 14.1
item 5). The cursor also carries the path the first page chose (`"p": "sort"` or `"walk"`, section
7), so deeper pages do not decide again.

**One direction per row comparison (M7).** A row constructor comparison `(a, b) < (x, y)` is
evaluated left to right with one operator ([row-wise comparison](https://www.postgresql.org/docs/18/functions-comparisons.html#ROW-WISE-COMPARISON)),
so it cannot mix directions. Today's `desc` sorts carry an ascending `crop_id` tiebreak, which is
mixed. Rule: **the tiebreak takes the direction of the last sort key**, so every single-key sort is
one row comparison, `<` for `desc` and `>` for `asc`, and one B-tree serves both directions, because
an index scanned backward gives the reverse order ([indexes and ORDER BY](https://www.postgresql.org/docs/18/indexes-ordering.html)).
The visible effect: equal sort values now come in descending `crop_id` order under a `desc` sort
(contract diff, 15a section 10 addendum in 15c). A sort with several keys of mixed directions (none
in `CROP_SORT_FIELDS`, `crop_browse.py:17`; among review sorts only `detector_disagreement_default`,
section 7) uses the expanded form `k1 < :k1 OR (k1 = :k1 AND (k2, t) > (:k2, :t))`, whose cost is an
index range on `k1` with a filter for the tie group. Section 7 lists every crop and review sort with
its keys and directions; S1b-i builds `keyset_clause` for all of them, so the form is chosen once.

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

1. **Exact** when every term of the normalised filter is on a **whitelist** of summary keys of one
   table of section 6: `sum_items` (class family, `class_source`, `label_source`, validated,
   excluded, holdout, photo source), `sum_review` (counted tab, class, holdout), `sum_region` (class
   family, region status, validated, verifier), `sum_box` (detector, state), `sum_photos` (source,
   located), the cluster summaries (partition, cluster, detector class, counted tab), `photo_counts`
   (one photo). `exact_total(filter)` (new, `src/db/queries/totals.py`, S1b-i) returns P3 over the
   `_now` view, or `None`. The whitelist names filter fields and values, not columns: a term that
   reads more than a key is not one (`review_status=pending` also tests `review_dismissed_at`,
   `origin=detector` also tests `import_ids`; m11).
2. **Capped** otherwise: `count_items(filter, *, cap) -> (n, capped, unknown)` runs P4 with the
   page's `FROM` and `WHERE` and the **caller's cap**. A list passes `TOTAL_CAP = 10_000`
   (`src/domain/list_totals.py`, new) and serves `min(n, cap)` with `total_capped`; a guard passes
   its limit plus one (table below). `LIMIT` stops the scan at that many rows ([LIMIT and OFFSET](https://www.postgresql.org/docs/18/queries-limit.html)).
3. **Bounded by time.** A capped count stops at `cap + 1` matches or at the end of its scope, not
   after a number of rows read, so P4 runs under `statement_timeout = :count_budget_ms` (P4); on
   cancel the page carries `total_unknown: true` (15c section 2) and the interface shows "many".
   `:count_budget_ms` is **50 ms** (round 3, M6: the slowest passing count had p99 40.9 ms,
   `m6b_vac` `x4_total_capped_vac_r8`, and the passing counts cancelled at most 0.009% at a 50 ms
   budget, `m6a` `p4_class_daterange_r8_budget50ms`).
4. **Locate** ranks the same way: rows before the item, capped; `index_capped` when the rank reached
   the cap (the filmstrip shows "10,000+ before"). Its cursors do not need the rank (P5).
5. **Hidden counts** (row 7) follow rules 1 to 3 per hidden kind, with a `capped` flag.
6. **Carried**: the first page's `total`, `total_capped` and `total_unknown` ride in the cursor, as
   `CarriedCounts` does today; deeper pages cost their page only.
7. **Exact where it matters.** Dashboards, class counts, (class, validated) and counted tabs read
   summaries. Jobs count exactly inside the job (rows 50, 60, 64, 78) and report progress from
   their own counters; a job total is never a capped count.
8. **Cost.** When the index does not hold every filter column, each candidate is a heap fetch
   ([index-only scans](https://www.postgresql.org/docs/18/indexes-index-only-scans.html)): up to
   `cap + 1` on page 1 of a list whose total is not a key (m5). The end-game cases of the review are
   keys now or short ranges of X2 or X4. Timed by section 13 item M6.
9. **Test.** `tests/storage/test_totals_rule.py` holds the whitelist; the (DB) property test
   `tests/db/test_totals_differential.py` draws random filters and asserts that `exact_total` of
   every exact one equals an uncapped `count(*)`, and that capped ones give `min(count(*), cap)`.

**Every count with a cap, and every interface that reads one** (M5; backend from `rg -n
'count|total|cap' src/services/curation`, frontend from `rg -n '\.total\b' frontend/src/lib`):

| Use | Where | Cap | When capped or unknown |
|---|---|---|---|
| list and page totals (crops, table, review queue, region rows, photo list, matching items) | rows 1, 2, 6, 24 | 10,000 | `total_capped` / `total_unknown` |
| locate rank | rows 3 to 5 | 10,000 | `index_capped` |
| hidden counts | `hidden_counts.py:33` | 10,000 | `capped` |
| bulk selection guard | `selection_summary.py:129` (`MAX_BULK_SELECTION`, `item_selection.py:35`) | 20,001 | above 20,000 refused as today; the selection's own count P13 only below |
| pool fetch guard, sync path | `selection/pool_fetch.py:62` (`OP_SELECT_MAX_N`, `diverse_order.py:97`, `:110`) | `max_n() + 1` | refused as today |
| pool fetch guard, job path | `selection/job.py:106` (`OP_SELECT_JOB_MAX_N`, `diverse_order.py:99`, `:124`) | `job_max_n() + 1` | refused as today |
| sort path choice | section 7 | `T_sort + 1` (section 13 item M7) | walk path |
| vector search path choice | P12 | gate B's upper threshold + 1 | gate B |
| unembedded in scope | `semantic_search_run.py:100` (row 22) | 10,000 | described as "10,000+" |
| field coverage `n_scored` | `field_coverage.py` (row 43) | 10,000 | `capped`; the total is exact (P3) |
| other row-38 callers (`cohorts.py:33`, `audit.py:193`, `:226`, `assist/scope.py:103`, `assist/segment.py:61`, `classes_write.py:128`, `review_empty_reason.py:26`, `:34`, `region_stage_control.py:92`, `workflow_steps.py`, `reprocess.py:205`, `export_readiness.py:114`, `training/preflight_checks.py:181`, `vlm_scope.py:231`, `autolabel/vlm_stage.py:186`, `holdout_freeze.py:80`, `dataset_import/runner.py:174`) | the slice that moves each | a displayed count 10,000; a comparison against a limit, that limit + 1; inside a job, exact | the lane lists each caller with its cap in the commit |
| auto-label sizing and progress | `autolabel/selection.py` (row 50), `autolabel/progress.py:62-75` | none: exact count in the job, then the job's `processed` / `total` counters | never capped |

| Interface read (frontend) | Change when `total_capped` or `total_unknown` |
|---|---|
| `pager.svelte.ts:123`, `:195`; `components/itemFilter/MatchingItemsView.svelte:80`, `:138`; `workspace/views/TableView.svelte:196`; `DetectionsView.svelte:109`, `:153`; `GalleryView.svelte:75`, `:135`; `components/slots/SlotGallery.svelte:248-256`; `components/clusters/ClustersStatusBar.svelte:41`; `clusters/firstLoad.ts:24` | show "10,000+" (or "many") instead of the number |
| `components/review/ReviewHeader.svelte:79-82`, `workspace/views/ReviewView.svelte:145`; `editor/editorController.svelte.ts:169`, `:179`, `components/editor/ImageEditor.svelte:80` | "#n of 10,000+"; the filmstrip shows "10,000+ before" from `index_capped` |
| `review/reviewController.svelte.ts:69`, `:77`, `:144`; `review/reviewSlotActions.svelte.ts:129`; `clusters/clusterController.svelte.ts:394` | no plus or minus one on a capped or unknown total |
| `components/workspace/BulkBar.svelte:181`, `workspace/selection.ts:97-109` (`pressSelectAll`), `workspace/workspaceController.svelte.ts:433` | "Select all 10,000+ matching": the act is by filter; its dry run (`POST /selection/summary`, guard cap 20,001) shows the real number up to 20,000, or the refusal |
| `autoLabelController.svelte.ts:197-198`, `labelOps/opStatusLine.ts:23-25`, `components/ScoresCard.svelte:291` | none: job counters, never capped |
| `scores.ts:17` | "10,000+ / N" (the total is exact) |
| `api_cropRead.ts:240`, `api_clusters.ts:147` | carry the two flags into the pager |

## 6. Summary tables: base rows and a delta log

A writer never updates a summary row. In its own transaction it **inserts** signed delta rows into
one insert-only table, `sum_deltas`; a compactor folds them into the base rows; a reader sums base
rows and pending deltas in one statement. Inserts of fresh keys never wait on each other, so 30
writers labelling one class do not serialise, and a rolled-back transaction leaves no delta.

### 6.1 DDL (migration `0011_storage_indexes`; the cluster family in `0016_cluster_cutover`)

Every table has `project_id` with RLS(02a) (15a section 1). Base key columns are `NOT NULL` with a
sentinel for "none" (`class_id` `-1`, text `''`), as the spike's `COALESCE(class_source, '')`
(`summary_pg.sql:26`), so `ON CONFLICT` always finds its arbiter ([INSERT](https://www.postgresql.org/docs/18/sql-insert.html));
readers map `-1` back to "no class" in one function (`domain/tallies.class_key`), because
`item_state.class_id IS NULL` is the same fact (the `missing` bucket of rows 10 and 15, M7). Counts
may reach 0 and stay as rows. `fillfactor 50` as the spike's (`summary_pg.sql:4`): the compactor's
counter updates change no indexed column and stay HOT ([HOT](https://www.postgresql.org/docs/18/storage-hot.html)).

```sql
-- base rows: written only by the compactor (6.5) and the rebuild's corrections through it (6.6),
-- except the zero photo_counts row a photo insert adds (6.2)
-- PROJECT below = project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE (15a section 1)
CREATE TABLE sum_items (PROJECT, source text NOT NULL, class_id integer NOT NULL,   -- item totals, class counts, statistics
  class_source text NOT NULL, label_source text NOT NULL,
  class_validated boolean NOT NULL, excluded boolean NOT NULL, holdout boolean NOT NULL, n bigint NOT NULL,
  PRIMARY KEY (project_id, source, class_id, class_source, label_source, class_validated, excluded, holdout)
) WITH (fillfactor = 50);
CREATE TABLE sum_review (PROJECT, tab text NOT NULL, class_id integer NOT NULL, holdout boolean NOT NULL,   -- counted tabs (6.2)
  n bigint NOT NULL, PRIMARY KEY (project_id, tab, class_id, holdout)) WITH (fillfactor = 50);
CREATE TABLE sum_region (PROJECT, class_id integer NOT NULL, region_status text NOT NULL,   -- region drain and statistics
  region_validated boolean NOT NULL, region_verifier text NOT NULL, n bigint NOT NULL,
  PRIMARY KEY (project_id, class_id, region_status, region_validated, region_verifier)) WITH (fillfactor = 50);
CREATE TABLE sum_box (PROJECT, detector text NOT NULL, state text NOT NULL, n bigint NOT NULL,   -- box-row totals, requeue preview
  PRIMARY KEY (project_id, detector, state)) WITH (fillfactor = 50);
CREATE TABLE sum_photos (PROJECT, source text NOT NULL, located boolean NOT NULL, n bigint NOT NULL,   -- photo totals
  PRIMARY KEY (project_id, source, located)) WITH (fillfactor = 50);
CREATE TABLE photo_counts (photo_key bigint PRIMARY KEY REFERENCES photos(photo_key) ON DELETE CASCADE, PROJECT,
  items integer NOT NULL, labeled integer NOT NULL, validated integer NOT NULL,   -- an image card (count_fields.py:30)
  unlabeled integer NOT NULL, in_holdout integer NOT NULL) WITH (fillfactor = 90);   -- X24 indexes unlabeled
CREATE TABLE sum_deltas (                     -- insert-only: every change of a counted fact, until compacted
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind smallint NOT NULL,                     -- 1 items, 2 review, 3 region, 4 box, 5 photos, 6 photo_counts,
                                              -- 7 cluster, 8 cluster_class, 9 cluster_review, 10 cluster_dup
  delta_id bigint GENERATED ALWAYS AS IDENTITY,
  source text, class_id integer, class_source text, label_source text,   -- key columns: the kind's own carry the
  class_validated boolean, excluded boolean, holdout boolean, tab text,  -- base table's values and sentinels; the
  region_status text, region_validated boolean, region_verifier text,    -- rest are NULL
  detector text, state text, located boolean, photo_key bigint,
  partition_id uuid, run_id uuid, cluster_uid uuid, detector_class_name text, group_id uuid,   -- 0016 adds the
                                              -- foreign keys of partition_id and run_id, ON DELETE CASCADE
  n bigint NOT NULL,                          -- the signed count (photo_counts: items; cluster: n_live)
  m integer[] NULL,                           -- the kind's other signed integer measures, in base-column order
  conf_sum double precision NULL, max_updated timestamptz NULL,   -- cluster kind only
  PRIMARY KEY (project_id, kind, delta_id)
) WITH (autovacuum_vacuum_scale_factor = 0);   -- and a vacuum threshold equal to the compactor batch (below)
-- one view per base table: base rows and pending deltas, read by every summary reader (6.4)
CREATE VIEW sum_items_now WITH (security_invoker = true) AS
  SELECT project_id, source, class_id, class_source, label_source, class_validated, excluded, holdout, n FROM sum_items
  UNION ALL
  SELECT project_id, source, class_id, class_source, label_source, class_validated, excluded, holdout, n
  FROM sum_deltas WHERE kind = 1;
-- sum_review_now, sum_region_now, sum_box_now, sum_photos_now, photo_counts_now (m[1..4] as labeled,
-- validated, unlabeled, in_holdout) the same; the cluster views in 0016
-- 0016_cluster_cutover: for the partitions cluster_activations names and the runs 6.7 names
-- PART below = partition_id uuid NOT NULL REFERENCES cluster_partitions(id) ON DELETE CASCADE, PROJECT,
--              cluster_uid uuid NOT NULL, detector_class_name text NOT NULL
CREATE TABLE sum_cluster (PART, n_all integer NOT NULL, n_live integer NOT NULL, labelled_live integer NOT NULL, validated_live integer NOT NULL,
  max_updated_live timestamptz NULL, conf_sum_live double precision NOT NULL, conf_n_live integer NOT NULL,
  PRIMARY KEY (partition_id, cluster_uid, detector_class_name)
) WITH (fillfactor = 50);
CREATE TABLE sum_cluster_class (PART, class_id integer NOT NULL, n integer NOT NULL,
  PRIMARY KEY (partition_id, cluster_uid, detector_class_name, class_id)) WITH (fillfactor = 50);
CREATE TABLE sum_cluster_review (PART, tab text NOT NULL, n integer NOT NULL,
  PRIMARY KEY (partition_id, cluster_uid, detector_class_name, tab)) WITH (fillfactor = 50);
CREATE TABLE sum_cluster_dup (PART, run_id uuid NOT NULL REFERENCES near_dup_runs(id) ON DELETE CASCADE,
  group_id uuid NOT NULL, n integer NOT NULL,
  PRIMARY KEY (partition_id, run_id, cluster_uid, detector_class_name, group_id)) WITH (fillfactor = 50);
CREATE INDEX ix_sum_cluster_det ON sum_cluster (partition_id, detector_class_name, cluster_uid);   -- P7 detector form
-- revision 2: one card row per cluster, written only by the compactor (6.5) and the partition fill (6.7);
-- it has no delta kind of its own: its counts are those of sum_cluster and sum_cluster_review rolled up
CREATE TABLE sum_cluster_card (partition_id uuid NOT NULL REFERENCES cluster_partitions(id) ON DELETE CASCADE, PROJECT,
  cluster_uid uuid NOT NULL,
  n_all integer NOT NULL, n_live integer NOT NULL, labelled_live integer NOT NULL, validated_live integer NOT NULL,
  max_updated_live timestamptz NULL, conf_sum_live double precision NOT NULL, conf_n_live integer NOT NULL,
  needs_review integer NOT NULL,              -- the card's "to review" count (section 14.3 question 1)
  review_n integer[] NOT NULL,                -- per counted tab, in the order of the review-tab table
  top_class_ids integer[] NOT NULL, top_class_n integer[] NOT NULL,   -- top three classes and their counts
  top_det text NOT NULL,                      -- the detector's top class ('' when none)
  dup_groups integer NOT NULL,                -- near-duplicate groups of the current run with live members
  PRIMARY KEY (partition_id, cluster_uid)
) WITH (fillfactor = 50);
-- revision 2: one cluster's pending deltas without scanning the kind's whole backlog (round 3, M2)
CREATE INDEX ix_sum_deltas_cluster ON sum_deltas (partition_id, cluster_uid, kind) WHERE kind >= 7;
ALTER TABLE near_dup_runs ADD COLUMN summaries_named_at timestamptz NULL;   -- 6.7
```

`sum_cluster_card` exists because the card grid reads every cluster of the partition (19,402 at 10 M)
under any of eight sorts: summing 273,471 `sum_cluster` rows per request cost 212 ms before any card
field was read (section 14.2). A materialised view is not used: `REFRESH MATERIALIZED VIEW`
"completely replaces the contents", and `CONCURRENTLY` still recomputes the whole view and needs a
unique index ([REFRESH MATERIALIZED VIEW](https://www.postgresql.org/docs/18/sql-refreshmaterializedview.html)),
so each refresh would cost the whole roll-up and the cards would be stale between refreshes; the
compactor instead rewrites only the cards whose base rows it just changed. Its counters change no
indexed column, so its updates stay HOT at fillfactor 50 (as `sum_*` updates were in every round 3
compactor cell, `m3`). `ix_sum_deltas_cluster` serves one cluster's reads (`pl_total_summary`, the
cluster header): with 10,000 pending deltas a per-cluster probe without it scanned the kind's whole
backlog (`m2` `b10000_p7_r8` 3,111 ms p50; with the index `b10000_p7_r8_idxcluster` 369.7 ms), and
it showed no write cost at the noise (`m1` `label1_w30_delta_paced_idxcluster`: service p99 38.1 vs
36.7 ms).

`sum_deltas` is vacuumed by count, threshold = the compactor batch measured by section 13 M3. Only
with `security_invoker = true` does a view apply the invoking user's privileges and row-security
policies ([CREATE VIEW](https://www.postgresql.org/docs/18/sql-createview.html)), keeping 02a in
force. "Live" is not excluded and not holdout (`cluster_cards.py:116`). `detector_class_name` and
`detector_confidence` are fold outputs on `item_state` (15a 2.4), `''` when none. Box partitions
(row 41) have no summary.

### 6.2 The one function that emits deltas

`src/db/repos/state.py` gains **`store_states(session, project, *, folds=None, box_keys=None,
delete_items=(), photos_added=(), capture=(), delete_photos=()) -> StoreOutcome`**, the only
function that writes `item_state`, `box_state`, `photo_capture` and `sum_deltas`, and the only one
that deletes counted rows ahead of a cascade. It replaces `upsert_item_states`, `upsert_box_states`
and `delete_states` (`state.py:115-186`); `photo_capture` leaves `results.RESULT_SPECS`. In the
caller's transaction, in this order:

1. `item_state` upsert (`upsert_distinct`) `RETURNING old.<counted>, new.<counted>`: a row the `IS
   DISTINCT FROM` guard leaves alone is not returned, and old values are `NULL` for an insert
   ([INSERT](https://www.postgresql.org/docs/18/sql-insert.html), [RETURNING old and new](https://www.postgresql.org/docs/18/dml-returning.html)).
   Each pair is aliased (`old.class_id AS o_class_id, new.class_id AS n_class_id`): two output
   columns of one name cannot both be read back by name (round 3, section 14.1 item 4).
2. `box_state` upsert and stale-box `DELETE`, both `RETURNING` the counted columns.
3. `delete_items` (vanished items): `DELETE` of their `box_state` and `item_state` rows `RETURNING`
   the counted columns ([DELETE](https://www.postgresql.org/docs/18/sql-delete.html)).
4. `capture`: `INSERT INTO photo_capture ... ON CONFLICT (photo_key) DO UPDATE ... RETURNING
   old.lat IS NOT NULL, new.lat IS NOT NULL`. The upsert locks the row it replaces, so of two racing
   writers (`source_backfill`, `location_purge`) the second sees the first's committed row as `old`:
   the `located` move counts once without a photo lock (M4).
5. `delete_photos` (erasure, project delete; the caller holds the photos `FOR UPDATE`, rank 5): their
   items as in 3, their `photo_capture` rows `RETURNING`, their `source` for `sum_photos`; the
   caller's catalogue delete then cascades.
6. `photos_added` (from `photos.insert_photos`): `sum_photos` `+1` (`located = false`) and the
   photo's zero `photo_counts` base row (a new key no other transaction can hold).
7. `tally_context`, one read of what the counted columns lack: each touched photo's `source`; from
   S6 the changed items' memberships in the partitions `cluster_activations` names (serving,
   pending) and their groups in the runs of 6.7. It runs after 1 to 6 (6.7 relies on it).
8. `src/domain/tallies.py` (new, pure) `tally_deltas(changes, context)`: per (old, new) pair, minus
   one on each key the old row counts in and plus one on each key of the new row, summed per (kind,
   key), zeros dropped; then **one** `INSERT INTO sum_deltas ... SELECT ... FROM unnest(...)`.

Counted columns of `item_state`: `photo_key`, `class_id`, `class_source`, `label_source`,
`class_validated`, `excluded`, `holdout`, `region_status`, `region_validated`, `region_verifier`,
`updated_at`, `detector_class_name`, `detector_confidence`, and the columns the counted review tabs
read. Review-tab membership is one declarative table in `src/domain/review_tabs.py` (coordinator
answer 4): a tab is **counted** (a key of `sum_review` and `sum_cluster_review`) when every column
its clauses read is a counted column or the membership score; a tab that reads a result table
(`model_disagreements`: probe; `all`: probe entropy and the vector row) is not counted, and its
totals are P4 and its card count P13 (section 11 question 6). A property test asserts the SQL and
the Python reading of the table select the same rows on generated data.

**Callers, every path that changes a counted fact:**

| Path | How it reaches `store_states` |
|---|---|
| every human act, machine writer and ingest group | `apply()` (`src/services/record/apply.py:126-130` today) |
| attach (S1a-ii, `src/services/record/attach.py`): ingest, re-detect, open vocabulary, import reconcile | through `apply(photo_scope=True)` |
| class merge job (15a section 7) | `apply(item_keys=<chunk>)` with no new input: the fold resolves `merged_into` |
| `fold.rebuild` (15a 4.1, M11) | per chunk, `store_states(folds=..., delete_items=...)` |
| photo insert (ingest, combine, import) | `photos.insert_photos` calls it with `photos_added` |
| capture writer, location purge (I6) | `store_states(capture=...)` |
| erasure of a photo or person, project delete (15a section 8) | per chunk, `store_states(delete_photos=...)` before the catalogue delete |

`tests/storage/test_counted_writers.py` (an `rg` ratchet) fails when a module other than
`src/db/repos/state.py` writes `item_state`, `box_state`, `photo_capture` or `sum_deltas`, or a
module other than `src/db/repos/sum_compact.py` writes a base summary table.

### 6.3 Why delta inserts never wait

The only unique key of `sum_deltas` ends in an identity value, which `nextval` hands out distinct
and never reuses ([sequence functions](https://www.postgresql.org/docs/18/functions-sequence.html));
an inserter waits only for a conflicting key of an uncommitted transaction ([index uniqueness checks](https://www.postgresql.org/docs/18/index-unique-checks.html)),
which a fresh identity never is. No writer locks a summary row, so none holds one through its vector
insert or commit (B3). Concurrent inserts meet only on the rightmost leaf page of a (project, kind)
for a buffer lock's instant (section 13 item M1). Revision 0's tally advisory lock and lock rank 8
are gone (15a section 7).

### 6.4 The reader: base plus deltas, one statement

P3 reads the `_now` views; P7 reads `sum_cluster_card` and the pending cluster deltas in one
statement (P7). A statement sees one snapshot ("as of the instant the query begins
to run", [transaction isolation](https://www.postgresql.org/docs/18/transaction-iso.html)) and the
compactor moves a batch from deltas to base rows in one transaction, so each delta counts exactly
once at any snapshot. Cost grows with the pending deltas of the (project, kind), read through the
key prefix `(project_id, kind)`. Round 3 (M2): P3 met its bar with 10,000 pending deltas per kind
(class total 4.1 / 7.8 ms, class plus flags 4.3 / 8.0 ms, `m2` `b10000_p3a_r8`, `b10000_p3b_r8`);
the compactor of 6.5 keeps the backlog at 1,641 to 3,536 pending rows under 30 writers (`m3`,
batch 1,000 to 50,000 at 0.2 s), well under that.

Sorts on a summary value read the base rows only: `unlabeled_count` (X24) orders by the compacted
`photo_counts.unlabeled`, so a photo's place in that sort lags its card numbers by at most one
compaction (today's counts lag by the Valkey drain, `image_counts_dirty.py`). The card numbers
themselves are exact (P6 reads `photo_counts_now`).

### 6.5 The compactor

`src/db/repos/sum_compact.py` (new, S1a-i), one job kind `sum_compact` per project under the job
lease (`src/clients/job_lease.py`), started by a beat entry and looping until a batch moves fewer
rows than `:batch`:

```sql
WITH batch AS (
  SELECT project_id, kind, delta_id FROM sum_deltas
  WHERE project_id = :p LIMIT :batch FOR UPDATE SKIP LOCKED
), moved AS (
  DELETE FROM sum_deltas d USING batch b
  WHERE (d.project_id, d.kind, d.delta_id) = (b.project_id, b.kind, b.delta_id)
  RETURNING d.*
), k1 AS (
  INSERT INTO sum_items AS s (project_id, source, class_id, class_source, label_source,
                              class_validated, excluded, holdout, n)
  SELECT project_id, source, class_id, class_source, label_source, class_validated, excluded, holdout, sum(n)
  FROM moved WHERE kind = 1 GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
  ORDER BY 1, 2, 3, 4, 5, 6, 7, 8
  ON CONFLICT (project_id, source, class_id, class_source, label_source, class_validated, excluded, holdout)
  DO UPDATE SET n = s.n + EXCLUDED.n
)  -- k2 .. k10: the same for each base table (cluster kinds: sums of m[], conf_sum, greatest(max_updated))
SELECT count(*) FROM moved;
```

The conflict target of each `kN` is the base table's primary key: `(project_id, <keys>)` for kinds 1
to 5, the key alone for kinds 6 to 10 (`photo_counts (photo_key)` and the four cluster tables,
whose keys hold no `project_id`, 6.1); round 3 failed with "no unique or exclusion constraint
matching the ON CONFLICT specification" until it used them (section 14.1 item 1). Every upsert orders
its rows by the group keys, so two compactors that meet (a lapsed lease) lock base rows in one order
and rarely deadlock; a deadlock is still retried whole.

**Card refresh** (revision 2), a second statement in the same transaction, only when the batch
moved cluster kinds. It is a separate statement because sub-statements of one `WITH` "cannot see
one another's effects on the target tables" (queries-with page below), while a later statement of
the same transaction sees them:

```sql
-- statement 2a: lock the batch's existing cards in key order
SELECT 1 FROM sum_cluster_card c
WHERE (c.partition_id, c.cluster_uid) IN (SELECT * FROM unnest(:touched_partitions, :touched_clusters))
ORDER BY c.partition_id, c.cluster_uid FOR UPDATE;
-- statement 2b: roll the touched clusters' base rows up into their cards
WITH touched AS (
  SELECT DISTINCT partition_id, cluster_uid FROM unnest(:touched_partitions, :touched_clusters) AS t(partition_id, cluster_uid)
)
INSERT INTO sum_cluster_card AS c (partition_id, project_id, cluster_uid, n_all, n_live, labelled_live, validated_live,
       max_updated_live, conf_sum_live, conf_n_live, needs_review, review_n, top_class_ids, top_class_n, top_det, dup_groups)
SELECT t.partition_id, :p, t.cluster_uid, <the roll-up of P7 revision 1 for one cluster: sums over
       sum_cluster, the top three of sum_cluster_class, the top detector class, the counted tabs of
       sum_cluster_review in review-tab order, the groups of sum_cluster_dup of the current run>
FROM touched t
ORDER BY 1, 3
ON CONFLICT (partition_id, cluster_uid) DO UPDATE SET n_all = EXCLUDED.n_all, n_live = EXCLUDED.n_live  -- ... every column
```

`:touched_*` are the `(partition_id, cluster_uid)` pairs of the moved rows of kinds 7 to 10, which
the first statement returns beside its count. The roll-up reads base rows only (the moved deltas
are in them now), so the card equals the roll-up of its base rows at commit. Statement 2a locks the
existing cards before 2b takes its snapshot (in read committed "each command" sees the data
committed before it began, transaction isolation page): a second compactor waits in 2a for the
first to commit, and its 2b then rolls up a snapshot that includes the first. A card two lapsed
compactors create at once is the one case 2a cannot lock (no row yet); the drift check corrects it. The drift check (6.6)
also compares every card with the roll-up of its base rows and rewrites a differing card. Cost: a
batch that labels inside one cluster touches one card; the compactor's rate with the refresh is
timed in round 4 (R4-2) against twice M1's peak delta rate.

**Batch and beat (round 3, M3).** `:batch` = 10,000, beat interval 0.2 s, sleeping only when a batch
moved fewer rows than `:batch`: under 30 paced writers it moved every delta produced (160,797 of
161,425 in 60 s), its transaction p50 / p99 was 43.0 / 67.2 ms, at most 2,373 deltas were pending
at the end, and the writers' response p99 was within 0.4 % of the cell without a compactor (`m3`
`b10000 i0.2` against `control`). Batch 1,000 raised writer p99 by 11 %, a 1 s interval by 67 %, a
5 s interval left 17,266 pending; batch 50,000 gave a transaction p99 of 77.2 ms and 3,536 pending.
Compaction alone moved 74,032 rows per second at batch 10,000 on a seeded backlog (5.6 times M1's
peak delta rate of 13,191 per second). `sum_deltas`' vacuum threshold is therefore 10,000 (6.1).

Sub-statements of one `WITH` share one snapshot and talk only through `RETURNING`; `DELETE ...
RETURNING` feeding an `INSERT` is the documented way to move rows ([WITH, data-modifying statements](https://www.postgresql.org/docs/18/queries-with.html#QUERIES-WITH-MODIFYING)).
The `GROUP BY` affects each base row once (the INSERT page's cardinality rule). `SKIP LOCKED` gives a second compactor (a lapsed lease) another batch, as for "a
queue-like table" ([SELECT, locking clause](https://www.postgresql.org/docs/18/sql-select.html#SQL-FOR-UPDATE-SHARE));
even without it a `DELETE` ignores a row a committed transaction deleted (transaction isolation
page), so a delta moves once; two upserts of one base row apply on the committed value, and a
deadlock between compactors is retried whole. **A crash mid-batch** rolls the one transaction back:
deltas stay, base rows and cards are unchanged, readers stay exact.

### 6.6 Rebuild without a lock (R7, the drift check)

`rebuild_tallies(session, project, *, kinds)` (`sum_compact.py`) computes, at one snapshot, the true
count of every key from the state tables and the recorded count from the `_now` views, and inserts
the difference as correcting deltas:

```sql
BEGIN ISOLATION LEVEL REPEATABLE READ;
INSERT INTO sum_deltas (project_id, kind, source, class_id, class_source, label_source,
                        class_validated, excluded, holdout, n)
SELECT :p, 1, k.source, k.class_id, k.class_source, k.label_source, k.class_validated, k.excluded,
       k.holdout, k.truth - k.recorded FROM (
  SELECT source, class_id, class_source, label_source, class_validated, excluded, holdout,
         sum(t) AS truth, sum(r) AS recorded
  FROM (SELECT p.source, coalesce(s.class_id, -1) AS class_id, coalesce(s.class_source, '') AS class_source,
               coalesce(s.label_source, '') AS label_source, s.class_validated, s.excluded, s.holdout,
               1 AS t, 0 AS r
        FROM item_state s JOIN photos p USING (photo_key) WHERE s.project_id = :p
        UNION ALL
        SELECT source, class_id, class_source, label_source, class_validated, excluded, holdout, 0, n
        FROM sum_items_now WHERE project_id = :p) x
  GROUP BY 1, 2, 3, 4, 5, 6, 7) k
WHERE k.truth <> k.recorded;
-- one statement per kind and every measure of the kind (n, m[], conf_sum), the same snapshot
COMMIT;
```

The statement above is kind 1; the other nine have the same shape over their state tables and every
measure of the kind (`n`, each `m[]` position, `conf_sum`), which round 3 wrote and ran for all ten
(section 14.1 item 2). The truth of a float measure is summed in `double precision`
(`sum(s.detector_confidence::float8)`): a `real` sum drifted from the writer's double-precision
tally on 5,833 keys at 1 M in round 3's first rebuild (item 3). `max_updated_live` is not
corrected (it only grows). Round 3 (M4): all ten kinds took 138.8 s idle and 153.7 s under 30
writers at 10 M, writer p99 +4.6 % (service) and +3.3 % (response) against the paced M1 cell,
oldest transaction 153 s, and 0 mismatching keys afterwards in all ten views (`m4`, `m4_check`):
the nightly drift check fits a quiet hour, and the partition fill of 6.7 costs about kinds 7 to 10
(12.4 + 11.9 + 35.0 + 7.6 s idle).

Every statement of a repeatable read transaction sees the first statement's snapshot, and a plain
`INSERT` of new rows is not among the commands that fail with a serialization error (transaction
isolation page). Exact, because a writer's state change and its delta commit together (the snapshot
holds both or neither), compaction moves counts atomically, and every later change brings its own
delta. Integer counts are exact; `conf_sum` is corrected too, so `avg_conf` drifts only between
rebuilds; `max_updated_live` only grows between rebuilds (a leaving member keeps the newest time).
No lock is taken; the snapshot holds back vacuum for its duration (section 13 item M4). The nightly
drift check is this rebuild: it logs the correcting rows and alerts when there are any.

### 6.7 Partition- and run-keyed summaries, filled before the pointer flips

Cluster activation and a near-duplicate flip do not rebuild. The clustering job, after writing the
partition's memberships:

1. sets `cluster_activations.pending_partition_id` (02d OCC, one short transaction); from its commit
   on, every `store_states` reads the partition in `tally_context` and emits its cluster kinds;
2. takes `pg_current_snapshot()` and waits, bounded by its lease, until no transaction id in
   `pg_snapshot_xip` of it is `in progress` (`pg_xact_status`, [system information functions](https://www.postgresql.org/docs/18/functions-info.html)).
   A writer that read the activation row before step 1 had already written its state rows
   (statement 7 runs after 1 to 6), so it holds a transaction id and is in that list; once the list
   has drained, no such writer is left;
3. runs the 6.6 statements for the partition's kinds (base rows are empty, so the correction is the
   whole count minus the deltas written since step 1), compacts them, and writes every card of the
   partition by the roll-up of 6.5 statement 2b over all its clusters (one statement);
4. leaves the serving flip to activation, which is then a pointer change (15a section 7).

A near-duplicate run does the same with `near_dup_runs.summaries_named_at` set in step 1 (still
`building`); the flip to `current` changes no summary. A retained partition keeps its rows; a pruned
one loses rows and deltas by cascade. On a wait timeout the job leaves the partition pending.

### 6.8 Tests

The property test (`tests/db/test_tallies_property.py`, S1a-i): random acts from two sessions
through `apply()`, attach, `fold.rebuild`, erasure, project delete, capture writes and class merge,
with the compactor and `rebuild_tallies` at random points; after each step every `_now` view equals
the `GROUP BY` over the state tables (integer columns exactly). Unit, compactor and partition-fill
tests: 15c S1a-i and S6.

### 6.9 Cost per label write

`store_states` replaces revision 0's two `tally_inputs` reads and per-table upserts with
`RETURNING` on the writes it already makes, one `tally_context` read and one `INSERT`. A single
label is about **22 round trips** (round 3's prototype, `notes.md` section 3, section 14.1 item 7):
scope setting, operation row, 2 registry reads, the items lock, the claims insert, 7 reads and writes
for `load_item_inputs` and results, the `item_state` update, `tally_context` as 3 reads (photo
source, serving membership, near-duplicate group), one delta insert, 2 event rows, the operation
update. It emits 5.95 delta rows (`m1` `delta_rows_per_label_item`); a 1,000-item chunk about 2,750;
a 50-photo ingest group 54. Measured: 1 writer 22.1 / 29.2 ms service p50 / p99, 30 paced writers
(450 labels per second) 20.1 / 36.7 ms service and 89.3 ms response p99 (`m1`). The Rust slice
sends these as one pipelined batch where the driver allows (18 section 2.1, tokio-postgres
pipelining), which lowers the round-trip count but not the statements.

### 6.10 Summary candidates not built

`sum_embedding (project, proposal_name, embedding state)` for row 14 is built only if row 14 misses
the dashboard bar at 10 M; it would need the vector writers to emit through `store_states`.

## 7. Sorts

**Placement** (coordinator decision on M3): a sort value that a label never changes lives on the
identity or detection table and is indexed there, where a label writes nothing: creation time on
`items` (X28), detector confidence, blur and detector class on `detections` (X8, X31, X32), capture
time on `photo_capture` (X13). Crop area and rank change only with geometry, an exclusion or attach,
never with a label: `items.sort_area` and `items.sort_rank` (15a 2.3) are copies `store_states`
writes only when the fold's `area` or `rank` differs (X29, X30). Scores sort on their score tables.
Mutable keys on `item_state` stay within its five B-trees (X1 to X5).

| Sort id (`crop_browse.py:17`, `review_sorts.py`, `image_view.py:66-80`) | Keys and directions, tiebreak last | Index |
|---|---|---|
| `updated_at`, review `recent` | `item_state.updated_at` desc or asc, `crop_id` | X1; with up to 16 classes, X2 branches (P1) |
| `created_at` | `items.created_at`, `crop_id` | X28 |
| `crop_area_norm`, `crop_rank_in_image` | `items.sort_area`, `items.sort_rank`, `crop_id` | X29, X30 |
| `confidence` (`check_machine` default `confidence:asc`) | `item_state.class_confidence`, `crop_id` | X4 on lists inside the review predicate; else sort path only |
| `validated`, `class_name` (registry name order: `array_position(:ids_by_name, class_id)`) | `item_state`, `crop_id` | sort path only |
| `detector_confidence` (`to_label` default, `review_sorts.py:342`) | `detections.confidence` desc nulls last, `item_key` | X8 |
| `blur_lap_ratio`, `detector_class_name` | `detections.blur_lap_ratio`, `detections.class_name`, `item_key` | X31, X32 |
| `cluster_score`; review `representativeness` (desc), `atypicality` (asc) | `cluster_memberships.score`, `subject_key` | X23 (partition), X22 (one cluster) |
| `mistakenness_score`, `uniqueness_score` (desc) | `item_scores.score`, `item_key` | X20 |
| review `uncertainty_entropy` (desc), `disagreement_entropy_asc` (asc) | `probe_predictions.entropy`, `item_key` | X21 |
| `region_score` (`review_sorts.py:80`, desc, missing last) | `item_state.region_max_score`, `crop_id` | sort path over the regions queue |
| audit `crop_id:asc` | `item_state.crop_id` | sort path over the audit sample |
| `detector_disagreement_default` (mixed) | VLM confidence rank asc (`CASE s.vlm_confidence WHEN 'low' THEN 1 WHEN 'medium' THEN 2 WHEN 'high' THEN 3 ELSE 0 END`, today's `VLM_CONFIDENCE_RANK_SORT`, `review_sorts.py:41-54`), then `class_confidence` desc missing last, `crop_id` | sort path over the disagreements queue (expanded keyset form, section 4) |
| images `indexed_at`, `taken_at`, `file_name`, `unlabeled_count` | `photos.photo_key`, `photo_capture.taken_local` (today's field), `photos.storage_key`, `photo_counts.unlabeled`, `photo_key` | X9, X13, X12, X24 |

**Two paths** (coordinator answer 1). At most `T_sort` matching rows: read them through the most
selective index and sort them (top-N, `LIMIT` rows held), the **sort path**. More: walk the sort's
index in order and apply the filter, the **walk path**; it reads about (rows served) / (fraction
matching) entries, so a filter just over `T_sort` on 10 M rows is its worst case.

Revision 2, from round 3 (section 14.2 "sort path" and "walk path"):

- **The sort path reads its key from `item_state` where a copy is there**: `created_at` (a copy of
  `items.created_at`), `area` and `rank` (the fold's geometry, equal to `items.sort_area` and
  `sort_rank`), `detector_confidence` and `detector_class_name` (fold outputs, 15a 2.4), besides the
  keys that live there (`updated_at`, `class_confidence`, `class_validated`, `class_id`,
  `region_max_score`, `crop_id`, `vlm_confidence`). Only `blur_lap_ratio`, `cluster_score`, the score
  tables and the photo sorts join. Round 3 joined every placed key and paid 4.2 to 5.6 µs per matched
  row (sort path at 100,000 rows: 424 to 564 ms p50 for the joined sorts against 76 to 161 ms for
  the `item_state` sorts, `m7`). The walk path still walks the indexes of the table above (X28 to
  X32, X8).
- **What decides the path.** When the filter is summary keys the exact total (P3) decides. Else the
  sort path's read is itself the guard: it materialises at most `T_sort + 1` matching rows (`LIMIT`
  inside the read) and switches to the walk when it got `T_sort + 1`; no second count runs. The
  first page's choice rides in the cursor (section 4).
- **`T_sort`** lies between 10,000 (every sort met: worst `created_at` 49 / 82 ms with its join)
  and 100,000 (none met); round 4 (R4-5) fixes it from the sort path at 20,000, 40,000 and 70,000
  rows on the `item_state` copies with a real class filter. Expected from the round 3 rates (about
  1 µs per row for `item_state` keys, `updated_at` 17 ms at 10,000 and 105 ms at 100,000): 40,000
  to 50,000.
- **The band.** A walk over `item_state`'s own index (X1) costs about 1.7 µs per entry read (103 ms
  for 61,000 entries at 0.1 %), so for `updated_at` the two paths cross near 33,000 matching rows at
  about 33 ms: no band. A walk over another table's index probes `item_state` per entry, about
  8.7 µs (`created_at` 530 ms for 61,000 entries), so for the joined walks the paths cross near
  75,000 matches at about 75 ms p50: matches between `T_sort` and about 100,000 miss the 50 ms p50 bar
  on both paths for `created_at`, `crop_area_norm`, `crop_rank_in_image`, `detector_confidence`,
  `blur_lap_ratio`, `detector_class_name` and `cluster_score` (estimate from cells; R4-5 measures it).
  Sorts with no walk index (`validated`, `class_name`, `region_score`, audit `crop_id`,
  `detector_disagreement_default`) are bounded by the sort path alone: above `T_sort` matches they
  miss (100,000 rows: 88 to 161 ms p50, `m7`). Per coordinator answer 1 such sorts go to the owner
  (section 14.3 question 2), never refused with an error.

**Script filters** (M7, shape 2): `DETECTOR_DISAGREES_SCRIPT` (`review_tab_clauses.py:45-49`)
becomes `s.detector_class_name <> '' AND s.class_id IS NOT NULL AND s.detector_class_name <>
<registry name of s.class_id>` (both names in registry form, the fold output of 15a 2.4);
`MODEL_DISAGREES_SCRIPT` (`:53-57`) becomes `EXISTS (SELECT 1 FROM probe_predictions p WHERE
p.item_key = s.item_key AND p.<predicted class> <> <registry name of s.class_id>)`. Both are
declared once in `src/domain/review_tabs.py`.

Planner statistics for the correlated class columns (`st_item_state_class`, statistics target
1,000 on `class_id`): 15a 2.6 ([extended statistics](https://www.postgresql.org/docs/18/planner-stats.html#PLANNER-STATS-EXTENDED)).

## 8. Text search: a B-tree, the trigram index and row security

**Box text** (today the `text_fold` keyword and its `text_fold.sub` wildcard field,
`region_text_search.py:1-11`; the review tab's case-insensitive wildcard). `box_state.text_fold` is
already folded by `src/domain/region_text.py` (normalised, upper case, confusables mapped, one word),
so case needs no `ILIKE`. Two indexes: X7, `CREATE INDEX ix_box_state_text_trgm ON box_state USING
gin (text_fold gin_trgm_ops) WITH (fastupdate = on, gin_pending_list_limit = 512) WHERE text_fold IS
NOT NULL`, for fragments; and X33 (revision 2), `CREATE INDEX ix_box_state_text ON box_state
(project_id, text_fold) WHERE text_fold IS NOT NULL`, for every equality form. Migration `0011`
runs `CREATE EXTENSION IF NOT EXISTS pg_trgm` before X7 (round 3 found the database without it,
section 14.1 item 8).

**Row security decides which index a text match can use** (round 3, section 14.2 "text"). The
record's tables have forced row security (02a 1.4). "The system will enforce conditions from
security policies ... before any user-supplied conditions from the query itself that contain
non-leakproof functions", and only "functions and operators marked as leakproof ... may be executed
before conditions from security policies"; only a superuser can mark one ([CREATE FUNCTION,
`LEAKPROOF`](https://www.postgresql.org/docs/18/sql-createfunction.html), [CREATE POLICY,
notes](https://www.postgresql.org/docs/18/sql-createpolicy.html)). An index condition is evaluated
before the policy's filter, so a non-leakproof operator such as `LIKE` or pg_trgm's `%` stays a
filter after a sequential scan: round 3's plan of the fragment count (`m6a` `p4_text_gin_r1`) is a
sequential scan of `box_state` with the policy and `text_fold ~~ '%ZTNP%'` in one filter, 5,099,275
rows removed for 12, 1,661 ms, while the equality form (text `=` is leakproof) used an index and met
its bar (3.7 / 6.3 ms, `m9` `exact_first_r8`). Round 4 (R4-3) confirms the mechanism from
`pg_proc.proleakproof` and the same plan as a role that bypasses row security.

| Match (`TextMatch`) | SQL | Index |
|---|---|---|
| `exact` | `b.text_fold = :key` | X33 |
| `fuzzy` (`FUZZY_FUZZINESS = 1`, transpositions, `region_text_search.py:29-30`), any length | `b.text_fold = ANY(:variants)`: every string at Damerau-Levenshtein distance at most 1 from the key over the folded alphabet of `region_text.py`, built by one domain function | X33 (one index probe per variant) |
| `fragment` (3 characters or more) | `b.item_key IN (SELECT op_text_fragment(:frag_escaped))` (`%`, `_`, `\` escaped by the caller) | X7, inside the function below |
| `same_text_as` (`same_text_key`, `region_text_search.py:37`) | the box's own `text_fold` by `(item_key, box_id)` (`uq_box_state_item_box`), then `exact` | X33 |

**Fuzzy is the variant list for every length** (revision 2). A key of `n` characters over an
alphabet of `A` folded characters has at most `n(A - 1)` substitutions, `(n + 1)A` insertions, `n`
deletions and `n - 1` transpositions: 546 strings for a 7-character plate over 36 characters, each
one B-tree probe of X33. The variants are exact matches, so no distance check follows and no
plate at distance 1 is missed by construction. pg_trgm itself notes that its indexes "may not be as
efficient as regular B-tree indexes for equality operator" ([pg_trgm](https://www.postgresql.org/docs/18/pgtrgm.html)),
and round 3's variant lists on X7 took 64 to 170 ms p50 (`m9` `var1` to `var3`); the trigram
similarity form (`%` with a derived threshold, revision 1) is not built, since `%` cannot use X7
under row security (fuzzy 4.7 to 5.3 s, `m9` `fuzzy4` to `fuzzy9`). A unit test checks the variant
generator against a brute-force distance on generated plates; R4-3 times it for 1 to 10 characters.

**Fragments go through one project-bound search function** (revision 2). The only operators X7
serves are non-leakproof, so the fragment read runs where row security is not applied, inside a
function that applies the project itself and fails closed:

```sql
-- 0011, run by the migrator (a superuser: only a superuser can create a role with BYPASSRLS)
CREATE ROLE op_text_search NOLOGIN BYPASSRLS;
GRANT SELECT (project_id, item_key, text_fold) ON box_state TO op_text_search;
GRANT SELECT (project_id, item_key, text) ON text_lines TO op_text_search;
BEGIN;
CREATE FUNCTION op_text_fragment(frag text) RETURNS SETOF bigint
  LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public, pg_temp ROWS 1000 AS $f$
  SELECT DISTINCT b.item_key FROM box_state b
  WHERE b.project_id = op_bound_project_strict()      -- the setting the policies read; raises when unbound
    AND b.text_fold IS NOT NULL AND b.text_fold LIKE '%' || frag || '%'
$f$;
ALTER FUNCTION op_text_fragment(text) OWNER TO op_text_search;
REVOKE ALL ON FUNCTION op_text_fragment(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION op_text_fragment(text) TO op_app;
COMMIT;
-- op_text_word(w text): the same for text_lines, lower(l.text) ~ ('\m' || w), index X25
```

This follows the documented pattern for definer functions: a fixed `search_path` ending in
`pg_temp`, execute revoked from `PUBLIC` and granted in the same transaction ([CREATE FUNCTION,
writing `SECURITY DEFINER` functions safely](https://www.postgresql.org/docs/18/sql-createfunction.html#SQL-CREATEFUNCTION-SECURITY));
`BYPASSRLS` is the role attribute that skips policies ([CREATE ROLE](https://www.postgresql.org/docs/18/sql-createrole.html)).
`op_bound_project_strict()` reads the same transaction setting as the policies (02a 1.4) and raises
when no project is bound or the scope is not a project, so the function cannot answer for another
project or for none. The owner role has column-level `SELECT` on three columns and nothing else.
Tests (`db` feature of the slice that builds it): bound to project A it never returns an item of
B; unbound it raises; `op_app` cannot alter it; its result equals the `LIKE` over the same rows run
by the owner. Options considered: marking `textlike` and pg_trgm's operators `LEAKPROOF` (needs a
superuser change to system functions that a dump does not carry, and `LIKE` raises on some
patterns, which the definition of leakproof excludes); an n-gram table on B-trees (about 15 rows
per plate, hand-made trigram search); both rejected. Write cost of the chosen forms: X33 adds one
B-tree entry per box with text (a text change is already not HOT, X7 holds `text_fold`); the
function costs nothing on writes.

Items are `SELECT DISTINCT b.item_key ...` (or the function's rows), used as an `EXISTS` or `IN`
predicate inside P1, P4, P8 and P9. "A pattern with no extractable trigrams will degenerate to a
full-index scan" (pg_trgm page): a fragment shorter than three characters is refused with 400
`text_too_short` as a served sentence (15c contract addendum). The GIN index carries no project
column; the project test is a recheck inside the function.

**GIN pending list** (round 3, M9): `fastupdate` on with `gin_pending_list_limit` 512 kB gave the
lowest insert p99 under 30 box writers (service 8.2 / 17.3 ms p50 / p99, against 14.3 / 40.6 ms with
`fastupdate` off, H; 21.1 at 4 MB, H; 28.3 at 16 MB, H; `m9` `gin_insert_*`). A pending-list flush
runs inside an ordinary insert and readers scan the pending list too ([GIN fast
update](https://www.postgresql.org/docs/18/gin.html#GIN-FAST-UPDATE)); the reader side of that
setting was not measured, so R4-3 times fragment reads while the box writers run.

**Item text lines** (today `item_text_tokens` with a `prefix` per word, `src/domain/item_text.py:89`,
`:110`): `text_lines.text` of an item, index X25 `gin (lower(text) gin_trgm_ops) WITH (fastupdate =
on, gin_pending_list_limit = 512) WHERE item_key IS NOT NULL`; a word `w` matches through
`op_text_word(w)` (word-start regular expression `~`, served by the same index, non-leakproof like
`LIKE`). `item_text_tokens` goes (15a 5.2).

## 9. Box rows: the index and the page

Today the page sorts items by their newest box `detected_at` (`routers/curation/regions.py:137-149`)
and `inner_hits` return only the boxes that matched, text included. The spike's semi-join walk:
194.7 / 593 deep, exact total 600 / 2496 (`s8_*`).

Design: items **newest first**, one ordered index per box filter shape, since only equality on every
column before the sort column bounds an ordered range (multicolumn indexes page): X6 `(project_id,
state, detector, item_key DESC)`, X26 `(project_id, state, item_key DESC)`, X27 `(project_id,
detector, item_key DESC)` (M6; box rows change far less often than item rows). `<box predicate>` is
the same text in both statements: state, detector and the text match of section 8.

```sql
-- the page's items: distinct items with a matching box, newest first
SELECT DISTINCT b.item_key FROM box_state b
WHERE b.project_id = :p AND <box predicate>
  AND b.item_key < :after_item_key
  AND EXISTS (SELECT 1 FROM item_state s WHERE s.item_key = b.item_key AND <item filter>)
ORDER BY b.item_key DESC LIMIT :limit_plus_one;
-- their matching boxes (rows): only the boxes that match, as inner_hits did
SELECT b.* FROM box_state b
WHERE b.project_id = :p AND b.item_key = ANY(:keys) AND <box predicate>
ORDER BY b.item_key DESC, b.ordinal;
```

The index delivers `item_key` in order inside the equality prefix, so `DISTINCT` is a unique step
over sorted input that stops after the page's items (an item holds at most 74 boxes). A text-only
filter takes the GIN candidates (section 8) and orders their items (a bitmap read, sorted); a filter
with neither state, detector nor text walks X1-ordered items with an `EXISTS` on
`uq_box_state_item_box`. Inside one box cluster the order is the box membership score (X22), as
today's cluster branch (`regions.py:120-135`). `total_rows` is exact from `sum_box` for a state or
detector filter (alone or together), else P4; `total` (items) is P4 over the `DISTINCT` form.
`rows_truncated` and the three-inner-hit window go (contract diff). The order changes from newest
detected box to newest item (15c contract addendum). Round 3 (M9, 8 readers, first page): state and
detector 4.1 / 9.1 ms, state 6.2 / 18.9 ms, detector 5.7 / 9.0 ms (`m9`, H, so upper bounds),
against the spike's 194.7 / 593: met.

## 10. Job readers: streaming large sets

| Reader | Form | Why |
|---|---|---|
| jobs that need one consistent pool: clustering, near-duplicates, region false-positive rows, outlier and diverse orders, re-index sampling (rows 8, 59, 61, 64) | P11, one `COPY (SELECT ... ) TO STDOUT (FORMAT BINARY)`, read with psycopg's `cursor.copy()` and the pgvector adapter (15a 6.4) | one statement is one snapshot, so the pool is consistent while writers run ([COPY](https://www.postgresql.org/docs/18/sql-copy.html)); binary avoids parsing 1,024 numbers per row; 1 M vectors in about 14 s (plan 15 section 1). The keyset-batch form re-reads the filter per batch and is not consistent: no job that needs one pool uses it |
| exports (row 58), combine (67), import reconcile (68) | a **manifest**: one short `REPEATABLE READ READ ONLY` transaction reads `(photo_key, item_key, class, bbox, split)` in key order through a server-side cursor into a file under the job's store key; the slow per-file work then reads the manifest with no transaction open | the export sees one state of the labels; a transaction held open for the whole file copy would hold the xmin horizon for minutes and stop vacuum from pruning dead `item_state` versions ([DECLARE](https://www.postgresql.org/docs/18/sql-declare.html), [SET TRANSACTION](https://www.postgresql.org/docs/18/sql-set-transaction.html), [routine vacuuming](https://www.postgresql.org/docs/18/routine-vacuuming.html)); 1 M rows read in seconds (section 13 is not needed: low risk) |
| long walks that write as they go: reprocess targets, requeue, reclassify, scores, auto-label, source backfill, class merge (rows 50 to 57, 66) | keyset batches `WHERE <scope> AND key > :last ORDER BY key LIMIT 5000`, one short transaction per batch | these jobs tolerate rows changing between batches; their totals are exact counts inside the job (section 5 rule 7) |
| several readers of one snapshot (a parallel export) | `pg_export_snapshot()` in the leader, `SET TRANSACTION SNAPSHOT` in each reader ([functions-admin, snapshot synchronization](https://www.postgresql.org/docs/18/functions-admin.html#FUNCTIONS-SNAPSHOT-SYNCHRONIZATION)) | only if one manifest reader is too slow; not built now |

P11 must walk the vector key and probe `item_state`, not sort the joined set (up to 20 GB at 10 M):
the S3 (DB) test asserts it with `EXPLAIN` at 1 M. The job holds what it streams (10 M x 1,024 x 2
bytes = 20 GB) and bounds its pool by today's settings. A crashed `COPY` has no resume: the job
re-reads (minutes at 10 M) and checkpoints its output, not its read.

## 11. Questions for the coordinator

Revision 0's questions 1 to 5 are answered in 15c ("Coordinator answers") and applied. New:

6. **Card counts of tabs that read result tables.** Answered by round 3: P13 per visible card took
   about 30 ms per card (`m10` `p13_all_per_card_r8`), and the card grid needs the count for every
   card it sorts and filters, not only the visible ones. It is now a product question for the
   owner (section 14.3 question 1), because the `all` count is the card's "N to review" chip.
7. **Area and rank are not immutable** (decision on M3): a geometry edit changes both today
   (`item_geometry.py:141`, `:215-217`). Placed on `items` (X29, X30) and written only when they
   change; a label still writes nothing there. Confirm.
8. **What moves "Recently updated"** (m6): 15a 2.4 now moves `updated_at` only with facts a person
   sees on the item (class, validation, exclusion, holdout, geometry, review dismissal, notes of a
   person), so region-worker and VLM-only writes stay HOT and do not reorder the default list. The
   owner's reading of "recently updated" decides; until then the plan carries this rule.

## 12. Trace: every shape in exactly one slice

| Slice | Shapes |
|---|---|
| S1a-i | none (summary machinery) |
| S1a-ii | 31, 34, 54, 55, 78, 80 |
| S1b-i | 1, 3, 4, 5, 7, 9, 32, 33 |
| S1b-ii | 24, 25, 26, 27, 28, 29, 30 |
| S2a | 66 |
| S2b | 2, 15, 16, 19, 36, 38, 43, 46, 47 |
| S3 | 14, 20, 21, 22, 23, 35, 37, 59, 64, 65 |
| F | none |
| S4 | 6, 40, 42, 49 |
| S5 | 45, 61, 63 |
| S6 | 8, 10, 11, 12, 13, 41, 60, 79 |
| S7 | 39, 58, 67, 68 |
| S8a | 17, 18, 44, 48, 50, 77 |
| S8b | 56, 57, 76 |
| S8c | 51, 52, 53 |
| S9 | 62, 69, 70, 71, 72, 73, 74, 75 |

6 + 8 + 7 + 1 + 9 + 10 + 4 + 3 + 8 + 4 + 6 + 3 + 3 + 8 = 80, each number once.

## 13. Measurement round 3

What every slice's entry condition waits for (15c). A measurement agent runs it on the gate A spike:
the directory `v2_read_shapes` under `artifacts_local/speed_1113/` of the main checkout (gitignored),
whose `notes.md`, `lib.py`, `load.py` and `summary_pg.sql` describe the setup. Engine: container
`op-spike-v2-pg` (`postgres:18.6-trixie`, compose project `op-spike-v2`, loopback port 4770,
8 CPUs), database `v2_10m` (10 M `objects`, 5.1 M `boxes`, 1.37 M `images`; `v2_1m` for the 1 M
runs). It has no `vector`, which no item here needs, and did not have `pg_trgm`: round 3 ran
`CREATE EXTENSION pg_trgm` in `v2_10m` (section 14.1 item 8). Round 3 ran 2026-11-15 08:59 to
2026-11-16 01:29 in `artifacts_local/speed_1113/m3_storage/` (`results_round3.md`, `notes.md`,
`raw/m3_<item>_10m.json`); the table below is the round as run, with the stand-ins section 14.1
lists, and section 14 its outcome.

**Build**, `m3_build.sql` beside `load.py`, once per database, into a new schema `m3` (gate A's
tables stay untouched): `m3.items (item_key, crop_id, photo_key, created_at, sort_area, sort_rank)`
and `m3.item_state` (record names; the columns of P1, P4, X1 to X5 and the counted columns of 6.2)
from `objects` (`item_key` = `row_number() OVER (ORDER BY crop_id)`, `photo_key` from `images` the
same way, `class_excluded` as `excluded`, `test_holdout` as `holdout`, `confidence` as
`class_confidence`); `m3.detections (item_key, analyser_id = 1, confidence, blur_lap_ratio =
random(), class_name)`; `m3.photos (photo_key, source)`; `m3.box_state` from `boxes` (`text_fold =
upper(text)`); `m3.cluster_memberships (partition_id = 1, subject_key, cluster_uid, score)`; the
indexes of 15a 2.6 revision 1 (X1 to X8, X22 to X32, with predicates); the base tables of 6.1 by
`GROUP BY` (as `summary_pg.sql`), an empty `m3.sum_deltas` and the `_now` views; then `VACUUM
(FREEZE, ANALYZE)`. **Runner**, `m3_run.py`: reuses `lib.pg`, `lib.pct`, `lib.cgroup_cpu_s`,
`lib.load1`, `lib.wait_load` and `run.py`'s fork per client; writes `raw/m3_<item>_<size>.json`
(p50, p95, p99, max, every latency, engine CPU-seconds, host load at start and end,
`pg_stat_activity` wait events every 100 ms). Each item runs under the chain's exclusive hold
(`hold.sh`), host load below 16 at start.

| Item | Before | Statements | Load pattern | Record | Bar |
|---|---|---|---|---|---|
| M1 delta upkeep | S1a-i, S2a | a label transaction: `SELECT ... FOR UPDATE` of the items, an `UPDATE m3.item_state SET class_id, class_source = 'human', class_validated = true, updated_at = now() WHERE item_key = ANY(:k) AND (...) IS DISTINCT FROM (...) RETURNING old.<counted>, new.<counted>`, `tally_context` as three reads (photo source, serving membership, near-duplicate group), one `INSERT INTO m3.sum_deltas SELECT ... FROM unnest(...)`, and the other statements of 6.9 (operation row and update, 2 registry reads, claims insert, 7 stand-ins for `load_item_inputs` and results, 2 events): about 22 round trips, `COMMIT`; ingest variant: insert 50 photos' items (about 7 per photo) into `item_state` with deltas and a 50 x 2 KB `bytea` insert standing for the vectors; baseline: the same transactions with gate A's `w_sum` hot-row upserts instead of deltas | 1, 8 and 30 concurrent writers, all labelling unvalidated items of class `person` (single labels and chunks of 1,000), 60 s per cell, closed loop (capacity) and paced at 15 labels per second per writer (latency, the bar's cells); then 30 ingest writers on `person` plus 30 labellers, paced | transactions per second; transaction p50 / p99; `Lock` and `LWLock` wait events by relation; delta rows per label | 30 writers: no `Lock` wait on `sum_deltas` or a summary table; label transaction p99 at 30 writers at most 2x its 1-writer p99; ingest writers scale with the writer count (plan 13 ingest table not below) |
| M2 read against pending deltas | S1a-i | P3 (class total, class plus flags) and P7 (200 cards) over the `_now` views | 8 readers (P7 also 1) while 0, 100 and 10,000 delta rows are pending per kind, spread over the 200 largest keys in +1 / -1 pairs so the totals stay true (compactor stopped) | p50 / p99 per backlog | P3 50 / 200 ms, P7 200 / 800 ms at 10,000 pending; else the largest backlog that meets them |
| M3 compaction | S1a-i | the 6.5 statement | batches of 1,000, 10,000 and 50,000 (beat 0.2 s), and 1 s and 5 s at 10,000, while 30 paced label writers of M1 run (about 450 labels per second); one closed-loop cell; a cell without compactor as control; compaction alone on a seeded backlog of 1,000,000 zero-sum rows | deltas compacted per second; compactor transaction time; writer p99 change | compaction at least twice M1's peak delta rate; writer p99 within 10 % of M1; picks `:batch` and the beat interval |
| M4 rebuild under writers | S1a-i | the 6.6 statements for `sum_items`, `sum_review`, `photo_counts` and the cluster kinds | 10 M rows, the 30 writers of M1 running | duration; writer p99 during it; oldest `backend_xmin` age; after rebuild and compaction, each view equals the `GROUP BY` | correct; writer p99 within 10 % of M1; duration recorded for the drift-check schedule and the partition fill of 6.7 |
| M5 HOT ratio and index cost | S1a-i (before `0011`) | M1's label and a region-only update (`region_status`, `region_max_score`), with `updated_at` moved by both, then only by the label (m6) | on clones `item_state_ff100`, `_ff90`, `_ff80`, `_ff70` (same columns and five B-trees, no foreign keys), 30 paced writers, 100 s per cell; 300 s mixed cells (15 label + 15 region writers) at fillfactor 100 and 80 | `n_tup_hot_upd / n_tup_upd` per table; WAL bytes per transaction (`pg_current_wal_lsn` difference); index sizes before and after | region-only updates HOT at least 90 % under the m6 rule; the numbers fill 15a 2.6's fillfactor rows |
| M6 capped counts and the drained queue | S1b-i | P4 with cap 10,001 for class plus date range, `origin`, text, detector class, and a whole-project filter; P1 and P4 on "unvalidated in class `person` after 99.5 % of it is validated" (an `UPDATE` sets that first) through X2 and through X4 | 1 and 8 readers | p50 / p99; rows read and heap fetches (`EXPLAIN (ANALYZE, BUFFERS)`) | 50 / 200 ms; picks `:count_budget_ms` as the p99 of the slowest passing cell |
| M7 sort paths | S1b-i | each sort of section 7: the sort path over 1,000, 10,000, 100,000 and 1 M filtered rows, standing in as `item_key BETWEEN a AND b` in a `MATERIALIZED` CTE, then `ORDER BY .. LIMIT 61`; the walk path at 0.1 %, 1 % and 10 % selectivity, standing in as `item_key % m = 0` with `enable_sort` and `enable_seqscan` off; 2.5 s (1 reader) and 3 s (8 readers) per cell | 1 and 8 readers, first and deep pages | p50 / p99 per cell | 50 / 200 ms; picks `T_sort`; lists sorts that miss on both paths |
| M8 several classes | S1b-i | P1's `UNION ALL` with 2, 4 and 16 classes, with and without a validated term, first and deep pages | 8 readers | p50 / p99; plan node (Merge Append or Sort) | 50 / 200 ms |
| M9 box rows and trigram | S4 | the 9 statements with X6, X26, X27 and the text forms of section 8 (exact, fragment, fuzzy for `m` 5 to 10, the variant list for `m <= 4`); GIN inserts | 8 readers; 30 writers inserting boxes with text, `fastupdate` on and off | page p50 / p99; insert p50 / p99 | pages 50 / 200 ms; picks `fastupdate` by insert p99 |
| M10 cluster cards and placement | S6 | P7 over the `_now` views; the `pl_*` joins of gate A on `m3.cluster_memberships`; P13 for the `all` tab per visible card; P13 cards under the cluster screen's filters (coordinator answer 5) | 8 readers | p50 / p99 | 200 / 800 ms; misses go to the coordinator with the numbers |

## 14. Round 3: outcome, causes and changes

Source of every number: `artifacts_local/speed_1113/m3_storage/` (gitignored), `results_round3.md`
and the cell records `raw/m3_<item>_10m.json`, cited as `m<N>` and the cell name. 10 M means about
10.6 M items, 1.36 M photos, 5.0 M boxes, 9.7 M memberships, 19,402 clusters in the serving
partition. Bars: page shapes 50 / 200 ms, dashboards 200 / 800 ms (p50 / p99, 8 readers; plan 15
section 6). **H** marks a cell that ran with host load 16 or more: its latency is an upper bound.

**Met** (no change): delta upkeep without a `Lock` wait on `sum_deltas` or a summary table (all 14
delta cells of `m1`, against 12,743 `transactionid` and 3,460 `tuple` waits of the hot-row design
at 30 writers); class totals with 10,000 pending deltas (7.8 / 8.0 ms p99, `m2`); compaction speed
and writer cost (6.5); the rebuild under writers, 0 mismatches (6.6); region-only updates HOT at
fillfactor 90 and below (`m5`); capped counts for class and date range, origin, whole project
(p99 13.7 to 18.0 ms, `m6a`); the drained review queue through X2 and X4 (p99 at most 45.5 ms,
`m6b_*`); several classes (p99 at most 9.1 ms, H, `m8`); the sort path up to 10,000 rows and the
walk path at 1 % and 10 % (`m7`); exact, state and detector box pages (`m9`); placement, facets,
locate and totals (`pl1`, `pl1b`, `pl4`, `pl9b`, `pl_total_*`, p99 at most 62.8 ms, `m10`).

### 14.1 The 12 statements that had to change to run

| # | Statement (revision 1) | What ran | Now in |
|---|---|---|---|
| 1 | compactor `ON CONFLICT (project_id, <keys>)` for all ten base tables | the key alone for kinds 6 to 10 (their keys hold no `project_id`) | 6.5 |
| 2 | rebuild written for `sum_items` only | all ten kinds, every measure; `max_updated_live` not corrected | 6.6 |
| 3 | cluster truth `sum(detector_confidence)` (float4) | `sum(detector_confidence::float8)` | 6.6 |
| 4 | `RETURNING old.x, new.x` | aliased pairs | 6.2 step 1 |
| 5 | keyset compared with text parameters | typed binding (`::uuid`) | section 4 |
| 6 | no order inside the compactor's upserts | `ORDER BY` the group keys | 6.5 |
| 7 | label transaction: one context read and 12 other statements | about 22 round trips, context as 3 reads | 6.9; section 13 M1 |
| 8 | the database "has `pg_trgm`" | `CREATE EXTENSION pg_trgm` | section 8 (`0011`); section 13 intro |
| 9 | pending deltas "per key read" | 100 and 10,000 rows per kind over the 200 largest keys, +1 / -1 pairs | section 13 M2 |
| 10 | M5 on `item_state`, 30 writers, 300 s | clones per fillfactor, 100 s cells, 300 s mixed cells | section 13 M5 |
| 11 | sort path over K filtered rows, walk at a selectivity | `item_key BETWEEN` in a materialised CTE; `item_key % m = 0` with sort and sequential scan off | section 13 M7; R4-5 replaces both with real class filters |
| 12 | compaction "while the 30 writers of M1 run" | 30 paced writers, one closed-loop cell, compaction alone on a seeded zero-sum backlog | section 13 M3 |

Review tabs were stand-ins (`check_machine`, `to_label`, `vlm_low`, `notes.md` section 2) because
the review-tab table does not exist yet; they count like the real ones but are not the real set.

### 14.2 Every miss: cause, options, the change

Each entry: the person's screen, the cause as the data shows it, the options with their documented
mechanism and write cost, the one choice, and what round 4 (section 15) must show.

**Cluster cards (P7), at every backlog.** Screen: the cluster grid, the screen a person labels
from; it requests every card at once (`max_clusters: 2000`), sorted by one of eight sorts, and
shows 24 at a time. Numbers: 332.8 / 504.4 ms with no pending delta, 343.3 / 481.0 at 100, 3,111.1
/ 3,320.9 at 10,000 (`m2` `b*_p7_r8`); 461.2 / 1,356.4 in `m10` `p7_cards_r8` (H). Cause, from the
plan of `m2` `b0_p7_r8`: ranking the clusters is a sequential scan of 273,471 `sum_cluster` rows and a
hash aggregate into 19,402 clusters, 212 ms of the 439 ms (the planner expected 67 groups); the
five `LATERAL` subqueries then cost about 1.1 ms per card (226 ms for 200; the largest cards hold
57 detector-class rows each); with 10,000 pending deltas each probe scans the kind's whole backlog,
since `sum_deltas` has no index on the cluster. The screen's real request, 2,000 cards, multiplies
the per-card part by ten. Options: (1) keep the form and add the partial index on the cluster
(369.7 / 462.9 ms at 10,000 pending, `b10000_p7_r8_idxcluster`): still over 200 ms p50 at 200 cards
and per card for 2,000; (2) a materialised view of the cards: each refresh recomputes every card
(REFRESH page) and cards are stale between refreshes; (3) one card row per cluster, rolled up by the
compactor for the clusters each batch touched, read with the pending deltas in one pass: no change
to writers, one extra statement per compactor batch that moves cluster kinds. **Chosen: (3)**,
`sum_cluster_card` (6.1, 6.5, 6.7) and P7 (section 2), plus `ix_sum_deltas_cluster` for one-cluster
reads. Round 4 R4-2: 2,000 cards, four sorts, at 0, 100 and 10,000 pending, against 200 / 800 ms;
compactor rate with the card refresh at least 26,382 rows per second (twice M1's peak).

**Cards under an item filter, and the `all` count per card (P13).** Screen: the cluster grid with
a filter (subject scope, blur, confidence, class, dates; `clusterListController.svelte.ts`), and the
card's "N to review" chip (`needs_review_count`, which is `review_counts.all`, `types.ts:275`), its
sort and its filter. Numbers: 6,095 / 7,209 ms for the `all` count of 200 cards, 6,081 / 7,400
filtered, one statement with a `LATERAL` per card no faster (6,067 / 7,122; `m10` `p13_*`); about
30 ms per card. Cause: the statement runs once per card over that card's members; the 200 largest
clusters hold many members, and the grid needs the numbers for every card it sorts, not only those
on screen. Options: per-card P13 for the 24 visible cards only (about 0.7 s, and the sort and filter
of the grid cannot be served); one set-based statement over the filtered members grouped by cluster
(cost grows with the rows the filter matches; gate A's unfiltered aggregation over all objects took
3,977 / 5,799 ms, `s3_all`); for the `all` count, a counted tab (owner question 1). **Chosen:** the
set-based P13 for filtered cards (section 2), the `all` count by owner question 1, and for
filters that match millions owner question 3. Round 4 R4-9: the set-based form for filters matching
10,000, 100,000, 1 M rows and the subject scope "largest object per photo" over the whole project.
The dashboard bar holds up to the matched size R4-9 reports; above it the owner's answer to
question 3 applies.

**Capped count by detector class.** Screen: every list total under a detector-class filter (the
vehicle workload filters by it all day). Numbers: 226.7 / 379.9 ms at 8 readers, 95 % cancelled at
a 25 ms budget (`m6a` `p4_detector_class_*`). Cause, from the plan of `p4_detector_class_r1`: a
sequential scan of `item_state` with `detector_class_name = 'car'`, 189,406 rows removed to find
10,001; no index on `item_state` holds the detector class. **This is the stand-in, not the design:**
P1's filter translation puts the detector class in an `EXISTS` on `detections`, served by X32
`(project_id, analyser_id, class_name, item_key)`, which round 3 did not time. Options: (1) the
plan's form through X32 (no write cost; X32 exists for the sort); (2) the detector class as a key of
`sum_items`, making the total exact (more base rows, no extra delta rows); (3) a sixth B-tree on
`item_state` (every label inserts one more index entry, `m5`: labels are never HOT). **Chosen: (1)**;
round 4 R4-4 times it with its plan. If it misses, (2) is the candidate for the next change, not (3).

**Text counts and text pages: fragment, fuzzy and variant matches.** Screens: the regions review tab
and every list with a plate-text filter. Numbers: text counts 1,824.6 / 7,140.4 ms (`EXISTS` form)
and 1,304.5 / 2,417.2 (trigram form), 100 % cancelled at every budget (`m6a`); fragment pages 410 to
1,593 ms p50, fuzzy 4,651 to 5,291, variant lists 64 to 170 (`m9`). Cause: row security. The plan of
the fragment count (`p4_text_gin_r1`) is a sequential scan with the policy and the `LIKE` in one
filter, and the documented rule (section 8) keeps a non-leakproof operator from running before the
policy; only the equality form, whose operator is leakproof, used an index (3.7 / 6.3 ms). The `m9`
cells recorded no plan; their shape agrees: a 3-character fragment (many matches, the scan stops
early) is faster than a 4-character one (410 vs 1,593 ms), and fuzzy costs a whole scan. Options:
mark the operators leakproof (rejected, section 8); an n-gram table on B-trees (rejected, hand-made
trigram search); a B-tree for every equality form plus one project-bound definer function for
fragments (no write cost beyond one B-tree entry per text box). **Chosen: the last** (section 8:
X33, the variant list for fuzzy at every length, `op_text_fragment`, `op_text_word`). Round 4 R4-3:
the mechanism from `pg_proc.proleakproof` and the plan as a role that bypasses row security; every
form with its plan against 50 / 200 ms, and fragment reads while 30 box writers fill the pending list.

**Sort path from 100,000 rows; walk path at 0.1 %.** Screens: every list with a sort other than the
filter's own index order. Numbers: sort path at 100,000 rows 76 to 564 ms p50 (all 17 sorts miss);
walk path at 0.1 % 92 to 530 ms p50 (10 of 12 miss), at 1 % only `crop_area_norm` misses by 2.6 ms
(`m7`). Cause: both paths cost per row read. The sort path costs about 1 µs per row when its key
is on `item_state` and 4.2 to 5.6 µs when it joins the key's table (section 7); the walk costs about
1.7 µs per entry on X1 and 8.7 µs per entry on another table's index. A walk at 0.1 % (10,600
matches) is not a path the rule takes: such filters are under `T_sort` and take the sort path,
which met at 10,000 for every sort. The stand-ins (section 14.1 item 11) read contiguous keys,
kinder than a real filter's scattered rows. Options: the sort path from `item_state`'s copies (no
write cost: the columns exist); more indexes on `item_state` (each a non-HOT insert per label;
rejected by 15a 2.6's five-index rule); a bar change for the band. **Chosen:** the sort path reads
`item_state`'s copies, the path is decided by the exact total or the sort path's own guard (section
7), `T_sort` from R4-5, and the remaining band goes to the owner (question 2). Round 4 R4-5: real
class filters of 10,000 to 100,000 matches; the 6 missing cells.

**Review by score (`pl2`), p99.** Screen: a review queue sorted by cluster fit. Numbers: p50 66.9 /
34.3 ms, p99 5,315.1 / 4,631.4 ms (first / deep), p95 5,123 and 4,405: bimodal (`m10`). Cause: the
cell timed the X23 walk for every parameter set, without the two-path choice of P8; one of its four
confidence bands per query matches few rows, so the walk reads most of the partition's 9.7 M
entries to fill 61 rows (no plan was recorded; which band is the slow one is inferred from the
bimodality, not shown). The capped count the rule would run first was itself slow (72.5 / 187.7
ms, `pl2_total_capped_r8`), while the counted tab's summary answered in 1.2 / 2.2 ms
(`pl2_total_summary_r8`). **Chosen:** no new mechanism: P8 as designed, with revision 2's decision
input (the counted tab's exact total, else the sort path's guard, section 7), so no separate
count runs. Round 4 R4-6: the routed form per band, with plans.

**Label updates are never HOT.** Numbers: 0 % HOT at every fillfactor (`m5`); each 100 s cell of
about 45,000 label updates grew X1 by 4.7 MB and X2 by 5.9 MB. Cause: by design. A label changes
`class_id`, `class_validated` and `updated_at`, which X1, X2 and X4 hold, and HOT needs "no columns
referenced by the table's indexes" changed ([HOT](https://www.postgresql.org/docs/18/storage-hot.html));
15a 2.6 already prices this as the cost of "Recently updated". There is no bar on label HOT; the
bar is region-only updates, which met: 100 % at fillfactor 90, 80, 70, 6.2 % at 100 (`m5`). Options:
take `updated_at` out of the indexes (the default list order goes) or fewer label-visible indexes
(the class pages go): both worse for the person. **Chosen:** keep; `item_state` fillfactor **90**,
the highest that met (heap 3.20 GB against 2.91 GB at 100 and 3.61 GB at 80, `m3_sizes_10m.json`).
Its mixed 300 s cell (labels and region writes together, where label versions also take page room)
ran at 100 and 80 only, and those cells re-labelled already labelled rows: round 4 R4-7 runs it at
90 with fresh rows; if region HOT falls under 90 % there, the fillfactor is 80.

**Label p99 at 30 writers against 1, closed loop.** Numbers: closed loop 58.97 / 29.21 ms = 2.02
(H, the engine at 100 % of 8 CPUs); paced at the design rate 89.3 / 82.5 ms response = 1.08
(`m1`). Cause: a closed loop of 30 writers on 8 CPUs measures queueing for CPU; the bar is about
latency at the label rate the design carries. **The bar is defined** (14.4) on the paced cells,
where it met; closed loops report capacity (1,133 labels per second at 30 writers).

**Chunks of 1,000 items.** Numbers: chunk p99 10,204 ms at 30 writers against 2,427 at 1 (4.2x;
the hot-row design 6.6x), 4,796 items per second at 30 writers, 531 at 1 (`m1` `label1000_*`, 60
transactions or fewer, so p99 is the maximum). Cause: 30 chunks of 1,000 rows and about 2,750 delta
rows each saturate 8 CPUs; `LWLock:BufferContent` showed 2,392 samples. What the person waits for:
a bulk label is a job (`label_op`, chunks of 1,000, 18b R5) with progress shown; nobody waits on one
chunk, but every other person's single label must stay fast while bulk jobs run. **The bar
changes** for chunks: (a) a bulk job's throughput is reported, not held to a ratio; (b) a single
label's paced response p99 stays within 2x of its value with no bulk job running, while `K`
`label_op` chunk jobs run at once; `K` is the per-deployment concurrency limit of that job kind,
fixed by round 4 R4-8 (candidates 1, 2, 4, 8). The person is not worse off: today a bulk label also
runs as a job.

### 14.3 Questions only the owner can answer

1. **What the cluster card's "N to review" count means.** Today it is the `all` review tab
   (`review_tab_clauses.py:111-151`), which also counts items flagged only by the probe's
   uncertainty and unclassed items that have a vector; those two facts live in result tables, so
   the count cannot come from the label-state summaries, and computing it per card took about 30 ms
   per card. Options: (a) keep today's meaning: probe runs and vector writes report each change
   through the item's write path (`store_states` reads entropy and vector presence in its context,
   under the item lock), so the probe job takes an item lock and writes a delta for each item whose
   membership changes, and a label pays about two more reads; (b) count the label-state signals
   only (VLM unmatched or new class pending, combine conflict, VLM empty or medium / low, cluster
   outlier, no class): exact and free, but the chip can be smaller than the `all` list it opens;
   (c) drop the chip, its sort and its filter. Recommendation: (a), because the chip is how a person
   picks the next cluster and a count that disagrees with the list it opens erodes trust; the cost
   lands on the probe job, not on the person.
2. **Sorts in the band at 10 M items.** For a filter matching about 40,000 to 100,000 items, seven
   sorts (creation time, crop area, rank in photo, detector confidence (the to-label tab's default),
   blur, detector class, cluster score) are estimated at about 75 ms for the first page (bar 50 ms);
   sorts with no walk index (validated, class name, region score (the regions tab's default), the
   audit order, detector disagreement) take 90 to 160 ms at 100,000 matches and about 1 s at 1 M.
   Options: (a) accept a slower first page for these, up to 200 ms p50 and 400 ms p99, deeper pages
   at the normal bar; (b) grey out such a sort when the filter is that large, with the reason; (c) an
   index per sort (each costs every label or region write an index entry). Recommendation: (a) up to
   100,000 matches, (b) above it, except the two tab defaults, which round 4 R4-5 times at real
   queue sizes before you decide.
3. **The cluster grid under a broad item filter** (for example "largest object per photo" over the
   whole project, matching millions). One aggregation over those members takes seconds at 10 M (gate
   A's unfiltered form 4.0 s; today's search engine 1.8 s at one client, `os10 s3_all`). Options:
   (a) accept it with a loading state the person can cancel; (b) make the subject scope (largest,
   two largest per photo) a key of the cluster summaries, so that control is exact and instant
   (cluster summary rows up to three times as many; a rank change emits cluster deltas); (c) refuse
   item filters on the grid above a size and ask to narrow. Recommendation: (a) now, (b) if the
   subject scope is used daily.

### 14.4 Bars given a checkable definition

| Bar (section 13) | Problem in round 3 | Definition from revision 2 |
|---|---|---|
| M1 "label p99 at 30 writers at most 2x its 1-writer p99" | closed loop measures CPU queueing | paced cells at 15 labels per second per writer: response p99 at 30 writers at most 2x the 1-writer paced response p99, cell load under 16 at start and end. Met: 1.08 (`m1`) |
| M1 "ingest writers scale (plan 13 ingest table not below)" | no factor; the table does not exist | storage never limits ingest: at 8 writers at least 1,000 photos per second committed (10x the 100 photos per second target of [13_backend_speed_plan.md](13_backend_speed_plan.md)), at 30 writers not below the 8-writer rate; warm cells of at least 60 s and 200 transactions. Round 3's small cells gave 1,125 and 1,472 (recomputed from `wall_s`): R4-10 confirms |
| M3, M4 "writer p99 within 10 % of M1" | two identical control cells differ by 28 % | inside one cell, 30 s windows alternating with and without the compactor (or rebuild): the median window p99 with it at most the larger of 1.10x and +10 ms of the median without. M4 met even against M1 (+3.3 %) |
| chunks | 14.2 | single-label paced response p99 with `K` chunk jobs at most 2x without; throughput reported |
| gate A "reads while 1,000 objects per second are relabelled" (plan 15 section 6) | not run: readers and writers never shared a cell | every page and dashboard shape meets its bar while 30 paced writers label 450 items per second (single labels) plus one chunk job (about 500 items per second) and the compactor runs; the writers' paced bar holds at the same time. R4-1 |

### 14.5 What the round 3 data does not reconcile

1. The same P7 statement gave 332.8 / 504.4 ms in `m2` and 461.2 / 1,356.4 in `m10` (later database
   state, H); not recorded why. The statement is replaced; R4-2 runs its first cell again at the end
   of the window to show the new form is stable.
2. P7 at 10,000 pending: 6,615.9 ms at 1 M but 3,111.1 at 10 M; the 1 M copy was not pristine.
   Replaced form; not pursued.
3. Two identical 30-writer cells without compactor differ by 28 % in response p99 (114.1 against
   89.3 ms): the reason for the windowed definition of 14.4.
4. WAL per label transaction 6,450 to 24,687 bytes across cells of one shape, tracking full-page
   images; checkpoint times were not recorded. R4-7 records `pg_stat_checkpointer` before and after.
5. The planner estimated 21,235 rows for 12 in the text count; under the chosen form the estimate
   no longer chooses the scan (the function and X33), so not pursued.
6. Three of 30 writers missing from the 30-writer paced hot-row cells: the baseline only.

## 15. Measurement round 4

Only what decides an open choice or closes a gap. Same engine, database `v2_10m`, schema `m3` as
round 3 left it (after the M6 drain and its restore, 0 mismatches), same runner. Each item is a step
of the chain under its own exclusive hold, load below 16 at start, as round 3:

```bash
cd artifacts_local/speed_1113/m3_storage
flock -o -w 21600 <the lock path chain.sh sets> ./hold.sh r4_<n> 585 m3_run.py r4_<n> 10m --budget 500
```

**Harness changes** (`m3_run.py`, `m3_readers.py`, `m3_sql.py`, before the window; dry-run each on
`v2_1m`): register `r4_0` to `r4_10` in `ITEMS`; pass `explains=` for every card, text, count and
routed-sort form so each cell records `EXPLAIN (ANALYZE, BUFFERS)`; a reader-and-writer cell (`rwcell`:
`wcell` writers and `rcell` readers forked together, both recorded); a windowed writer cell that
starts and stops the compactor every 30 s; a sort generator over real class filters of known size
(classes picked from `sum_items` by count, so the matched rows are scattered as a real filter's) in
place of `item_key BETWEEN` and `item_key % m`; a backlog generator from real label transactions
over at least 5,000 clusters (not +1 / -1 pairs on 200 keys); `pg_stat_checkpointer` in the writer
records. Delete the 3 stale M7 records first (`results_round3.md` section 3).

| Item | Question | Bar | Data and set-up | Cells | Time |
|---|---|---|---|---|---|
| R4-0 set-up (shared hold) | | | `sum_cluster_card` filled by the roll-up of 6.5; `ix_sum_cluster_det`, `ix_sum_deltas_cluster`; X33 on `box_state`; role `op_text_search` and `op_text_fragment`, `op_bound_project_strict` (superuser connection); `ANALYZE` | none | about 4 min (round 3: kinds 7 to 10 rebuild 67 s idle; X33 over 5 M rows estimated 1 min) |
| R4-1 reads under writers | do the page and dashboard bars hold while labels are written? | 14.4 last row | 8 readers cycling P1 class page (deep), P3 class total, P4 class and dates, P7 (2,000 cards), P8 routed, exact text, fragment; 30 paced label writers, one chunk writer, compactor 10,000 / 0.2 s | 1 cell of 180 s; the same readers alone 60 s | 5 min |
| R4-2 cluster cards | does the card row form meet the dashboard bar, and does the compactor keep its rate with the refresh? | P7 200 / 800 ms at 10,000 pending; compactor at least 26,382 rows/s; 0 mismatches after | P7 with 2,000 cards, sort drawn per query from `size_desc`, `needs_review_desc`, `dominant_class`, `updated_desc`; backlog 0 / 100 / 10,000 real deltas; detector form; compactor with refresh alone on a 1 M real backlog and under 30 paced writers; consistency check | 3 backlogs x 30 s at 8 readers, 3 x 20 s at 1 reader, detector 30 s, compactor 2 x 60 s, first cell repeated at the end | 6 min |
| R4-3 text | is row security the cause, and do X33 and the function meet the page bar, also while boxes are inserted? | 50 / 200 ms; insert p99 not above round 3's 512 kB cell (17.3 ms) | `SELECT proname, proleakproof FROM pg_proc WHERE proname IN ('textlike', 'texteq', 'similarity_op', 'textregexeq')`; the fragment count's plan as `m3_app` and as a role with `BYPASSRLS`; exact; fuzzy variants for 1 to 10 characters; fragment 3 / 4 / 5 characters first and deep; text counts (P4) by function and by variants; fragment readers with 30 box writers at 512 kB | about 18 reader cells x 20 s, 1 mixed cell 60 s | 8 min |
| R4-4 detector count | does the plan's form (EXISTS through X32) meet the page bar? | 50 / 200 ms | P4 cap 10,001 with `detector_class_name` as `EXISTS` on `detections`, alone and with a class; 1 and 8 readers | 4 x 20 s | 1.5 min |
| R4-5 sorts | where is `T_sort`; how wide and how slow is the band; the 6 missing cells | 50 / 200 ms per cell; outputs `T_sort` and the band per sort | the 6 cells of `photos_unlabeled_count` (round 3 section 3); sort path on `item_state` copies for `created_at`, `crop_area_norm`, `crop_rank_in_image`, `detector_confidence`, `detector_class_name` at 10,000 / 20,000 / 40,000 / 70,000 / 100,000 matches; walk path (planner settings unforced) at 40,000 / 70,000 / 100,000 for those plus `blur_lap_ratio`, `cluster_score`; sort path for `validated`, `class_name`, `region_score` at 20,000 / 40,000 / 70,000; the to-label and regions queues at the sizes the data holds; first and deep, 8 readers | about 116 cells x 3 s plus 6 | 2 holds, 11 min |
| R4-6 review by score | does the routed P8 meet the page bar in every confidence band? | 50 / 200 ms | `pl2`'s predicate, 4 bands, routed by the counted tab's total else the sort path's guard; plans per band | 8 x 20 s | 3.5 min |
| R4-7 fillfactor | does fillfactor 90 keep region-only updates at least 90 % HOT when labels run beside them? | 90 % HOT | `item_state_ff90`, 15 label + 15 region writers paced, labels on fresh rows, 300 s; checkpoint counters | 1 cell | 5 min, then restore and consistency check 3 min (shared hold) |
| R4-8 chunks | how many `label_op` chunk jobs may run at once? | 14.4 chunks row | 8 paced single-label writers alone, then beside K = 1, 2, 4, 8 chunk writers | 5 x 60 s | 5 min |
| R4-9 filtered cards | up to which matched size does the set-based P13 meet the dashboard bar? | 200 / 800 ms | filters matching about 10,000, 100,000 and 1 M items, and subject scope "largest object per photo" over the project; 8 readers for the first two, 1 reader (at most 6 queries) for the others | 4 cells, 30 to 60 s | 3.5 min |
| R4-10 defined bars | the windowed compactor bar and ingest throughput | 14.4 | 30 paced writers, 6 x 30 s windows alternating the compactor; ingest at 1, 8, 30 writers, 20 s warm-up, 60 s cells | 1 + 3 cells | 7 min |

Total: about 63 minutes of holds (estimated from round 3's cell durations), about 80 minutes with
start-up and the shared steps; one quiet window of **90 minutes**, with the chain's 4-minute load
gate per cell as the only reason to run longer. Order keeps the data clean: R4-0, the read-only
items (R4-3, R4-4, R4-5, R4-6, R4-9), R4-2, then the writing items (R4-1, R4-8, R4-10), R4-7 last
with its restore and the consistency check. Every cell writes `raw/m4_<item>_10m.json`; the
write-up follows `results_round3.md`'s form (outcome per bar, the file and field of every number).

## 16. What the Rust record slices take from this revision

Slice codes and names are those of plan 18b; "takes" lists the read shapes, indexes, summary tables,
settings and statements each builds from this file, with the section here.

| Slice (18b) | Takes from 15b revision 2 | Open until round 4 or the owner |
|---|---|---|
| R1b, the record and the fold | `0011`: X1 to X21, X24 to X33 (15a 2.6), `CREATE EXTENSION pg_trgm`, X7 and X25 with `fastupdate` on and a 512 kB pending list (8); the summary tables, `sum_deltas`, the `_now` views, `ix_sum_deltas_cluster` (6.1); role `op_text_search`, `op_bound_project_strict`, `op_text_fragment`, `op_text_word` (8); `item_state` fillfactor 90, `sum_deltas` vacuum threshold 10,000 (6.1, 6.5, 14.2); `store_states` with aliased `RETURNING` pairs (6.2); the tally (6.2); the compactor for all ten kinds, key-only conflict targets for kinds 6 to 10, ordered upserts, batch 10,000, beat 0.2 s, built table-driven so R8 adds the card refresh as a second statement without changing the first (6.5); the drift check over ten kinds with float sums in double precision (6.6); the label transaction's statements (6.9) | fillfactor 90 or 80 (R4-7); the compactor's rate with the card refresh (R4-2, read by R8) |
| R4, gallery reads | P1 with up to 16 class branches, P2, P4 with a 50 ms budget, P5, P6, P15 (2, 5); the cursor with typed binding and the chosen path (4); sorts reading `item_state`'s copies, the path decided by the exact total or the sort path's guard (7); capped totals (5) | `T_sort` and the band (R4-5, owner question 2); the detector-class count form (R4-4); reads under writers (R4-1) |
| R5, labelling, review, class counts | counted review tabs (6.2); P3 totals; the review queue P8 with the routed choice (2, 7); class counts from `sum_items` (row 16); chunks of 1,000 with a concurrency limit `K` on `label_op` (14.2) | `K` (R4-8); the routed review queue (R4-6); the "to review" definition (owner question 1) |
| R7, regions and boxes | P9 with X6, X26, X27 (9, met); the text forms of section 8 (X33 for exact, fuzzy and same-text; `op_text_fragment` for fragments; the 3-character minimum) | text bars and the pending list under writers (R4-3). 18b's R6b row names `0015` as the place of the `fastupdate` choice; X7 is created in `0011` with it (coordinator to align 18b) |
| R8, clustering and the cluster screens | `0016`: the four cluster summary tables, `ix_sum_cluster_det`, `sum_cluster_card` (6.1); the compactor's kinds 7 to 10 and the card refresh (6.5); the partition fill with the card roll-up (6.7); P7, P8 inside a cluster, P13 set-based (2); facets and placement reads (met) | P7 against its bar (R4-2); filtered cards (R4-9, owner question 3); the "to review" count (owner question 1) |
| R10, machine helpers | only if the owner answers question 1 with (a): the probe job and the embedding re-index report membership changes through `store_states` | owner question 1 |
| R6a, R6b, R9, R2b | nothing new beyond revision 1 (6.7's run-keyed fill for R6b) | none from this revision |

## Appendix. Review trace

One row per finding of the review of 15b and 15c (2026-11-15) and the place that now answers it.

| Finding | Answered in |
|---|---|
| B1 `image_contents` still used by core ingest | 15c S1a-ii (kept; `0012` only fences it), 15c section 8 (module table) |
| B2 exclusive rebuild lock | 6.3, 6.6, 6.7; 15a section 7; section 13 M4 |
| B3 summary rows locked to commit | 6.2, 6.3 (no row lock); 15a 2.6 (rank 8 removed); section 13 M1 |
| M1 class family on X2 | P1 (branches per class up to 16, family drops merged classes); section 7 two paths; M8 |
| M2 drained queue and end-game class | P1 (X2 with `class_validated`); 15a X4 review partial index; P8 note; section 5 rules 1, 3, 8; `sum_review`; M6 |
| M3 index per sort against five | section 7 placement; 15a 2.3 `items` copies, 2.6 X28 to X32; section 11 question 7; M5 |
| M4 drift paths | 6.2 callers (erasure, `fold.rebuild`, class merge, capture), 6.6 drift check; 15a sections 7 and 8 |
| M5 capped counts as guards | section 5 rules 2, 7 and the two tables; rows 36, 38, 43, 50; 15c section 2 |
| M6 box rows | section 9; 15a X26, X27; M9 |
| M7 dropped reader behaviours | P1 filter forms and normaliser test; P15 policy; section 7 script filters and VLM sort; rows 15, 24, 30, 43; 6.1 `missing` bucket; section 8 fuzzy bound; 15c section 2 |
| M8 attach without an owner | 15c S1a-ii |
| M9 deletion before the last importer | 15c section 8 |
| M10 red by file | 15c section 9 |
| M11 slices too large | 15c section 1 and section 3 (splits); section 12 |
| M12 round trips per label | 6.9; M1 |
| m1 P6 short page | P6 |
| m2 X22 direction | 15a X22; P8 note |
| m3 upsert order | 6.5 (writers insert only; the compactor's `GROUP BY`) |
| m4 advisory key space | 6.3 (no tally lock); 15a section 7 |
| m5 P4 heap fetches | section 5 rule 8; M6 |
| m6 machine churn non-HOT | 15a 2.4; section 11 question 8; M5 |
| m7 GIN `fastupdate` | section 8; M9; 15c S4 entry |
| m8 one filter file for six slices | P1 (one module per family) |
| m9 `photo_counts` serialises a photo | 6.1, 6.2 (a delta kind) |
| m10 mixed-direction sorts | section 4; section 7 table |
| m11 capped-total classification | section 5 rules 1 and 9 |
| Jobs (question 6 of the review) | section 10 |
