# 15b: storage reads, every query shape on the record

| Field | Value |
|---|---|
| Status | plan, revision 3.1 (2026-11-16; rows "Revision 2", "Revision 3" and "Revision 3.1" below). Revision 1 (2026-11-15): the read map of plan 15 section 12, written after the gate A measurement at 1 and 10 million objects (8 clients held); revision 1 folds in the independent review of 15b and 15c and the coordinator's decisions on it (appendix "Review trace"): the summaries are a delta log (section 6). Companion of [15_storage_plan.md](15_storage_plan.md) and [15a_storage_schema.md](15a_storage_schema.md); the cut-over slices that build it are [15c_storage_slices.md](15c_storage_slices.md) |
| Code base read | `int/storage` at `7cadf2968` (the integration code of `int/b4-stateless` plus storage step 1: `src/domain/claim*.py`, `fold*.py`, `results.py`, migration `0010_storage_record`, `src/db/repos/`, `src/services/record/apply.py`) |
| Checklist | the storage inventory of 2026-11-13 (`artifacts_local/speed_1113/storage_inventory.md`, gitignored), section 3: shapes 1 to 80 with file:line. Its paths are relative to `src/services/curation/` unless they start with another directory; its line numbers were re-checked on this tree (same code) |
| Measurement | gate A (`artifacts_local/speed_1113/v2_read_shapes/`, gitignored): `notes.md`, `queries_pg.sql`, `summary_pg.sql`, `raw/time_pg_1m.json`, `raw/time_os_1m.json`, `raw/time_pg_10m.json`, `raw/time_os_10m.partial.json` |
| Decided by the coordinator | page fetches are SQL on the record; exact totals only from summary keys, else a capped count; dashboards read summary tables kept by the writer; cluster placement is the side table; text substring search is a trigram index; the search engine leaves the curation stack (section 1.2 of the inventory stays until phase 3); the fallback for a pending measurement that fails is additive (a search copy without vectors for the failing shapes only) |
| Revision 2 (2026-11-16) | measurement round 3 folded in (`artifacts_local/speed_1113/m3_storage/results_round3.md`, `notes.md`, cells under `raw/`, gitignored). The 12 statements that had to change to run are corrected where they live (list in section 14.1). Every miss has its cause, options and one chosen change in section 14; round 4 (section 15) measures only what is still open; section 16 says what each Rust record slice of plan 18b takes from this file. The backend that builds this is Rust (plan 18 section 3.2): the statements here are its specification, the Python module names are where today's behaviour lives |
| Revision 3 (2026-11-16) | fix pass after the independent review of revision 2 (`artifacts_local/speed_1113/m3_storage/plan15b_r2_review.md`, gitignored; 7 major, 13 minor; trace in the appendix). Binding constraint added: the same backend later runs on **managed PostgreSQL** (for example Amazon RDS or Aurora), where nobody is a superuser, so no role can be given `BYPASSRLS` and no function can be marked `LEAKPROOF` (section 8). Changed: fragment and word search read **owner-read search copies** in schema `txt` through definer functions owned by a role without any special attribute (section 8, replacing revision 2's `BYPASSRLS` role); the function runs as its own statement with a cap and a per-call plan; fuzzy over a defined alphabet with a switch; the strict project function and the tightened scope function are written out; the card table gains the drift check's statement, the touched-set statement and a full join for clusters with only pending deltas (6.5, 6.6, P7); the sort band now adds the guard's cost to the walk (section 7); round 4 rewritten (section 15); every slice item marked final or waiting (section 16) |
| Revision 3.1 (2026-11-16) | fix pass after the second review (`artifacts_local/speed_1113/m3_storage/plan15b_r3_review.md`, gitignored; 1 blocker, 7 major, 8 minor; trace in the appendix). The search-copy mechanism stays, with defence in depth made part of the design: the read functions return item keys only and every use joins them to a forced base table in the caller's own statement; the probe raises on a foreign key; the copies carry a project foreign key and a project-led index (`btree_gin`); `op_app` holds `USAGE` and `EXECUTE` and nothing on the tables (writes and per-item tests are functions); the function text lives in one SQL file; a two-project matrix test, a mutation run and a catalogue test are the go bar; totals count after every filter in one snapshot; backup and restore written step by step for self-hosted and managed (02c section 6, no `BYPASSRLS` needed); round 4 grows by about 6 minutes |

**Revision 2 in one paragraph** (text mechanism replaced by revision 3, section 8). The delta log, the compactor, the rebuild, the capped counts and
the several-class pages met their bars at 10 million items. Changed by the measurement: cluster
cards read one card row per cluster, kept by the compactor (P7, 6.1, 6.5); box text uses a B-tree
for exact, variant and fuzzy matches and a search copy read through a project-bound function for
fragments, because row security keeps the trigram index from being used (section 8); the sort path reads its
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

Kept as the record of revision 1; nothing in this table is pending any more. What is still open
is section 15 (round 4), and section 16 says which slice item waits for which round 4 cell.

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
SELECT cluster_uid, coalesce(c.n_live, 0) + coalesce(d.n_live, 0) AS n,
       coalesce(c.validated_live, 0) + coalesce(d.validated_live, 0) AS validated,
       coalesce(c.labelled_live, 0) + coalesce(d.labelled_live, 0) AS labelled,
       greatest(c.max_updated_live, d.max_updated) AS max_updated,
       (coalesce(c.conf_sum_live, 0) + coalesce(d.conf_sum_live, 0))
         / NULLIF(coalesce(c.conf_n_live, 0) + coalesce(d.conf_n_live, 0), 0) AS avg_conf,
       coalesce(c.needs_review, 0) + coalesce(d.needs_review, 0) AS needs_review,
       coalesce(c.review_n, '{}') AS review_n, coalesce(c.top_class_ids, '{}') AS top_class_ids,
       coalesce(c.top_class_n, '{}') AS top_class_n, coalesce(c.top_det, '') AS top_det,
       coalesce(c.dup_groups, 0) AS dup_groups
FROM (SELECT * FROM sum_cluster_card WHERE partition_id = :serving) c   -- revision 3: a full join, so a
     FULL JOIN pend d USING (cluster_uid)                                -- cluster with only pending deltas shows
WHERE coalesce(c.n_live, 0) + coalesce(d.n_live, 0) > 0
  AND <card filters: min_size, max_size, validation_state, needs_review, flag, dominant_class, detector_class>
ORDER BY <card sort, tiebreak cluster_uid>
LIMIT :max_clusters;
```

A cluster whose only rows are pending deltas (new, or after a merge or a split) has no card row
until the compactor's next batch creates it (6.5, at most one beat, 0.2 s); the full join shows it
with exact counts and empty derived fields for that time (revision 3, review M6). The join
condition is one equality on `cluster_uid`, which a hash join serves; R4-2 records the plan.

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
| 6 | `GET /regions`, training candidates, suspected false positives, cohorts (`region_rows.py:222`; `cohorts.py:33`; `routers/curation/regions.py:255`, `:425`) | S4 | page | P9 (box predicate in both statements; a fragment as section 8's steps 1 to 3); `total_rows` P3 on `sum_box` for a state and/or detector filter, else P4; `total` (items) P4 | X6, X26, X27, X7 | spike `s8_deep` 194.7 / 593 and `s8_total` 600 / 2496; new design to be measured |
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
-- revision 2: one cluster's pending deltas without scanning the kind's whole backlog (round 3, M2);
-- in 0016 with the cluster kinds it serves (sum_deltas is filled then: CREATE INDEX CONCURRENTLY, 15a 2.6)
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
36.7 ms). That is one cell inside a 28 % noise band (14.5 item 3), and round 3's compactor cells ran
without the index, so every round 4 writer cell (R4-1, R4-2, R4-10) runs with it and records it.

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
, touched AS (                                -- revision 3: the clusters whose cards statement 2 rolls up
  SELECT DISTINCT partition_id, cluster_uid FROM moved WHERE kind BETWEEN 7 AND 10
)
SELECT (SELECT count(*) FROM moved) AS n_moved,
       (SELECT array_agg(partition_id ORDER BY partition_id, cluster_uid) FROM touched) AS touched_partitions,
       (SELECT array_agg(cluster_uid ORDER BY partition_id, cluster_uid) FROM touched) AS touched_clusters;
```

The two arrays are in key order and `NULL` when the batch moved no cluster kind (statement 2 is then
skipped). Kinds 7 to 10 all carry `partition_id` and `cluster_uid` (6.1).

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
the first statement returns beside its count (`touched` above). The roll-up reads base rows only (the moved deltas
are in them now), so the card equals the roll-up of its base rows at commit. Statement 2a locks the
existing cards before 2b takes its snapshot (in read committed "each command" sees the data
committed before it began, transaction isolation page): a second compactor waits in 2a for the
first to commit, and its 2b then rolls up a snapshot that includes the first. A card two lapsed
compactors create at once is the one case 2a cannot lock (no row yet); the drift check corrects it. The drift check
also compares every card with the roll-up of its base rows and rewrites a differing card, by the
card check of 6.6 (its own read-committed step with 2a's lock, not inside the repeatable-read
rebuild). Cost: a
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

**Card check** (revision 3, review M6). The cards are not rewritten inside the repeatable-read
transaction above: an `UPDATE` (or `INSERT ... ON CONFLICT DO UPDATE`) of a card a compactor changed
after the snapshot fails there with a serialization error ([transaction isolation, repeatable
read](https://www.postgresql.org/docs/18/transaction-iso.html#XACT-REPEATABLE-READ)), and a rewrite
from that old snapshot in a later transaction would put an older roll-up over a newer card. After
the rebuild commits and the compactor has folded its correcting deltas, the check walks the serving
and pending partitions' cards in key order, 500 clusters per transaction, in **read committed**,
with the compactor's own two statements: 2a locks the batch's existing cards `FOR UPDATE` in key
order; 2b rolls them up from the base rows and writes only cards that differ:

```sql
-- 2a as in 6.5, :touched_* = the next 500 (partition_id, cluster_uid) in key order of the union of
--   sum_cluster_card's keys and sum_cluster's distinct (partition_id, cluster_uid) (a missing card too)
-- 2b as in 6.5, with the conflict action limited to a differing card and the rewrites counted
INSERT INTO sum_cluster_card AS c (...) SELECT ... FROM touched t ORDER BY 1, 3
ON CONFLICT (partition_id, cluster_uid) DO UPDATE SET n_all = EXCLUDED.n_all  -- ... every column
  WHERE (c.n_all, c.n_live, c.labelled_live, c.validated_live, c.max_updated_live, c.conf_sum_live,
         c.conf_n_live, c.needs_review, c.review_n, c.top_class_ids, c.top_class_n, c.top_det, c.dup_groups)
        IS DISTINCT FROM (EXCLUDED.n_all, EXCLUDED.n_live, EXCLUDED.labelled_live, EXCLUDED.validated_live,
         EXCLUDED.max_updated_live, EXCLUDED.conf_sum_live, EXCLUDED.conf_n_live, EXCLUDED.needs_review,
         EXCLUDED.review_n, EXCLUDED.top_class_ids, EXCLUDED.top_class_n, EXCLUDED.top_det, EXCLUDED.dup_groups)
RETURNING c.partition_id, c.cluster_uid;      -- the rewritten cards: logged, an alert when any
```

Why it never writes an older roll-up: in read committed each statement sees what was committed
before it began (transaction isolation page). A compactor that committed before 2a is in 2b's
roll-up; a compactor that commits later waits in its own 2a for this check's lock and then rolls up
again over a snapshot that includes everything. Clusters with base rows but no card (the creation
race of 6.5) are in that union and are created. `max_updated_live` is
compared like the rest (the card takes the roll-up of its base rows). R4-2 injects a wrong card and
a missing card and asserts one rewrite each.

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
`RETURNING` on the writes it already makes, `tally_context` (3 reads, below) and one `INSERT`. A single
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
- **What decides the path** (revision 3: three steps, cheapest first; review M5). (1) When the
  filter is summary keys the exact total (P3) decides. (2) Else the page's own capped count (P4, cap
  10,001, which page 1 runs anyway for its total) decides when it is not capped: at most 10,000
  matches take the sort path. (3) A capped count says only "more than 10,000"; then the planner's
  row estimate of the filter (`EXPLAIN` without `ANALYZE`, planning only) decides when it is far
  from `T_sort`: above `3 x T_sort` the walk path runs at once, below `T_sort / 3` the sort path.
  (4) In between, the sort path's read is the guard: it materialises at most `T_sort + 1` matching
  rows (`LIMIT` inside the read) and switches to the walk when it got `T_sort + 1`. A wrong
  estimate costs time, never a wrong page: both paths return the same rows. The first page's choice
  rides in the cursor (section 4). Whether the estimate is good enough to skip the guard is R4-5's
  question (estimate against actual for every filter it runs); if it is not, step 3 is dropped.
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
  **What the person waits for just above `T_sort` is the sum** (revision 3, review M5): the capped
  count (p99 13.7 to 18.0 ms, `m6a`), then the guard (about `T_sort` rows at about 1 µs, 40 to 50 ms
  at `T_sort` = 40,000 to 50,000), then the walk (for a joined sort at 45,000 matches about 61 x
  10.6 M / 45,000 = 14,400 entries at 8.7 µs = 125 ms; for `updated_at` on X1 at 1.7 µs, 24 ms).
  First page estimates: about 190 ms for the joined sorts and about 90 ms for `updated_at`, so
  the slow band is "from `T_sort` to about 100,000 matches, first page up to about 200 ms" for the
  joined sorts, and `updated_at` misses the 50 ms bar there too unless step 3 skips the guard.
  Deeper pages pay only the walk (the choice rides in the cursor). R4-5 times this composed path
  (cells "guard then walk"); these sums are estimates from round 3's rates, not measurements.
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

## 8. Text search: B-trees for equality, owner-read search copies for fragments and words

Revision 3 replaces revision 2's mechanism (a role with `BYPASSRLS` owning the fragment function),
which cannot be set up on managed PostgreSQL and whose inner plan was never looked at (review M1,
M2). What stays: X33 for every equality form, the variant list for fuzzy (now over a defined
alphabet), the 3-character minimum for fragments. Revision 3.1 (second review) keeps the copies and
adds the layers of "How isolation holds" below.

**Box text** (today the `text_fold` keyword and its `text_fold.sub` wildcard field,
`region_text_search.py:1-11`; the review tab's case-insensitive wildcard). `box_state.text_fold` is
already folded by `src/domain/region_text.py` (`normalize` keeps letters and decimal digits, folds
Latin accents, upper-cases; then the profile's confusable map), so case needs no `ILIKE` and a folded
value holds no space, `%`, `_` or `\`. X33, `CREATE INDEX ix_box_state_text ON box_state
(project_id, text_fold) WHERE text_fold IS NOT NULL`, serves every equality form on `box_state`
itself. The trigram indexes X7 and X25 sit on the search copies below. `pg_trgm` is installed by
`0010`'s bootstrap step (15a section 12; "trusted", installable by a non-superuser with `CREATE` on
the database, [pg_trgm](https://www.postgresql.org/docs/18/pgtrgm.html)); round 3's spike database
lacked it (section 14.1 item 8), so `0011` only checks that it is there.

**Why row security keeps `LIKE` off an index.** "An index scan cannot be selected for queries on
security barrier views (or tables with row-level security policies) if an operator used in the
`WHERE` clause is associated with the operator family of the index, but its underlying function is
not marked `LEAKPROOF`" ([rules and privileges](https://www.postgresql.org/docs/18/rules-privileges.html));
"the system will enforce conditions from security policies ... before any user-supplied conditions
from the query itself that contain non-leakproof functions" ([CREATE FUNCTION,
`LEAKPROOF`](https://www.postgresql.org/docs/18/sql-createfunction.html)). Round 3's plans agree:
both text counts (`m6a` `p4_text_gin_r1`, `p4_text_exists_r1`) are sequential scans of `box_state`
with the policy and `text_fold ~~ '%ZTNP%'` in one filter, 5,099,275 rows removed for 12. Not
shown yet: a control plan without row security, and the `proleakproof` flags; the equality form's
index use (3.7 / 6.3 ms, `m9` `exact_first_r8`) is inferred from timing, since `m9` recorded no plan.
So the cause is documented and consistent with the plans; R4-3 records the control and the flags.

**Binding constraint: managed PostgreSQL.** The backend must also run on a managed service such as
Amazon RDS or Aurora for PostgreSQL, where nobody is a superuser. The most privileged user there is
`CREATE ROLE postgres WITH LOGIN NOSUPERUSER INHERIT CREATEDB CREATEROLE NOREPLICATION`, a member of
`rds_superuser`, which may add supported extensions, create roles and grant privileges ([Amazon RDS,
the rds_superuser role](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Appendix.PostgreSQL.CommonDBATasks.Roles.rds_superuser.html)).
PostgreSQL: "`CREATEROLE` does not confer the ability to grant or revoke the `BYPASSRLS` privilege"
([role attributes](https://www.postgresql.org/docs/18/role-attributes.html)); "only superuser roles
or roles with `BYPASSRLS` can specify `BYPASSRLS`" ([CREATE ROLE](https://www.postgresql.org/docs/18/sql-createrole.html));
`LEAKPROOF` "can only be set by the superuser" (CREATE FUNCTION). So on such a service no role of
ours can bypass row security and no function can be marked leakproof. The AWS pages do not say this
in one sentence; it follows from the master role's attributes and the PostgreSQL rules, and the
cloud check C-1 (section 15) confirms it on a real instance. A mechanism that needs either is not
chosen unless nothing else meets the bar.

**The mechanisms** (review: "do not simply patch the definer function"). Sizes and rates from the
spike data: about 2.65 M boxes with text at 10 M items (text on 80 % of plate and OCR boxes, which
are 65 % of 5.1 M boxes; 5 to 8 characters A to Z and 0 to 9; gate A `notes.md` section 1), X7 108 MB
(`m3_sizes_10m.json`), box writers 1,470 to 1,524 boxes per second (`m9` `gin_insert_*`).

| | (a) definer function owned by a `BYPASSRLS` role (revision 2) | (b) mark the operator functions `LEAKPROOF` | (c) posting table `(project_id, trigram, item_key)` probed by equality, `LIKE` as a recheck | (d) owner-read search copy (**chosen**) |
|---|---|---|---|---|
| How the index becomes usable | the role bypasses every policy, `FORCE` included ([row security](https://www.postgresql.org/docs/18/ddl-rowsecurity.html)) | `textlike` and pg_trgm's operator functions treated as leakproof, so `LIKE` may run before the policy | only equality on `uuid`, `integer`, `bigint` in index conditions (leakproof, R4-3 checks `uuid_eq`, `int4eq`, `int8eq`) | the copy's owner "normally bypass[es] row security" when the table is not `FORCE`d (row security page), and a definer function runs "with the privileges of the user that owns it" (CREATE FUNCTION), so no policy applies inside it |
| Isolation, fail closed | `project_id = op_bound_project_strict()` inside; raises unbound | the policy still applies to every caller | the policy applies to every caller (the table is an ordinary forced table); unbound raises in the policy | inside each function `project_id = op_bound_project_strict()` (raises unbound or outside project scope), and the functions return item keys only, joined by the caller to a forced base table before use (four layers, below); `op_app` has no privilege on the copies, and the policy would bind it (it is not the owner) if one were added |
| Bootstrap and migration, self-hosted | bootstrap (`op_super`) creates the role; changing a function's owner needs `SET ROLE` to the new owner and its `CREATE` on the schema ([ALTER FUNCTION](https://www.postgresql.org/docs/18/sql-alterfunction.html)), which revision 2 did not grant | superuser `ALTER FUNCTION ... LEAKPROOF` on system and extension functions, outside any migration | none beyond today's | bootstrap creates a role with no attributes and its schema; the migration (`op_owner`) creates the objects under `SET ROLE` (below) |
| Managed | impossible (no `BYPASSRLS`) | impossible (no superuser) | works | works: the same statements, the bootstrap run by the master user |
| Plan and bound | a GIN read inside a fence; bound as (d) once written so | as a plain query | per fragment of `k` characters, `k - 2` posting lists read and intersected; bounded only by the longest list (plate digit groups repeat) | one statement per call, planned with the actual fragment and project, at most `max_items` rows read from the heap (below) |
| Write cost | none beyond X7 | none | about 4.5 interior trigrams per text: about 12 M rows at 10 M, about 0.7 GB heap and 0.55 GB B-tree (60 and 46 B a row), against X7's 108 MB; about 6,800 row inserts per second at round 3's box rate | one upsert of one row per item whose box texts change, plus its GIN entries (about X7's, plus one project entry per row); about 2.65 M rows at most, about 0.16 GB heap and 0.06 GB key |
| Verdict | not chosen: cannot be set up on managed PostgreSQL | not chosen: superuser only, and wrong by definition: `LIKE` raises on some patterns, and "a function which throws an error message for some argument values but not others ... is not leakproof" (CREATE FUNCTION) | costed fallback, built only if (d) misses R4-3 or its go bar cannot pass: no privilege change at all and no function body to trust, but a hand-made trigram search 12 times X7's size | chosen: the documented GIN index, no special attribute, the same on both platforms |

Also considered: a `security_barrier` view (the same leakproof rule applies to it, rules page);
`btree_gin` to put `project_id` into the GIN index under row security (does not lift the rule; it is used on the copies, where no policy applies inside the function, so the index is scoped by project: [btree_gin](https://www.postgresql.org/docs/18/btree-gin.html), trusted, `uuid` supported, and "for queries that test both a GIN-indexable column and a B-tree-indexable column, it might be more efficient to create a multicolumn GIN index ... than to create two separate indexes"; on Amazon RDS `btree_gin` 1.3 and `pg_trgm` 1.6 are listed for PostgreSQL 17 and 18, [supported extension versions](https://docs.aws.amazon.com/AmazonRDS/latest/PostgreSQLReleaseNotes/postgresql-extensions.html)); partitioning by project
(DDL per project, rejected in 15a section 1).

**Roles and objects of (d)** (02a 1.5 has the role table; 02c section 6 the backup and restore). The
role `op_text_owner` is `NOLOGIN` with no attribute; it owns schema `txt`, the two copies and their
functions, and holds nothing else but `SELECT (item_key, project_id)` and `REFERENCES (item_key)` on
`items`, `REFERENCES (id)` on `projects` and `EXECUTE` on `op_bound_project_strict()`. `op_app` is
not a member of it and gets exactly `USAGE` on schema `txt` and `EXECUTE` on the seven functions:
**nothing on the two tables** (second review, decision 3), so the writes and the broad-path test go
through functions too. Privileges are the PostgreSQL defaults plus these grants ([privileges](https://www.postgresql.org/docs/18/ddl-priv.html):
no privilege on tables or schemas for `PUBLIC` by default, `USAGE` on a schema "allows the grantee to
look up objects within the schema", `EXECUTE` on functions is granted to `PUBLIC` by default).

**One source for the SQL** (second review, M7). The DDL below is the specification. Its only code
copy is one file, `src/db/sql/txt_search.sql`, in three marked parts (`tables`, `indexes`,
`functions`): migration `0011` runs all three in its transaction, the restore (02c section 6) runs
`tables`, then `functions`, then the refill, then `indexes`. No migration and no restore step holds
function text of its own: a changed function is an edit of the file plus a migration that runs the
file again. `db verify` and a `db` test compare the `prosrc` of every function in `txt` with the
file's text (the migration runs the file's current text on a fresh database; the file's statements
are idempotent and need only objects `0011` already has).

```sql
-- bootstrap (src/db/roles.py: ensure_roles, prepare_database), idempotent; self-hosted as op_super,
-- managed as the service's master user (CREATEROLE: the creator gets ADMIN on the new role, "SET FALSE",
-- role attributes page, so it grants itself SET before creating the schema owned by the role)
CREATE ROLE op_text_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
GRANT op_text_owner TO CURRENT_USER WITH SET TRUE;                       -- needed on managed only
CREATE SCHEMA IF NOT EXISTS txt AUTHORIZATION op_text_owner;
GRANT op_text_owner TO op_owner WITH ADMIN FALSE, INHERIT FALSE, SET TRUE;   -- migrations SET ROLE to it
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- both "trusted" (pg_trgm, btree_gin pages): CREATE on the database suffices
CREATE EXTENSION IF NOT EXISTS btree_gin;  -- operator class uuid_ops for the project column of the copies' index

-- 0011, as op_owner (owner of public's tables and of the scope functions)
CREATE OR REPLACE FUNCTION op_scope_project() ...   -- 02a 1.4, tightened: 'project' with an empty id raises
CREATE FUNCTION op_bound_project_strict() RETURNS uuid
  LANGUAGE plpgsql STABLE SET search_path = pg_catalog, pg_temp AS $f$
DECLARE v_pid text := current_setting('op.project_id', true);
BEGIN
  IF coalesce(current_setting('op.scope', true), '') <> 'project' THEN
    RAISE EXCEPTION 'project_scope_required' USING ERRCODE = 'insufficient_privilege';
  END IF;
  IF coalesce(v_pid, '') = '' THEN
    RAISE EXCEPTION 'project_not_bound' USING ERRCODE = 'insufficient_privilege';
  END IF;
  RETURN v_pid::uuid;
END $f$;
REVOKE ALL ON FUNCTION op_bound_project_strict() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION op_bound_project_strict() TO op_app, op_text_owner;
GRANT SELECT (item_key, project_id), REFERENCES (item_key) ON items TO op_text_owner;   -- the copies' foreign key; the writers' join
GRANT REFERENCES (id) ON projects TO op_text_owner;                                      -- the copies' project foreign key

-- txt_search.sql, part "tables" (each part begins with SET ROLE op_text_owner and ends with RESET ROLE,
-- so everything it creates is owned by that role)
SET ROLE op_text_owner;
CREATE TABLE IF NOT EXISTS txt.item_box_text (           -- one row per item with box text (S4, R7)
  item_key bigint PRIMARY KEY REFERENCES public.items(item_key) ON DELETE CASCADE,
  project_id uuid NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
  folds text NOT NULL                                    -- the item's box text_fold values, ' '-separated, by box_id
);
CREATE TABLE IF NOT EXISTS txt.item_line_text (          -- one row per item with OCR lines (today item_text_tokens)
  item_key bigint PRIMARY KEY REFERENCES public.items(item_key) ON DELETE CASCADE,
  project_id uuid NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
  words text NOT NULL                                    -- item_text_tokens of the item's lines, ' '-separated
);
-- both: ENABLE ROW LEVEL SECURITY without FORCE (the one exception to 02a 1.4, see the tests); policy
-- p_<table>_project exactly as 02a 1.4, dropped and created again by every run. op_app holds no
-- privilege on the tables, so the policy is a second lock on a door that is already shut: if a
-- grant to op_app is ever added by mistake, op_app (not the owner) is still bound by it.
REVOKE ALL ON txt.item_box_text, txt.item_line_text FROM PUBLIC;

-- part "indexes": X7 and X25, project first (btree_gin, GIN multicolumn; R4-3 measures it against folds alone)
CREATE INDEX IF NOT EXISTS ix_item_box_text_trgm ON txt.item_box_text
  USING gin (project_id, folds public.gin_trgm_ops) WITH (fastupdate = on, gin_pending_list_limit = 512);
CREATE INDEX IF NOT EXISTS ix_item_line_text_trgm ON txt.item_line_text
  USING gin (project_id, words public.gin_trgm_ops) WITH (fastupdate = on, gin_pending_list_limit = 512);

-- part "functions": one transaction, as the definer page asks ("create it and set the privileges
-- within a single transaction"); the default below makes a new function start without PUBLIC too
ALTER DEFAULT PRIVILEGES REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;   -- for objects op_text_owner creates

-- reads that return item keys only
CREATE OR REPLACE FUNCTION txt.fragment_items(q_frag text, q_max_items integer) RETURNS SETOF bigint
  LANGUAGE plpgsql STABLE SECURITY DEFINER PARALLEL UNSAFE
  SET search_path = pg_catalog, pg_temp SET plan_cache_mode = force_custom_plan
  SET gin_fuzzy_search_limit = 0 AS $f$
#variable_conflict error
DECLARE v_project uuid := public.op_bound_project_strict();     -- raises before anything is read
BEGIN
  IF q_frag IS NULL OR length(q_frag) NOT BETWEEN 3 AND 64
     OR q_max_items IS NULL OR q_max_items NOT BETWEEN 1 AND 10001
     OR strpos(q_frag, '%') > 0 OR strpos(q_frag, '_') > 0 OR strpos(q_frag, '\') > 0 OR strpos(q_frag, ' ') > 0 THEN
    RAISE EXCEPTION 'text_fragment_invalid' USING ERRCODE = 'invalid_parameter_value';
  END IF;
  RETURN QUERY SELECT t.item_key FROM txt.item_box_text AS t
               WHERE t.project_id = v_project AND t.folds LIKE '%' || q_frag || '%'
               LIMIT q_max_items;
END $f$;
CREATE OR REPLACE FUNCTION txt.word_items(q_words text[], q_max_items integer) RETURNS SETOF bigint
  LANGUAGE plpgsql STABLE SECURITY DEFINER PARALLEL UNSAFE
  SET search_path = pg_catalog, pg_temp SET plan_cache_mode = force_custom_plan
  SET gin_fuzzy_search_limit = 0 AS $f$
#variable_conflict error
DECLARE v_project uuid := public.op_bound_project_strict(); v_longest text;
BEGIN
  IF q_words IS NULL OR cardinality(q_words) NOT BETWEEN 1 AND 16
     OR q_max_items IS NULL OR q_max_items NOT BETWEEN 1 AND 10001
     OR EXISTS (SELECT 1 FROM unnest(q_words) AS u(w)
                WHERE u.w IS NULL OR length(u.w) NOT BETWEEN 1 AND 64 OR strpos(u.w, '%') > 0
                   OR strpos(u.w, '_') > 0 OR strpos(u.w, '\') > 0 OR strpos(u.w, ' ') > 0) THEN
    RAISE EXCEPTION 'text_words_invalid' USING ERRCODE = 'invalid_parameter_value';
  END IF;
  SELECT u.w INTO v_longest FROM unnest(q_words) AS u(w) ORDER BY length(u.w) DESC, u.w LIMIT 1;
  IF length(v_longest) < 3 THEN                          -- a short-word query takes the predicate, not the index
    RAISE EXCEPTION 'text_words_invalid' USING ERRCODE = 'invalid_parameter_value';
  END IF;
  RETURN QUERY SELECT t.item_key FROM txt.item_line_text AS t
               WHERE t.project_id = v_project
                 AND t.words LIKE '%' || v_longest || '%'                       -- the GIN condition
                 AND NOT EXISTS (SELECT 1 FROM unnest(q_words) AS u(w)          -- every word starts a token
                                 WHERE strpos(' ' || t.words, ' ' || u.w) = 0)
               LIMIT q_max_items;
END $f$;
CREATE OR REPLACE FUNCTION txt.fuzzy_items(q_key text, q_threshold real, q_max_items integer) RETURNS SETOF bigint
  LANGUAGE plpgsql STABLE SECURITY DEFINER PARALLEL UNSAFE
  SET search_path = pg_catalog, pg_temp SET plan_cache_mode = force_custom_plan
  SET gin_fuzzy_search_limit = 0 SET pg_trgm.word_similarity_threshold = 0.6 AS $f$
#variable_conflict error
DECLARE v_project uuid := public.op_bound_project_strict();
BEGIN
  IF q_key IS NULL OR length(q_key) NOT BETWEEN 1 AND 64
     OR q_threshold IS NULL OR q_threshold NOT BETWEEN 0.1 AND 1
     OR q_max_items IS NULL OR q_max_items NOT BETWEEN 1 AND 10001
     OR strpos(q_key, '%') > 0 OR strpos(q_key, '_') > 0 OR strpos(q_key, '\') > 0 OR strpos(q_key, ' ') > 0 THEN
    RAISE EXCEPTION 'text_fuzzy_invalid' USING ERRCODE = 'invalid_parameter_value';
  END IF;
  -- local to the function: the SET clause above restores the value at exit (CREATE FUNCTION, SET)
  PERFORM set_config('pg_trgm.word_similarity_threshold', q_threshold::text, true);
  RETURN QUERY SELECT t.item_key FROM txt.item_box_text AS t
               WHERE t.project_id = v_project AND q_key OPERATOR(public.<%) t.folds
               LIMIT q_max_items;
END $f$;

-- per-item tests for the broad walk (section 8, step 3): a boolean about a key the caller already holds
CREATE OR REPLACE FUNCTION txt.box_text_has(q_item_key bigint, q_frag text) RETURNS boolean
  LANGUAGE plpgsql STABLE SECURITY DEFINER PARALLEL UNSAFE SET search_path = pg_catalog, pg_temp AS $f$
#variable_conflict error
DECLARE v_project uuid := public.op_bound_project_strict();
BEGIN
  IF q_item_key IS NULL OR q_frag IS NULL OR length(q_frag) NOT BETWEEN 3 AND 64
     OR strpos(q_frag, '%') > 0 OR strpos(q_frag, '_') > 0 OR strpos(q_frag, '\') > 0 OR strpos(q_frag, ' ') > 0 THEN
    RAISE EXCEPTION 'text_fragment_invalid' USING ERRCODE = 'invalid_parameter_value';
  END IF;
  RETURN EXISTS (SELECT 1 FROM txt.item_box_text AS t
                 WHERE t.item_key = q_item_key AND t.project_id = v_project AND t.folds LIKE '%' || q_frag || '%');
END $f$;
CREATE OR REPLACE FUNCTION txt.line_text_has(q_item_key bigint, q_words text[]) RETURNS boolean
  LANGUAGE plpgsql STABLE SECURITY DEFINER PARALLEL UNSAFE SET search_path = pg_catalog, pg_temp AS $f$
#variable_conflict error
DECLARE v_project uuid := public.op_bound_project_strict();
BEGIN
  IF q_item_key IS NULL OR q_words IS NULL OR cardinality(q_words) NOT BETWEEN 1 AND 16
     OR EXISTS (SELECT 1 FROM unnest(q_words) AS u(w)
                WHERE u.w IS NULL OR length(u.w) NOT BETWEEN 1 AND 64 OR strpos(u.w, '%') > 0
                   OR strpos(u.w, '_') > 0 OR strpos(u.w, '\') > 0 OR strpos(u.w, ' ') > 0) THEN
    RAISE EXCEPTION 'text_words_invalid' USING ERRCODE = 'invalid_parameter_value';
  END IF;
  RETURN EXISTS (SELECT 1 FROM txt.item_line_text AS t
                 WHERE t.item_key = q_item_key AND t.project_id = v_project
                   AND NOT EXISTS (SELECT 1 FROM unnest(q_words) AS u(w)
                                   WHERE strpos(' ' || t.words, ' ' || u.w) = 0));
END $f$;

-- writes: the only way op_app changes a copy. Each row is taken only for an item of the bound project,
-- by two separate clauses (the explicit one and the policy of items, which applies to op_text_owner
-- because it does not own items); a NULL text removes the item's row
CREATE OR REPLACE FUNCTION txt.put_box_text(q_item_keys bigint[], q_folds text[]) RETURNS integer
  LANGUAGE plpgsql VOLATILE SECURITY DEFINER PARALLEL UNSAFE SET search_path = pg_catalog, pg_temp AS $f$
#variable_conflict error
DECLARE v_project uuid := public.op_bound_project_strict(); v_n integer;
BEGIN
  IF q_item_keys IS NULL OR q_folds IS NULL OR cardinality(q_item_keys) <> cardinality(q_folds)
     OR cardinality(q_item_keys) > 10000 THEN
    RAISE EXCEPTION 'text_put_invalid' USING ERRCODE = 'invalid_parameter_value';
  END IF;
  DELETE FROM txt.item_box_text AS t USING
    (SELECT u.k FROM unnest(q_item_keys, q_folds) AS u(k, f)
       JOIN public.items AS i ON i.item_key = u.k AND i.project_id = v_project WHERE u.f IS NULL) AS d
  WHERE t.item_key = d.k AND t.project_id = v_project;
  INSERT INTO txt.item_box_text AS t (item_key, project_id, folds)
    SELECT u.k, v_project, u.f FROM unnest(q_item_keys, q_folds) AS u(k, f)
      JOIN public.items AS i ON i.item_key = u.k AND i.project_id = v_project
     WHERE u.f IS NOT NULL ORDER BY u.k
  ON CONFLICT (item_key) DO UPDATE SET folds = EXCLUDED.folds
    WHERE t.project_id = v_project AND t.folds IS DISTINCT FROM EXCLUDED.folds;
  GET DIAGNOSTICS v_n = ROW_COUNT;
  RETURN v_n;
END $f$;
-- txt.put_line_text(q_item_keys bigint[], q_words text[]) RETURNS integer: the same body over txt.item_line_text

REVOKE ALL ON FUNCTION txt.fragment_items(text, integer), txt.word_items(text[], integer),
  txt.fuzzy_items(text, real, integer), txt.box_text_has(bigint, text), txt.line_text_has(bigint, text[]),
  txt.put_box_text(bigint[], text[]), txt.put_line_text(bigint[], text[]) FROM PUBLIC;
GRANT USAGE ON SCHEMA txt TO op_app;
GRANT EXECUTE ON FUNCTION txt.fragment_items(text, integer), txt.word_items(text[], integer),
  txt.fuzzy_items(text, real, integer), txt.box_text_has(bigint, text), txt.line_text_has(bigint, text[]),
  txt.put_box_text(bigint[], text[]), txt.put_line_text(bigint[], text[]) TO op_app;
RESET ROLE;                                              -- back to op_owner for the rest of 0011
```

`0011` writes the body of `op_scope_project` once (02a 1.4 with the `'project'` branch raising
`project_not_bound` on an empty id instead of returning `NULL`, review M7): with `NULL` a policy
compares `project_id = NULL` and returns nothing, silently. Every function body is written to the
rules of the definer page ([CREATE FUNCTION](https://www.postgresql.org/docs/18/sql-createfunction.html),
"Writing `SECURITY DEFINER` Functions Safely"): the search path is `pg_catalog, pg_temp` ("`pg_temp`
as the last entry") and every other name is schema-qualified; `LIKE`, `strpos`, `unnest` and `||`
are in `pg_catalog`, the operator class and `<%` are named with their schema (`public` keeps
`USAGE` for every role by default, so `op_text_owner` needs no grant there); the first statement of
each body is `#variable_conflict error` and **every reference is qualified** (second review, M6):
the parameters carry the prefix `q_`, the locals `v_`, every column carries its table alias and
every `unnest` its alias `u(...)`, so no name can be both a variable and a column ([PL/pgSQL
variable substitution](https://www.postgresql.org/docs/18/plpgsql-implementation.html#PLPGSQL-VAR-SUBST):
with `error` an ambiguous name raises instead of choosing). Revision 3's `word_items` named its
parameter `words`, the same as `t.words`; that was the defect. **Every body is compiled and run once
in round 4's set-up** (R4-0: the harness extracts this block unedited and calls each function with a
valid and an invalid argument) before any slice copies it into the file; a body that fails stops
the round. `SET gin_fuzzy_search_limit = 0` pins the one caller-settable parameter that could thin a
GIN scan silently ([client connection defaults](https://www.postgresql.org/docs/18/runtime-config-client.html)).
The `ALTER DEFAULT PRIVILEGES` line follows its page: it "revokes the public EXECUTE permission that
is normally granted on functions, for all functions subsequently created by" the role, and a
per-schema form would not do it; the explicit `REVOKE` and `GRANT` in the same transaction stay,
so a function is never visible to `PUBLIC` even for an instant ([ALTER DEFAULT
PRIVILEGES](https://www.postgresql.org/docs/18/sql-alterdefaultprivileges.html); the page does not
say whether `pg_restore` honours them, so the restore also creates the functions through the file,
02c section 6). `SET ROLE` makes the objects owned by `op_text_owner` without the `ALTER ... OWNER`
step revision 2 could not perform; it needs only the membership `WITH SET` from the bootstrap
([SET ROLE](https://www.postgresql.org/docs/18/sql-set-role.html)).

**How isolation holds, and what one mistake could expose** (second review, M7). A forced policy cannot
guard the copies: a definer function runs "with the privileges of the user that owns it" and the
owner "normally bypass[es] row security" ([row security](https://www.postgresql.org/docs/18/ddl-rowsecurity.html)),
and an owner that is forced cannot use the trigram index (the leakproof rule above). So inside a
function the barrier is one clause per statement, `t.project_id = v_project`; where a forced table
guards every statement, here a body that lost the clause would return every project's rows. Four
layers surround that clause so that its loss cannot expose text:

1. **The read functions return item keys only; nothing they return reaches the backend unless it
   was joined, in the caller's own statement running as `op_app`, to `items` or to the table the
   shape reads (`item_state`, `box_state`)**, whose forced policy filters the keys by the caller's scope.
   A body without the clause would return other projects' keys and the join would drop them. Text
   is never returned: the fuzzy candidates' text is read from `box_state` under its own policy
   (below). The two per-item tests return a boolean about a key the caller took from its own rows, and
   the writers return a count.
2. **The route probe detects it.** Its one statement (below) compares the keys the function produced
   with the keys that survived the join and, if they differ, the reader raises `text_isolation_fault`
   (an error, no result, a metric): in correct operation they are equal, because a copy row exists
   only for an item that exists in the same snapshot (foreign key, `REPEATABLE READ`).
3. **Each copy carries `project_id` with a foreign key to `projects` and the trigram index leads with
   it**, so a correct function walks only its own project's entries and a small project's search
   does not pay for a large neighbour's matches (cost, not isolation); deleting a project deletes its
   copy rows, as for every other table (15a section 1).
4. **The go bar below**: a two-project matrix over every function, a run with the clause removed
   from each statement in turn that must fail, and a catalogue test that no function lacks a matrix.

*Residual exposure, plainly.* If a function body lost its clause and the matrix test, the mutation
run and the probe all failed to catch it, a caller could still learn, from the time a call takes and
from the single yes-or-no the probe raises as an error, that another project has a matching text;
it would receive no text, no key and no count of another project. Accepted: it needs an error in a
reviewed, tested body, it exposes one bit and a timing, and the alternative that has none (the
posting table probed by equality, below) is 12 times the size and is kept as the costed fallback.

*Scopes.* The functions refuse the deployment scope and the core scope (`project_scope_required`,
`insufficient_privilege`), which is stricter than the policies (they admit the deployment scope):
no code running at the deployment scope can search text across projects, no admin or retention
feature may plan on it (a feature that needs it adds its own function and its own matrix), and core
images (project `NULL`) have no copy rows (`project_id NOT NULL`).

**The fallback stays costed.** The posting table `(project_id, trigram, item_key)` probed by
equality (mechanism (c)) keeps `FORCE` on every table and has no function body to trust. Round 4
builds it only if the copies miss R4-3's bars or the go bar below cannot be made to pass; its size
and write cost are in the table above.

**The plan inside, and its bound** (review M2). A PL/pgSQL statement is prepared once and may get
a generic plan "not dependent on specific parameter values"; a fresh plan "can use the current
parameter values" ([PL/pgSQL plan caching](https://www.postgresql.org/docs/18/plpgsql-implementation.html#PLPGSQL-PLAN-CACHING));
`SET plan_cache_mode = force_custom_plan` on the function makes every call a custom plan with the
actual fragment and project ([PREPARE](https://www.postgresql.org/docs/18/sql-prepare.html),
[plan_cache_mode](https://www.postgresql.org/docs/18/runtime-config-query.html#GUC-PLAN-CACHE-MODE)).
One row per item means no `DISTINCT`, so `LIMIT max_items` stops the heap phase at `max_items` rows:
the expected plan is a bitmap scan of X7 with both conditions in the index (`project_id = <constant>` and the `LIKE`; cost: the posting lists of the fragment's `k - 2`
trigrams intersected with the project's, GIN's fast scan skips along the shorter lists, [GIN](https://www.postgresql.org/docs/18/gin.html)) and at most `max_items` heap rows with their `LIKE` recheck. The
planner may instead pick a sequential scan with `LIMIT` when it expects many matches; R4-3 records
the inner plan of every call through `auto_explain` with `log_nested_statements` ("statements
executed inside a function", [auto_explain](https://www.postgresql.org/docs/18/auto-explain.html)),
and if any inner plan reads more than ten times `max_items` rows sequentially the functions get
`SET enable_seqscan = off` ([planner method configuration](https://www.postgresql.org/docs/18/runtime-config-query.html#RUNTIME-CONFIG-QUERY-ENABLE)).
`PARALLEL UNSAFE` is the default and is declared on purpose: a statement that contains the function
is planned serial (the `PARALLEL` paragraph of CREATE FUNCTION), which costs nothing here, since
every text statement is an index walk or a bounded sort (below); the function's `ROWS` default (1,000)
is only an estimate, and the `MATERIALIZED` CTE keeps it from being rescanned.

**How a page, a count and a sort use it** (reviews M3 and, second review, M4 and m3). The page's
transaction is `REPEATABLE READ READ ONLY` for every text read, so the probe and the statements after
it see one snapshot ([transaction isolation](https://www.postgresql.org/docs/18/transaction-iso.html#XACT-REPEATABLE-READ)).
The function is never a bare subquery: the reader writes it as a `MATERIALIZED` CTE, so it runs once
per statement and a nested loop cannot rescan it per outer row ([WITH queries](https://www.postgresql.org/docs/18/queries-with.html#QUERIES-WITH-CTE-MATERIALIZATION)).
Its keys are used only through a join to a base table, never returned.

1. **Route probe**, one statement, `max_items` = 10,001 (the list cap plus one):
   `WITH tk AS MATERIALIZED (SELECT k.item_key FROM txt.fragment_items(:frag, 10001) AS k(item_key))
   SELECT count(i.item_key) AS visible, count(*) > count(i.item_key) AS foreign_keys FROM tk LEFT JOIN
   items i ON i.item_key = tk.item_key` (or `word_items`). `items` is forced, so a key of another
   project does not join. `foreign_keys` true raises `text_isolation_fault` and the read ends.
2. **Narrow** (`visible` at most 10,000): the text predicate is the semi-join `s.item_key IN (SELECT
   item_key FROM tk)` inside the shape's own statement (P1, P8; P9's first statement with
   `b.item_key`), with `tk` as above, **together with the shape's other filters, its sort, its limit
   and its total**: the page carries `count(*) OVER ()` over the rows that pass every filter (at most
   10,000 rows, the sort path of section 7, whose bar met at 10,000 rows, `m7`), and a count shape
   (P4) counts the same statement's rows with the section 5 cap. The total is thus exact and counts
   what passes **all** the filters, not the number of text keys.
3. **Broad** (`visible` is 10,001): the page takes the walk path of its sort with the per-item test
   `txt.box_text_has(s.item_key, :frag)` (`txt.line_text_has(s.item_key, :words)` for words) as a
   filter on rows the walk already holds; the total is the capped count of section 5 over the whole
   predicate (text and every other filter, same snapshot), "10,000+" only when that count reaches
   the cap. The per-item test is a definer function, since `op_app` has no privilege on the copy;
   the plan row it replaces was an `EXISTS` on the copy. R4-3 times the per-call cost (a plpgsql
   call per walked row) in the deep cells. A broad match fills a page after few entries (section 7,
   walk path). A fragment or word query whose longest word has fewer than three characters skips
   the probe and takes this step directly.
4. **Fuzzy** (the trigram form below): one statement reads the candidates' text from the base table,
   `WITH tk AS MATERIALIZED (SELECT k.item_key FROM txt.fuzzy_items(:key, :threshold, 10001) AS k(item_key))
   SELECT b.item_key, b.box_id, b.text_fold FROM box_state b WHERE b.item_key IN (SELECT item_key FROM
   tk) AND b.text_fold IS NOT NULL` (policy of `box_state`, `uq_box_state_item_box`); the domain
   function keeps the items with a word at distance at most 1 and their kept keys enter step 2's
   statement as `= ANY(:kept)`. The keys that enter it come from `box_state` rows, not from the function.

P9's second statement, the matching boxes of the page's items, applies the text test to `box_state`
rows already bounded by the keys (at most 61 items, at most 74 boxes each; section 9):
`... WHERE b.project_id = :p AND b.item_key = ANY(:keys) AND b.text_fold LIKE :like ORDER BY
b.item_key DESC, b.ordinal` (a filter after the policy on those rows). For the trigram-fuzzy form it
has no `:like`: it returns the bounded rows' `text_fold` and the domain function marks the boxes
within distance 1 (second review, m4a). R4-3 times steps 1 to 4 inside P1, P4, P8 and P9 at three
selectivities.

| Match (`TextMatch`) | SQL | Index |
|---|---|---|
| `exact` | `b.text_fold = :key` | X33 |
| `fuzzy` (`FUZZY_FUZZINESS = 1`, transpositions, `region_text_search.py:29-30`) | closed alphabet: `b.text_fold = ANY(:variants)`; otherwise the trigram form below (step 4 above) | X33; X7 through `txt.fuzzy_items` |
| `fragment` (3 to 64 characters, after `fold`) | `txt.fragment_items` through the probe, then steps 2 and 3 above | X7 |
| `same_text_as` (`same_text_key`, `region_text_search.py:37`) | the box's own `text_fold` by `(item_key, box_id)` (`uq_box_state_item_box`), then `exact` | X33 |
| item text words (`?item_text=`, `item_text.py:110-121`) | `txt.word_items` with the query's words (`normalize_tokens`) through the probe, then steps 2 and 3 | X25 |

**Fuzzy: the variant list, its alphabet and its switch** (review M4). A key of `n` characters over
an alphabet `A` has at most `n(|A| - 1)` substitutions, `(n + 1)|A|` insertions, `n` deletions and
`n - 1` transpositions, each one probe of X33. `A` is defined per profile revision by one domain
function: the characters `c` with `fold(c) = c` (so no confusable-map key, which never occurs in a
folded value: `confusables_issue` keeps the fold idempotent) that match the profile's one-character
regex `text_charset` (`src/config/detection_profile.py:111-114`), taken from the letters and
decimal digits `normalize` keeps. The list is used when `text_charset` is set, `|A|` is at most 40
and the key at most 12 characters: at most 12 x 39 + 13 x 40 + 23 = 1,011 probes (the spike: `A` =
A to Z and 0 to 9, 36 characters, 546 strings for 7 characters). Otherwise (`text_charset` empty,
which keeps every non-space character; a larger alphabet; a longer key) the search switches to
`txt.fuzzy_items(key, threshold, 10001)` (above, step 4): pg_trgm's word similarity `key <% t.folds`
(served by X7, pg_trgm page) with the threshold the domain function derives from the key's length,
returning candidate keys only; step 4 reads their `text_fold` from `box_state` under its policy and
the domain function keeps exactly the items with a word at Damerau-Levenshtein distance at most 1. The threshold formula is R4-3's (recall against a
brute-force distance on the spike's texts must be 100 % for keys of 1 to 12 characters). When
10,001 candidates come back, the page shows the matches among them and marks the total unknown
(`total_unknown`, section 5); R4-3 reports how often that happens on the spike. A box text read by a human or another reader that
holds a character outside `A` is found by `exact`, `fragment` and the trigram form, not by the
variant list (stated, tested). A unit test checks the variant generator and the alphabet function
against a brute-force distance on generated texts.

**A named product change** (review M4): today's fuzzy query stops at `FUZZY_MAX_EXPANSIONS = 50`
(`region_text_search.py:30`, `:80`); both forms here return every stored text within distance 1, a
superset of today's results when more than 50 stored texts are that close. The pins of the 50 go:
`tests/region_text/test_plate_search.py:31` and `tests/curation/golden/item_filter_clauses_golden.json:713`.

**Words** (today `item_text_tokens`, a prefix query per word, `src/domain/item_text.py:89-121`). The
copy holds exactly today's tokens (each word of each line plus the joined line, `item_text_tokens`),
so "every word of `q` is a prefix of some token" is `strpos(' ' || words, ' ' || w) > 0` for every
word, as in `word_items`. Words have only letters and digits (`normalize_tokens`), so no escaping is
needed; the function refuses anything else. **Short words** (review M7): pg_trgm extracts no trigram
from fewer than three characters and "a pattern with no extractable trigrams will degenerate to a
full-index scan" (pg_trgm page). A query whose longest word has one or two characters takes the walk
path directly (step 3, with every word as the `strpos` test on `txt.item_line_text` by primary key),
so today's one- and two-character prefixes keep working; a fragment under three characters is
refused with 400 `text_too_short` (revision 1, 15c contract addendum).

**Who writes the copies, and how a photo's rows leave them.** `store_states` (6.2), the only writer of
`box_state`, calls `txt.put_box_text(:keys, :folds)` for the items whose box `text_fold` values
changed, in the same transaction after step 2, with the arrays ordered by `item_key`:
`SELECT array_agg(c.k ORDER BY c.k), array_agg(x.folds ORDER BY c.k) FROM unnest(:changed) AS c(k)
LEFT JOIN LATERAL (SELECT string_agg(b.text_fold, ' ' ORDER BY b.box_id) AS folds FROM box_state b
WHERE b.item_key = c.k AND b.text_fold IS NOT NULL) x ON true`; an item with no text left gets a
`NULL`, which removes its copy row. The repository that writes `text_lines` rows with an `item_key`
(`src/db/repos/text_search.py`, new) calls `txt.put_line_text` the same way with the tokens of
`item_text_tokens` (the tokens are made by the domain code, so the function takes them as
arguments; it takes a row only for an item of the bound project). The ratchet of 6.2
(`tests/storage/test_counted_writers.py`) names the writer of each copy. Both run as `op_app` under
`project_tx`; `op_app` has no privilege on the tables, so no other path writes them. The copy is
safe from lost updates only because `store_states` holds the `items` lock first (lock rank 6, 15a
section 7: "a fold reads its inputs after it holds the `items` lock"): two writers that change
different boxes of one item serialise there, and a two-writer test (go bar, item 6) proves it.
Write cost: one row and its GIN entries per changed item; `box_state` keeps X33, so a text change
there is not HOT (as before).

**Erasure** (second review, M3). The copies are in the erasure catalogue (15a section 8) as "cascade
through `items`": both foreign keys to `items` are `ON DELETE CASCADE`, so the rows of a photo leave
the copies in the same transaction as the `items` rows (a referential action runs inside the
deleting statement and bypasses row security, [row security](https://www.postgresql.org/docs/18/ddl-rowsecurity.html)),
and the foreign key to `projects` removes a deleted project's rows. `store_states(delete_photos=...)`
does nothing extra for them. Erased text stays in dead heap tuples and the GIN pending list until
vacuum, as in every table; the erasure record says so.

**GIN pending list** (round 3, M9): `fastupdate` on with `gin_pending_list_limit` 512 kB gave the
lowest insert p99 under 30 box writers (service 8.2 / 17.3 ms p50 / p99, against 14.3 / 40.6 ms with
`fastupdate` off, H; 21.1 at 4 MB, H; 28.3 at 16 MB, H; `m9` `gin_insert_*`), one 60 s cell each at
different host loads, and two identical cells of round 3 differ by 28 % (14.5 item 3), more than the
gap between 512 kB and 4 MB: the numbers do not choose 512 kB over 4 MB. The choice rests on the
documentation: readers scan the pending list too, and a larger list means a longer cleanup inside
an ordinary insert ([GIN fast update](https://www.postgresql.org/docs/18/gin.html#GIN-FAST-UPDATE));
512 kB is the small end that met. The storage parameter is in kilobytes ([CREATE INDEX](https://www.postgresql.org/docs/18/sql-createindex.html)).
R4-3 times fragment reads in 30 s windows alternating with and without 30 box writers (the windowed
method of 14.4); if the reads with writers miss the page bar, the fallback is `fastupdate = off`.

**Go bar of the slice that creates the copies and functions** (R1b's text part, after R4-3, with R7's
`put_line_text`; `db` tests of R1b and R7; nothing merges until every item passes, and C-1 repeats
items 1, 2 and 4 on the managed instance). Second review, M7, decision 1c.

1. **Two-project matrix, generated from the catalogue** (the test lists the functions of schema `txt`
   from `pg_proc`; a function without a matrix entry fails the test). Projects A and B hold the same
   texts, different texts, and a text only B holds (three fixtures, items of both projects). Bound to
   A, each read function is called with every argument shape: exact (a whole stored text), fragments
   of 3, 4 and 5 characters, a fuzzy key one character off at thresholds 0.1, 0.6 and 1, one word,
   several words, a word prefix, a one-character word, the empty string, `NULL`, wildcard-only (`%`,
   `_`, `%%%`, `\`), a space inside, 64 and 65 and 10,000 characters, `max_items` 0, 1, 10,001 and
   10,002. Assert: the result holds no key of an item of B (read from `items` by the test as owner),
   holds every key of A that the owner-run `LIKE` finds, and an invalid shape raises
   `invalid_parameter_value` and returns nothing. The mirror run is bound to B. The per-item tests,
   called with a key of B under A, return false; `put_*` called with B's keys under A change no row of
   B's copy (compared as owner before and after, `NULL` text included).
2. **No project, no scope, other scopes.** With no project bound, an empty project id, scope `core` and
   scope `deployment`, every function raises (`project_not_bound` / `project_scope_required`) before
   any row is read. A deployment-scope caller has no text search (stated above); that is the pass
   condition, not a gap.
3. **Mutation run.** The test builds, in a scratch schema, one variant of each function per
   statement with `v_project` removed from that statement, and asserts that item 1 fails for it
   (so item 1 would have caught the mistake: the test has been seen to fail) and that the route
   probe on that variant raises `text_isolation_fault`.
4. **Catalogue.** `proacl` of every function lists no `PUBLIC` entry and only `op_app` and the owner;
   `prosecdef` is true, `proowner` is `op_text_owner`, `proconfig` holds `search_path`, `plan_cache_mode`
   (reads) and `gin_fuzzy_search_limit` (set reads); the `prosrc` of every function equals the file's
   (one source); `op_app` has `USAGE` on `txt` and no privilege of any kind on either copy (`SELECT`
   from it raises `permission denied`) and cannot alter or drop an object of `txt` or create one in
   it; `op_app` is a member of no role; `pg_roles` shows `op_text_owner` without login and without
   `rolbypassrls`; every object of schema `txt` is owned by `op_text_owner` and `op_text_owner` owns
   nothing outside `txt` (a forgotten `RESET ROLE`); no view, rule, trigger or policy outside the
   copies references `txt`; the copies are the only tables with `relrowsecurity` and not
   `relforcerowsecurity` (asserted by name and by schema) and every other table with a `project_id`
   is forced (02a 1.4); `op_text_owner`'s own privileges are exactly those listed above.
5. **Results equal the source.** Each function's result equals the `LIKE` (or prefix) test over the
   source rows run by the owner; the 6.8 property test asserts after every step that each copy
   equals its source, `project_id` included (a row under another project's id fails it; the copy's
   foreign key is to `items (item_key)` only, a composite key would need a `UNIQUE (item_key,
   project_id)` on a 10 M-row table and the join of layer 1 already makes a mismatch harmless to
   isolation).
6. **Writers and erasure.** Two writers changing different boxes of one item leave the copy equal to
   its source; erasing a seeded photo leaves no row in either copy, no hit from `fragment_items` or
   `word_items`, after a vacuum (also in the erasure tests of 15a section 8).
7. **Round 4's set-up compiled and ran** every body of this block (R4-0) before the file was written.

## 9. Box rows: the index and the page

Today the page sorts items by their newest box `detected_at` (`routers/curation/regions.py:137-149`)
and `inner_hits` return only the boxes that matched, text included. The spike's semi-join walk:
194.7 / 593 deep, exact total 600 / 2496 (`s8_*`).

Design: items **newest first**, one ordered index per box filter shape, since only equality on every
column before the sort column bounds an ordered range (multicolumn indexes page): X6 `(project_id,
state, detector, item_key DESC)`, X26 `(project_id, state, item_key DESC)`, X27 `(project_id,
detector, item_key DESC)` (M6; box rows change far less often than item rows). `<box predicate>` is
state, detector and the text match of section 8. For `exact`, `fuzzy` and `same_text_as` it is the
same text in both statements. For a fragment (revision 3, review M3) the first statement takes
the narrow semi-join on the probe's keys or the walk with `txt.box_text_has` (broad), and the second statement takes `b.text_fold LIKE :like` on the rows
already bounded by `b.item_key = ANY(:keys)`, so only the boxes that match are returned (section 8).

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
filter takes the keys of section 8's steps 1 to 3 and orders their items; a filter
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
30 ms per card. Cause (inferred: `m10` holds no plan of the P13 cells, only of `p7_cards_r8`; R4-9
records plans): the statement runs once per card over that card's members; the 200 largest
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
Risk stated (review m3): a capped count through X32 still probes `item_state` for up to 10,001
candidates (the rest of the filter and the policy); round 3's text count plan shows 0.05 to 0.07 ms
per such probe cold (`m6a` `p4_text_gin_r1`, `item_state_pkey`), which would be 500 to 700 ms for
10,001 probes if they stay cold, so (2) may well be needed; R4-4 shows which.

**Text counts and text pages: fragment, fuzzy and variant matches.** Screens: the regions review tab
and every list with a plate-text filter. Numbers: text counts 1,824.6 / 7,140.4 ms (`EXISTS` form)
and 1,304.5 / 2,417.2 (trigram form), 100 % cancelled at every budget (`m6a`); fragment pages 410 to
1,593 ms p50, fuzzy 4,651 to 5,291, variant lists 64 to 170 (`m9`). Cause: row security, as
documented ("an index scan cannot be selected ... if ... its underlying function is not marked
`LEAKPROOF`", rules and privileges page, section 8) and consistent with the plans: the plan of the
fragment count (`p4_text_gin_r1`) is a sequential scan with the policy and the `LIKE` in one
filter. Not yet shown by a control (a plan of the same count without row security), and the
equality form's index use (3.7 / 6.3 ms) is inferred from timing: R4-3 records both. The `m9`
cells recorded no plan; their shape agrees: a 3-character fragment (many matches, the scan stops
early) is faster than a 4-character one (410 vs 1,593 ms), and fuzzy costs a whole scan. Options
(revision 3, section 8's table, with the binding constraint that the backend must also run on
managed PostgreSQL, where no role can bypass row security and no function can be marked
leakproof): (a) a definer function owned by a `BYPASSRLS` role (revision 2's choice; impossible on
managed); (b) mark the operators leakproof (superuser only, and `LIKE` is not leakproof by the
definition); (c) a posting table probed by equality (works everywhere, about 1.3 GB at 10 M against
X7's 108 MB, hand-made trigram search); (d) an owner-read search copy read through project-bound
definer functions owned by a role without any special attribute. **Chosen: (d)** with X33 for
equality and the variant list for fuzzy over a defined alphabet (section 8: `txt.item_box_text`,
`txt.item_line_text` and seven functions: `fragment_items`, `word_items`, `fuzzy_items`, two per-item tests, two writers); (c) is the
fallback if (d) misses R4-3 or its go bar. Round 4 R4-3: the leakproof flags and a control plan; the functions'
inner plans; every form inside P1, P4, P8 and P9 at three selectivities against 50 / 200 ms; reads
while 30 box writers fill the pending list.

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
`item_state`'s copies, the path is decided by the exact total, the page's capped count, the
planner's estimate or the sort path's own guard (section 7, cheapest first), `T_sort` from R4-5,
and the remaining band goes to the owner (question 2). Revision 3 (review M5): just above `T_sort`
the person waits for the count, the guard and the walk together, estimated up to about 190 ms for
the joined sorts (section 7). Round 4 R4-5: real class filters of 10,000 to 100,000 matches on the
copies, the composed "guard then walk" path, the planner's estimates against the actual counts; the
6 missing cells.

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
90 with fresh rows; if region HOT falls under 90 % there, the fillfactor is 80. Revision 3 (review
m5): every `m5` cell updated about 45,000 scattered rows of 10.6 M, so pages never ran short of room;
the bar and fillfactor 90 hold for sparse updates. A region re-run of a whole detector writes many
versions per page in batches, where 10 % free space is used up before pruning frees it; R4-7 adds
that dense cell, and if its region HOT share falls under 90 % the fillfactor question goes to the
coordinator with both numbers (80 costs 0.41 GB more heap at 10 M, `m3_sizes_10m.json`).

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
runs as a job (job kind `label_op`, `src/config/job_kinds/families_a.py:46`, in the exclusive group
`label_op` at `:53`, registered in `src/tasks/jobs_a.py:96`; checked in revision 3, review m8).

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
   (c) drop the chip, its sort and its filter; (d) keep today's meaning but count it per probe batch:
   after each batch the probe job (and the vector writer) recomputes the `all` count of the
   clusters its batch touched and writes it to the card, with no item lock and nothing added to a
   label, so the chip can lag the list by one batch; (e) show (b)'s count on the card and the exact
   `all` count when the cluster is opened (one P13 for one cluster, about 30 ms, `m10`).
   Recommendation: (a), because the chip is how a person
   picks the next cluster and a count that disagrees with the list it opens erodes trust; the cost
   lands on the probe job, not on the person.
2. **Sorts in the band at 10 M items.** For a filter matching from `T_sort` (about 40,000) to about
   100,000 items, seven sorts (creation time, crop area, rank in photo, detector confidence (the
   to-label tab's default), blur, detector class, cluster score) are estimated at up to about 190 ms
   for the first page once the count and the guard are added to the walk (section 7; bar 50 ms);
   sorts with no walk index (validated, class name, region score (the regions tab's default), the
   audit order, detector disagreement) take 90 to 160 ms at 100,000 matches and about 1 s at 1 M.
   Options: (a) accept a slower first page for these, up to 200 ms p50 and 400 ms p99, deeper pages
   at the normal bar; (b) grey out such a sort when the filter is that large, with the reason; (c) an
   index per sort (each costs every label or region write an index entry); (d) sort within a
   bounded window: the first 10,000 matches in the filter's own index order, sorted, with a notice
   "sorted among the first 10,000"; (e) a precomputed order per saved filter, kept by a job and
   shown with its time. These numbers are estimates: the question is put to you with R4-5's
   measured first-page times per sort, so it cannot be decided before round 4. Recommendation: (a) up
   to 100,000 matches, (b) above it, except the two tab defaults, which round 4 R4-5 times at real
   queue sizes before you decide.
3. **The cluster grid under a broad item filter** (for example "largest object per photo" over the
   whole project, matching millions). One aggregation over those members takes seconds at 10 M (gate
   A's unfiltered form 4.0 s; today's search engine 1.8 s at one client, `os10 s3_all`). Options:
   (a) accept it with a loading state the person can cancel; (b) make the subject scope (largest,
   two largest per photo) a key of the cluster summaries, so that control is exact and instant
   (cluster summary rows up to three times as many; a rank change emits cluster deltas); (c) refuse
   item filters on the grid above a size and ask to narrow; (d) a cached grid per filter, refreshed
   by a background job and shown with its "as of" time, instant once computed and stale until the
   next refresh. Recommendation: (a) now, (b) if the subject scope is used daily.

### 14.4 Bars given a checkable definition

| Bar (section 13) | Problem in round 3 | Definition from revision 2 |
|---|---|---|
| M1 "label p99 at 30 writers at most 2x its 1-writer p99" | closed loop measures CPU queueing | paced cells at 15 labels per second per writer: response p99 at 30 writers at most 2x the 1-writer paced response p99, cell load under 16 at start and end. Met: 1.08 (`m1`) |
| M1 "ingest writers scale (plan 13 ingest table not below)" | no factor; the table does not exist | storage never limits ingest: at 8 writers at least 1,000 photos per second committed (10x the 100 photos per second target of [13_backend_speed_plan.md](13_backend_speed_plan.md)), at 30 writers not below the 8-writer rate; warm cells of at least 60 s and 200 transactions. Round 3's small cells gave 1,125 and 1,472 (recomputed from `wall_s`): R4-10 confirms |
| M3, M4 "writer p99 within 10 % of M1" | two identical control cells differ by 28 % | inside one cell, 30 s windows alternating with and without the compactor (or rebuild): the median window **response** p99 (from the scheduled start, what the person waits for; about 89 to 114 ms in round 3, so +10 ms is about 10 %) with it at most the larger of 1.10x and +10 ms of the median without. M4 met even against M1 (+3.3 %) |
| chunks | 14.2 | single-label paced response p99 with `K` chunk jobs at most 2x without; throughput reported |
| gate A "reads while 1,000 objects per second are relabelled" (plan 15 section 6) | not run: readers and writers never shared a cell | every page and dashboard shape meets its bar while 30 paced writers label 450 items per second (single labels, half of them on items with a serving cluster membership, so cluster deltas and card refreshes happen) plus one chunk job (about 500 items per second), 10 paced box writers (5 transactions per second of 10 boxes with text each), 10 paced ingest writers (0.3 groups of 50 photos per second each) and the compactor run; the writers' paced bar holds at the same time. Engine CPU is recorded, not a pass condition; a miss with the engine at 95 % or more of its CPUs is marked "saturated" (a closed system measures queueing, 14.2). R4-1 (revision 3, review m9) |

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

Only what decides an open choice or closes a gap; revision 3 gives every choice this file leaves
"to measurement" its cell (list at the end). Same engine, database `v2_10m`, schema `m3` as round 3
left it (after the M6 drain and its restore, 0 mismatches), same runner. Each item is a step of the
chain under its own exclusive hold, load below 16 at start, as round 3:

```bash
cd artifacts_local/speed_1113/m3_storage
flock -o -w 21600 <the lock path chain.sh sets> ./hold.sh r4_<n> 585 m3_run.py r4_<n> 10m --budget 500
```

**Harness changes** (`m3_run.py`, `m3_readers.py`, `m3_sql.py`, `m3_build.sql`, before the window;
dry-run each on `v2_1m`, which is not part of the window's time): register `r4_0` to `r4_10` in
`ITEMS`; pass `explains=` for every card, text, count and routed-sort form so each cell records
`EXPLAIN (ANALYZE, BUFFERS)` of the outer statement; **nested plans**: a second login role
`m3_app_explain` (same grants as `m3_app`) on which the superuser sets `session_preload_libraries =
'auto_explain'`, `auto_explain.log_min_duration = 0`, `log_analyze`, `log_buffers`,
`log_nested_statements` on and `log_format = json` (these settings are superuser-only,
[auto_explain](https://www.postgresql.org/docs/18/auto-explain.html)); a few calls of each function
form run on that role and the harness copies the logged plans from the server log into the cell
record; timed cells stay on `m3_app`, so the logging does not slow them. A reader-and-writer cell
(`rwcell`: writers and readers forked together, both recorded, the readers tagged by window); a
windowed writer cell that starts and stops the compactor, or the box writers, every 30 s; a sort
generator over real class filters of known size (classes picked from `sum_items` by count, so the
matched rows are scattered as a real filter's) in place of `item_key BETWEEN` and `item_key % m`;
a backlog generator from real label transactions over at least 5,000 clusters (not +1 / -1 pairs on
200 keys); `pg_stat_checkpointer` in the writer records. **Sample rule**: a p99 is reported only
for a cell with at least 100 finished queries; a cell with fewer reports p50 and the maximum with
its count `n` (round 3 reported p99s of 6 to 60 samples, which are maxima). Delete the 3 stale M7
records first (`results_round3.md` section 3).

**Text set-up** (R4-0, for R4-3): role `m3_text_owner`, schema `txt`, `btree_gin`, `m3.op_bound_project_strict`
and the tightened `m3.op_scope_project` in `v2_10m`, and **the seven functions created from section
8's DDL block as written** (the harness extracts the block unedited; each function is then called
once with a valid and once with an invalid argument, and a body that does not compile or run stops
the round before R4-3: second review, M6 and decision 6; the slices copy only a tested text).
`txt.item_box_text` is filled from `m3.box_state` by one `INSERT ... SELECT ... string_agg ... GROUP
BY item_key`, then X7 (project first) is built on it; a clone `txt.item_box_text_plain` with the
index on `folds` alone is built for the comparison and dropped at the end. **Two projects of very
different size**: the spike's project L (10 M items) and a planted project S of 20,000 items (base
rows in `items` and `box_state`, texts drawn from the same distribution so that the same fragments
match in both), with its copy rows; both are removed at the end. The spike's texts are uniform
over A to Z and 0 to 9, so a 3-character fragment matches about 2.65 M x 4.5 / 46,656 = about 260
boxes and no fragment is broad: R4-3 plants two groups absent from the data, one in the copy rows
and `text_fold` of 3,000 random items (mid) and one in 50,000 (broad) of L, and the same two
fragments in 200 and 2,000 items of S (shared hold, under a minute). The spike has no item text
lines, so `word_items` is timed by the first wave sweep after its slice on set P (section 1.3);
R4-3 times the same mechanism on box text and calls every other function of the block at least once.

| Item | Question | Bar | Data and set-up | Cells | Time |
|---|---|---|---|---|---|
| R4-0 set-up (shared hold) | | | `sum_cluster_card` filled by the roll-up of 6.5; `ix_sum_cluster_det`, `ix_sum_deltas_cluster`; X33 on `box_state`; the text set-up above (the bulk fill time is a floor for the restore refill of 02c section 6, which goes through `put_box_text`; R4-3 times that path too); `m3_app_explain`; `ANALYZE` | none | 9 min (round 3: kinds 7 to 10 rebuild 67 s idle; X33, the copy with X7 and the comparison index about 1 to 1.5 min each, estimated; project S and the body check about 1 min) |
| R4-3 text | is row security the cause; do X33, the variant list and the owner-read functions meet the page bar inside every shape that uses them, also while boxes are written; what plan runs inside the functions | 50 / 200 ms per shape; recall 100 %; box write service p99 not above round 3's 512 kB cell (17.3 ms) by more than the windowed rule of 14.4; **every recorded nested plan shows constants** (a `$1` in a filter or index condition is a generic plan: the round fails the cell); **the project-led index is used** (its `Index Cond` holds `project_id = <constant>` beside the `LIKE` or `<%` condition) and a call bound to S reads fewer index buffers than L's for the same fragment; the matrix test (go bar items 1 to 3) passes on the set-up | `proleakproof` of `texteq`, `textlike`, `int4eq`, `int8eq`, `uuid_eq`, `textregexeq`, `similarity`, `similarity_op`, `word_similarity_op` and the operator families of `gin_trgm_ops` and the B-tree `text_ops` (the SQL behind `\dAo+`, rules and privileges page); control: the fragment count of round 3 as the superuser (no row security) and as `m3_app`; inner plans of `fragment_items`, `fuzzy_items`, `box_text_has` and the probe for every form below (nested statements, constants visible, `Rows Removed by Filter` on `project_id`); **two-project cells**: the same fragment at narrow, mid and broad bound to S and bound to L, on the project-led index and on `item_box_text_plain` (the cost of a small project beside a large one, and whether the index earns its size); the matrix of the go bar run once on the set-up (seconds); `fragment_items` alone at 3, 4 and 5 characters (narrow), mid and broad; the text predicate inside P1 (class page by `updated_at`), P4 (count), P8 (review by score) and P9 (both statements), each narrow, mid and broad (section 8 steps 1 to 4, with another filter beside the text predicate in half of the cells, so the total after every filter is timed), P1 and P9 also deep at broad (the walk with `box_text_has`); exact first and deep; variant list at 1, 2, 4, 7, 10, 12 characters; `fuzzy_items` at 3, 7, 10, 12 characters with recall against a brute-force distance over every distinct text, and how often 10,001 candidates come back; fragment and exact readers in 6 x 30 s windows alternating with and without 30 paced box writers (512 kB list), the writers also calling `put_box_text` as `store_states` does | 31 reader cells x 20 s at 8 readers, 6 two-project cells x 20 s, the windowed cell 180 s, plans and flags without timing | 22 min |
| R4-7 fillfactor | does fillfactor 90 keep region-only updates at least 90 % HOT beside labels, and in a dense region re-run? | 90 % HOT | `item_state_ff90`: (1) 15 label + 15 region writers paced, labels on fresh rows, 300 s; (2) dense: 10 writers re-running the region columns of one detector's items in `item_key` order, batches of 1,000, 300 s; checkpoint counters | 2 cells | 10 min, then restore and consistency check 4 min (round 3: rebuild 163 s, `m3_m5_restore_10m.json`; check 79 s) |
| R4-4 detector count | does the plan's form (EXISTS through X32) meet the page bar? | 50 / 200 ms | P4 cap 10,001 with `detector_class_name` as `EXISTS` on `detections`, alone and with a class; 1 and 8 readers; plans | 4 x 20 s | 2 min |
| R4-5 sorts | where is `T_sort`; how slow is the composed path just above it; does the planner's estimate decide well enough to skip the guard (section 7 step 3); the 6 missing cells | 50 / 200 ms per cell; outputs `T_sort`, the band per sort, estimate against actual | the 6 cells of `photos_unlabeled_count` (round 3 section 3); sort path on `item_state` copies for `created_at`, `crop_area_norm`, `crop_rank_in_image`, `detector_confidence`, `detector_class_name` at 10,000 / 20,000 / 40,000 / 70,000 / 100,000 matches; walk path (planner settings unforced) at 40,000 / 70,000 / 100,000 for those plus `updated_at`, `blur_lap_ratio`, `cluster_score`; sort path for `validated`, `class_name`, `region_score` at 20,000 / 40,000 / 70,000; the to-label and regions queues at the sizes the data holds; first and deep, 8 readers. Then, with `T_sort` computed by the script (the largest size where every copy sort meets), **guard then walk** (the page's capped count, the guard to `T_sort + 1`, the walk) for `updated_at` and the seven joined sorts at `T_sort` + 1 %, + 10 % and 2 x, first page; for every filter the `EXPLAIN` row estimate beside the actual count | about 111 cells x 3 s plus 24 composed cells x 5 s plus 6 | 2 holds, 13 min |
| R4-6 review by score | does the routed P8 meet the page bar in every confidence band? | 50 / 200 ms | `pl2`'s predicate, 4 bands, routed by the counted tab's total else the steps of section 7; plans per band | 8 x 20 s | 3.5 min |
| R4-9 filtered cards | up to which matched size does the set-based P13 meet the dashboard bar? | 200 / 800 ms | filters matching about 10,000 and 100,000 items at 8 readers (30 s each, at least 100 queries); about 1 M items and the subject scope "largest object per photo" over the project at 1 reader, 10 queries each (p50 and maximum with `n`, sample rule); plans of every form (the P13 cause is inferred until then, 14.2) | 4 cells | 4 min |
| R4-2 cluster cards | does the card row form meet the dashboard bar; does the compactor keep its rate with the card refresh; does the card check find and fix a wrong or missing card | P7 200 / 800 ms at 10,000 pending; compactor at least 26,382 rows/s; 0 mismatches after; one rewrite per injected card | seed a 1 M-row backlog from real label transactions over at least 5,000 clusters (about 168,000 transactions at round 3's 1,133 per second, 2.5 min), compact to each size; P7 (full join) with 2,000 cards, sort drawn per query from `size_desc`, `needs_review_desc`, `dominant_class`, `updated_desc`, at 0 / 100 / 10,000 pending; detector form; compactor with refresh alone on the 1 M backlog and under 30 paced writers; the card check of 6.6 with one wrong and one missing card injected; consistency check; first cell repeated at the end (14.5 item 1) | 3 x 30 s at 8 readers, 3 x 20 s at 1 reader (p50 and maximum), detector 30 s, compactor 2 x 60 s, card check, repeat 30 s | 11 min |
| R4-1 reads under writers | do the page and dashboard bars hold while labels, boxes and ingest are written? | 14.4 last row | 8 readers cycling P1 class page (deep), P3 class total, P4 class and dates, P7 (2,000 cards), P8 routed, exact text, fragment (narrow); writers as 14.4's last row (30 paced label writers, half on items with a serving membership, one chunk writer, 10 box writers, 10 ingest writers, compactor 10,000 / 0.2 s with `ix_sum_deltas_cluster`); engine CPU recorded | 1 cell of 180 s; the same readers alone 60 s | 5 min |
| R4-8 chunks | how many `label_op` chunk jobs may run at once? | 14.4 chunks row | 8 paced single-label writers alone, then beside K = 1, 2, 4, 8 chunk writers | 5 x 60 s | 6 min |
| R4-10 defined bars | the windowed compactor bar and ingest throughput | 14.4 | 30 paced writers, 6 x 30 s windows alternating the compactor (response p99, `ix_sum_deltas_cluster` present); ingest at 1, 8, 30 writers, 20 s warm-up, 60 s cells | 1 + 3 cells | 7.5 min |
| final check (shared hold) | | 0 mismatches | consistency check of every `_now` view and every card | none | 1.5 min |

**Total and window.** About 98 minutes of steps (the "Time" column, from round 3's cell durations
and the estimates named there; revision 3.1 added 3 minutes to R4-0 and 3 to R4-3), about 111
minutes with the chain's start and stop around each item;
the round fits one quiet window of **two hours**, provided no other heavy work runs on the host:
the chain's load gate waits up to 4 minutes per cell at load 16 or more, and round 3 lost six holds
of M7 to loads of 21 to 31 (`results_round3.md` section 3). The dry-runs on `v2_1m` happen before
the window. Order (keeps the data clean and runs first what the first slice waits for): R4-0, R4-3,
R4-7 with its restore, then R4-4, R4-5, R4-6, R4-9, R4-2, R4-1, R4-8, R4-10, the final check. Every
cell writes `raw/m4_<item>_10m.json`; the write-up follows `results_round3.md`'s form.

**Must run before the first record slice (R1b) builds its waiting parts:** R4-0, R4-3, R4-7 (about
45 minutes of steps). **Can run later, each before the slice that waits for it (section 16):** R4-4,
R4-5, R4-1 (R4, gallery reads); R4-6, R4-8 (R5); R4-2, R4-9 (R8); R4-10 (changes only constants of
R1b's compactor). If the window is cut short, the chain stops after a finished item and resumes.

**Cloud check C-1** (outside the window; it needs a managed instance, so the coordinator schedules
it): on an Amazon RDS for PostgreSQL 18 instance (16 or later is needed for `GRANT ... WITH SET`,
[GRANT](https://www.postgresql.org/docs/18/sql-grant.html)), as the master user:

1. **Facts.** `SELECT version()`; `rolname, rolbypassrls` of `postgres` and `rds_superuser`;
   `CREATE ROLE x BYPASSRLS` fails (the evidence that (a) is impossible there); `CREATE EXTENSION`
   of `pg_trgm` and `btree_gin` works for the master user; `auto_explain` through the parameter
   group; `PGOPTIONS='-c op.scope=deployment'` accepted on the instance endpoint the backup uses.
2. **Bootstrap, `0011`, the tests.** The bootstrap of section 8 and 02a 1.5, `0011`'s text part as
   `op_owner`, and the go bar of section 8 (items 1 to 4: the matrix, the scopes, the mutation run,
   the catalogue) pass.
3. **Backup and restore, required** (02c section 6, the managed procedure): fill the instance with
   the 1 M-item spike data plus its copies; `db backup` ends without error and its per-table row
   counts equal the counts the owner reads with `FORCE` off; `db restore` into the same instance
   follows the numbered steps; record that `pg_restore` ends with no error (the dump's extension
   entries included), the order of `ROW SECURITY` entries against `TABLE DATA` in `pg_restore
   --list`, the time of each phase (data, indexes, constraints, refill, copy index build) and the
   extrapolation to 10 M items, that `db verify` passes (forced flags, `proacl`, ownership,
   `prosrc`), and that go bar items 1, 2 and 4 pass again after the restore. If the data load
   fails with "COPY FROM not supported with row-level security", the fallback of 02c section 6
   (data-only restore under `NO FORCE`, or `--rows-per-insert`) is run and timed instead.

Needed before R1b's bootstrap part and the restore command merge; the self-hosted path does not
wait for it, but its restore test (02c section 7) is the same sequence.

**Every choice left to measurement and its cell:** the text mechanism's confirmation, the project-led
index against `folds` alone, the inner plans and the `enable_seqscan` setting of the functions, the
per-call cost of `box_text_has` in the walk, the fuzzy threshold, the `fastupdate`
fallback (R4-3); `item_state` fillfactor 90 or 80 (R4-7); the detector-class count form (R4-4);
`T_sort`, the planner-estimate step, the band (R4-5); the routed review queue (R4-6); the filtered
card size limit (R4-9); the card form, the compactor rate with the refresh, the card check (R4-2);
the bars under writers and the write cost of `ix_sum_deltas_cluster` (R4-1, R4-2, R4-10); `K`
(R4-8); the windowed compactor bar and ingest (R4-10); managed set-up, restore time and the
dump-extension question (C-1).

## 16. What the Rust record slices take from this revision

Slice codes and names are those of plan 18b; "takes" lists the read shapes, indexes, summary tables,
settings and statements each builds from this file, with the section here. Revision 3 marks every
item **final** (build it now) or **waits** for the named cell; a slice builds its final items first
and a waiting item only after its cell, so nothing is built that round 4 may change. Owner questions
are marked the same way.

| Slice (18b) | Final (build now) | Waits for round 4 or the owner |
|---|---|---|
| R1b, the record and the fold | `0011`: X1 to X6, X8 to X21, X24, X26 to X33 (15a 2.6; X33 on `box_state`); the summary tables, `sum_deltas`, the `_now` views (6.1); `op_bound_project_strict` and the tightened `op_scope_project` (8); the bootstrap's `op_text_owner`, schema `txt`, `pg_trgm`, `btree_gin`, `op_owner`'s membership `WITH SET` and the default-privileges line (8; 02a 1.5) for self-hosted; `sum_deltas` vacuum threshold 10,000 (6.1, 6.5); `store_states` with aliased `RETURNING` pairs (6.2); the tally (6.2); the compactor for all ten kinds, key-only conflict targets for kinds 6 to 10, ordered upserts, batch 10,000, beat 0.2 s, built table-driven so R8 adds the card refresh as a second statement and the touched-set output without changing the first (6.5); the drift check over ten kinds with float sums in double precision (6.6); the label transaction's statements (6.9) | both text copies, X7 and X25 (project first), the seven `txt` functions, the SQL file `src/db/sql/txt_search.sql`, `store_states`' `put_box_text` call and the go bar (8): R4-3 (and the posting-table fallback if it misses R4-3 or the go bar); backup and restore on a managed service (02c section 6): C-1; `item_state` fillfactor 90 or 80: R4-7; the bootstrap on a managed service: C-1; batch and beat constants: R4-10 (constants only) |
| R4, gallery reads | P1 with up to 16 class branches, P2, P4 with a 50 ms budget, P5, P6, P15 (2, 5); the cursor with typed binding and the chosen path (4); sorts reading `item_state`'s copies; the path decision's steps 1, 2 and 4 (7); capped totals (5) | `T_sort` and step 3 (planner estimate): R4-5; the band: owner question 2, asked with R4-5's numbers; the detector-class count form: R4-4; the text predicate inside P1, P4, P8 (8 steps 1 to 4, totals after every filter): R4-3; reads under writers: R4-1 (a miss goes back as code work) |
| R5, labelling, review, class counts | counted review tabs (6.2); P3 totals; class counts from `sum_items` (row 16); chunks of 1,000 under the job kind `label_op` (14.2) | the routed review queue: R4-6; `K` above 1 (today's exclusive group): R4-8; the "to review" definition: owner question 1 |
| R7, regions and boxes | P9 with X6, X26, X27 (9, met); `exact` and `same_text_as` on X33; the variant list and its alphabet function, the switch rule (8); the 3-character minimum for fragments; the dropped `max_expansions` pins (8) | P9 with fragments and `fuzzy_items`' threshold: R4-3; the pending-list setting under writers: R4-3; the `text_lines` copy writer (`src/db/repos/text_search.py`, calling `put_line_text`), `line_text_has` and word search: R4-3 for the mechanism, then the wave sweep on set P (no item text in the spike) |
| R8, clustering and the cluster screens | `0016`: the four cluster summary tables, `ix_sum_cluster_det`, `ix_sum_deltas_cluster` (moved here from R1b, revision 3), `sum_cluster_card` (6.1); the compactor's kinds 7 to 10, the touched-set output and the card refresh (6.5); the card check (6.6); the partition fill with the card roll-up (6.7); P8 inside a cluster, facets and placement reads (met) | P7 (full join) against its bar and the compactor's rate with the refresh: R4-2; filtered cards (P13 set-based): R4-9 and owner question 3; the "to review" count: owner question 1 |
| R10, machine helpers | nothing yet | only if the owner answers question 1 with (a) or (d): the probe job and the embedding re-index report membership changes (through `store_states` for (a), per probe batch for (d)) |
| R6a, R6b, R9, R2b | nothing new beyond revision 1 (6.7's run-keyed fill for R6b); R6b sets no GIN setting (18b correction below) | none |

**Correction for 18b** (on `int/native`, not edited here; for the coordinator): (1) section 3,
row R6b, column "Revision": "`0015_near_dup_cutover` (writes 15b M9's `fastupdate` choice)" becomes
"`0015_near_dup_cutover`"; (2) section R1b, **Builds**: "indexes X1 to X21 and X24 to X32" becomes
"indexes X1 to X6, X8 to X21, X24 and X26 to X33, the schema `txt` search copies with X7 and X25
(project first, `fastupdate = on, gin_pending_list_limit = 512`) and their seven functions, `op_bound_project_strict`
and the tightened `op_scope_project` (15b section 8)", and "fillfactors from 15b M5" stays; (3) row
R8: `0016` also creates `ix_sum_deltas_cluster` (15b 6.1).

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

**Review of revision 2** (2026-11-16, 0 blocker, 7 major, 13 minor), answered by revision 3:

| Finding | Answered in |
|---|---|
| M1 the `BYPASSRLS` role cannot be created or made owner as `0011` said | section 8 (mechanism (a) not chosen: impossible on managed PostgreSQL; (d) needs a role with no attribute, created by the bootstrap, objects created under `SET ROLE`); 02a 1.5 (role table, self-hosted and managed, tests); 02c section 6 (backup and restore order); section 15 C-1 |
| M2 the fragment function a planner fence with an unseen plan, no cap, `ROWS 1000`, parallel unsafe | section 8 ("the plan inside, and its bound": per-call custom plan, `max_items`, one row per item, standalone statement); R4-3 inner plans through `auto_explain` nested statements |
| M3 text never timed inside its shapes; P9's second statement for fragments | section 8 steps 1 to 3; section 9; R4-3 cells inside P1, P4, P8, P9 at three selectivities |
| M4 the variant list's alphabet; superset of today's 50 expansions | section 8 "Fuzzy" (alphabet from `text_charset` and the fold, bounds 40 and 12, switch to `fuzzy_items`) and "A named product change" (the two pins dropped) |
| M5 the guard paid again by the walk; no composed cell; copies not timed | section 7 (decision steps, the summed band); 14.2 sort entry; owner question 2; R4-5 composed cells and estimates |
| M6 card consistency: check statement, serialization, touched set, pending-only clusters | 6.6 card check (read committed, 2a's lock); 6.5 `touched` output; P7 full join; R4-2 |
| M7 strict scope function, word function body, partial index predicate, short words, `ix_sum_deltas_cluster`'s migration | section 8 (DDL of `op_bound_project_strict`, `op_scope_project` tightened, `word_items` body, copies without a partial predicate, short-word rule); 6.1 and section 16 (`ix_sum_deltas_cluster` in `0016`) |
| m1 cause cited from the weaker pages; no control; operator list | section 8 (rules and privileges sentence, "consistent, not yet controlled"); 14.2 text entry; R4-3 flags of nine functions, operator families, control plan |
| m2 card cost breakdown verified | stands: verified, nothing to change |
| m3 detector stand-in, probe cost | 14.2 detector entry (500 to 700 ms risk, fallback (2)); R4-4 |
| m4 two causes without plans | 14.2 (P13 marked inferred; `pl2` already marked); R4-6, R4-9 record plans |
| m5 fillfactor dense regime; stale rationale | 14.2 label entry; R4-7 dense cell; 15a fillfactor row |
| m6 GIN numbers do not choose 512 kB over 4 MB | section 8 "GIN pending list" (the choice rests on the documentation; windowed cell; fallback `fastupdate = off`) |
| m7 X33 for every equality form | stands: X7 leaves `box_state` (section 8), so exact and same-text need X33 too |
| m8 windowed bar service or response; `label_op` today | 14.4 (response p99); 14.2 chunks entry (job kind checked in the code) |
| m9 writers in R4-1; CPU; six-sample p99 | 14.4 last row; R4-1; section 15 sample rule; R4-9 |
| m10 run time without seeding, restore, load gate | section 15 "Total and window" (92 minutes of steps, two-hour window, groups) |
| m11 owner questions' options | 14.3 (options (d), (e) added; question 2 put with R4-5's numbers) |
| m12 stale text | header status; section 1.2 note; 6.9; 15a fillfactor row |
| m13 `ix_sum_deltas_cluster` write cost on one cell | 6.1 note; R4-1, R4-2, R4-10 run with it |


**Review of revision 3** (2026-11-16, 1 blocker, 7 major, 8 minor), answered by revision 3.1. The
review's verdict on isolation (a missing project clause in one body exposes every project, with no
second barrier) is answered by the layers of section 8 "How isolation holds":

| Finding | Answered in |
|---|---|
| B1 managed backup and restore cannot work | 02c section 6: the dump of a managed service runs with `--enable-row-security` under the deployment scope; the restore does **not** use that option: `pg_restore` creates the tables itself, `FORCE` is part of the table's creation and `ENABLE ROW LEVEL SECURITY` is a post-data entry, so the data loads while row security is not yet on ([pg_restore](https://www.postgresql.org/docs/18/app-pgrestore.html): the option "is relevant only when restoring the contents of a table which has row security"); fallbacks if C-1 shows otherwise; C-1 step 3 runs it. The review read the option's note, which concerns a data-only restore into existing tables |
| M1 `USAGE` on `txt` never granted | section 8 DDL (`GRANT USAGE ON SCHEMA txt TO op_app`, `EXECUTE` on seven functions, nothing on tables); 02a 1.5; go bar item 4 (privileges asserted exactly) |
| M2 restore order: `--clean`, `--no-privileges`, refill role, recreated functions | 02c section 6 numbered procedure: `DROP SCHEMA txt CASCADE` before `pg_restore`; the grants named and re-applied; the refill is `op_app` calling `put_box_text` per project; functions created through the one file in one transaction, default privileges revoke `PUBLIC` |
| M3 erasure catalogue and test miss the copies | section 8 "Erasure"; 15a section 8 (catalogue rows, test scans `public` and `txt`, leaves-nothing test extended); go bar item 6 |
| M4 totals wrong when a text filter combines with another | section 8 "How a page ... use it": narrow total by `count(*) OVER ()` after every filter in the shape's statement, broad total by the capped count over the whole predicate; R4-3 times cells with another filter |
| M5 round 4 does not prove the plan | section 15: two projects of very different size, constants in nested plans, project-led index used, plain index comparison, per-call cost of the per-item test, matrix smoke run |
| M6 ambiguous `words` in `word_items` | section 8: `q_` and `v_` prefixes, every reference qualified, `#variable_conflict error`; every body compiled and run in R4-0 before any slice copies it |
| M7 isolation hangs on one clause per statement | section 8 "How isolation holds": keys only joined to a forced table in the caller's statement, the probe's `text_isolation_fault`, project foreign key and project-led index, the go bar (matrix, mutation run, catalogue); one SQL file; residual exposure stated; posting table kept as the costed fallback |
| m1 `gin_fuzzy_search_limit` not pinned | section 8 DDL (`SET gin_fuzzy_search_limit = 0` on the set reads; `proconfig` test) |
| m2 foreign key to `items (item_key)` only | stands: a composite key needs `UNIQUE (item_key, project_id)` on a 10 M-row table, and the join of layer 1 makes a mismatched row harmless to isolation; the 6.8 property test compares `project_id` (go bar item 5) |
| m3 three steps not one snapshot | section 8: `REPEATABLE READ READ ONLY` for text reads |
| m4 (a) P9's second statement for trigram-fuzzy; (b) refill time; (c) policy drop and create | (a) section 8 after step 4; (b) 02c section 6: indexes built after the refill; R4-0 times the `put_box_text` path; (c) stands: between drop and create the table denies `op_app` (fail closed), and the step runs before the API starts |
| m5 missing tests | go bar items 4 and 6 (`proacl`, converse ownership, no view or trigger, `db verify` checks, concurrency test) |
| m6 managed assumptions in C-1 | section 15 C-1 steps 1 to 3 |
| m7 stale text (02a 238, `relrowsecurity` by schema) | 02a 1.5 role table; go bar item 4 asserts by name and by schema |
| m8 scope refusal not stated | section 8 "Scopes" |
