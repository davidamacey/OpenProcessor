# P2a-2: People (phase spec)

| Field | Value |
|---|---|
| Status | plan, 2026-11-13. Starts after the P2a-1 exit, P1's N1 and N7, and the owner's answer to Q3 (face model licence) ([90](90_owner_questions.md)). Faces are on by default in library projects (owner decision 2026-11-13) |
| Master plan | [00_photo_library_plan.md](00_photo_library_plan.md): section 5.3 (one face-identity service), section 2.3 (the `faces` role), section 11.5 (eval sets) |
| Design source | [cluster plan](../cluster_search_makemodel_plan.md) section 3.2 (prototypes, review band, cannot-links, split, quality gate, abstain) |
| Scope rule | only what P2b depends on, plus naming and Send to project; merge suggestions, the unassigned-faces browser, "me" and birth dates are later (00 section 2.2) |

Notation: `{L}` = `/curation/projects/{project}/library`.

---

## 1. Scope and non-goals

**In scope.** One face-identity service for core `/persons` and library People (the union-find grouping
deleted); batch grouping and incremental assignment on the library's `faces` index; the review band;
naming, hiding a person from People, cover face; confirm, reject ("not this person"), merge, split;
**forget this person** (delete their embeddings and derived data); person fields in rules, search, suggest
and facets; face thumbnails; Send to project; the People UI; the faces eval kit.

**Non-goals.** Exclusion rules (P2b); merge suggestions, unassigned-faces browser, "me", birth dates (later);
pets as identities.

---

## 2. The one face-identity service (lane L-P1)

| Concern | Rule |
|---|---|
| Code | `src/services/face_identity/` is the only identity code, parameterised by index name and bound scope; library People (`src/services/library/people/`) and the core `/persons` routes (`src/routers/persons.py`) both call it. The union-find grouping (`method='union_find'` in `person_partition.py`, `similarity_components` in `person_grouping.py`) is deleted in the same change |
| Batch method | mutual-kNN graph (k 30) → average-linkage AHC through `src/services/curation/clustering/methods/ahc.py` at a calibrated cut |
| Quality gate | `det_score ≥ 0.7`, `side_px ≥ 40`, `abs(yaw_deg) ≤ 60` for prototypes; gated faces may join but never form or bridge a person |
| Same-photo cannot-link | two faces of one photo are never one person |
| Thresholds | `t_accept`, `margin`, `t_review` from an isotonic fit on LFW pairs of the **training identities**; precision is measured on held-out identities (bar 1); values stored with each run and the face model id |
| Stable ids | Jaccard ≥ 0.5 with the previous membership keeps an id; named, merged and confirmed faces are anchors (must-link) |
| Incremental | the assignment tick (advisory `LIBRARY_PERSON_ASSIGN`) scores new faces against each person's up to 8 prototypes (best pair): accept above `t_accept` with margin; review band → `suggested` and `candidate_person_ids`; below → unassigned for the next batch run |
| Minimum person | 3 mutually linked faces of acceptable quality |
| Merge | writes must-links; queries expand merged ids at once (00 section 3.1); the drain normalises `faces.person_ids` |
| Core behaviour change | core `/persons` grouping results change with the method (better precision, no chaining); its route contracts stay; `CHANGELOG.md` says so |
| Privacy | `library.faces_enabled` off → people routes answer `400 faces_disabled`; purge of all faces uses `src/services/face_privacy.py` on the project's faces index with a dry-run count and the `privacy_purges` audit |

### 2.1 Forget this person

`POST {L}/people/{person_id}/forget` (dry run returns counts; the write needs `confirm_count`) runs the job
`library_person_forget` (`exclusive_group='library_people'`):

1. Delete every face doc whose `person_id` is the person, or is merged into it (embeddings included), and
   remove the person from every `candidate_person_ids`.
2. Delete the person's prototypes, constraints, any exclusion rules on the person (P2b), memories whose
   subject is the person (P2c), and cached face thumbnails.
3. Re-project the affected photos (`faces.person_ids`, `faces.count`).
4. Keep the `persons` row as `state = 'forgotten'` with no name, so stale references resolve.
5. Write a counts-only `privacy_purges` row.

What stays: the photos, their other faces and every non-face signal. New photos of that person are no longer
recognised (they form a new unnamed person if they recur); to keep catching them, use an exclusion rule
instead of forget. The UI states both.

---

## 3. Routes and wire models (`src/routers/curation/library_people.py`, `src/schemas/curation/library_people.py`)

| Route | Request | Response |
|---|---|---|
| `GET {L}/people` | `include_hidden`, `named`, `sort: Literal['photos','name','recent']`, `cursor` | `People {people: list[PersonCard {person_id, name, hidden, photo_count, cover_url}], next_cursor}` |
| `GET/PATCH {L}/people/{person_id}` | `{name?, hidden?, cover_face_id?, expected_revision}` | `Person {...PersonCard, face_count, suggested_count, state, revision}` |
| `GET {L}/people/{person_id}/photos` | `LibraryFilter`, `cursor` | `TimelinePhotos` |
| `GET {L}/people/{person_id}/faces` | `state`, `cursor` | `Faces {faces: list[FaceCard {face_id, image_id, thumbnail_url, score, state, quality}], next_cursor}` |
| `GET {L}/people/{person_id}/suggestions` | `cursor` | `Faces` (review band, highest score first) |
| `POST {L}/people/{person_id}/faces/confirm`, `/reject` | `{face_ids}` | `PeopleWriteResponse {n_faces, person_id, op_id}` |
| `POST {L}/people/merge` | `{person_ids, into_person_id, expected_revisions}` | `Person` |
| `POST {L}/people/{person_id}/split` | `{face_ids, new_name?}` | `{person, new_person}` |
| `POST {L}/people/{person_id}/forget` | `{dry_run, confirm_count?}` | `ForgetResponse {n_faces, n_photos, n_rules, n_memories, job_id}` |
| `GET {L}/faces/{face_id}/thumbnail` | `size` | bytes, from the grid or preview tier with padding |
| `POST {L}/people/grouping/runs` | `{mode: Literal['incremental','full']}` | `JobRef` |
| `GET {L}/people/grouping/status` | | `GroupingStatus {last_run_at, faces_total, faces_unassigned, thresholds, model}` |
| `POST {L}/send_to_project` | `{selection, target: {slug} \| {new: {slug, display_name}}, include_items: bool = true, dry_run}` | `SendToProjectResponse {n_photos, n_items, classes: list[{class_name, count}], n_left_out_by_rules, job_id}` (combine copy path with a selection source; P2b fills `n_left_out_by_rules`) |

P2a-1 models change: `LibraryFilter.person` (all of) and `person_any`; rule field `person`
(`contains_any`, `contains_all`, `contains_none`, `count_gte`); `PhotoDetail.people`; suggest kind `person`;
facet `person`. `make contracts` in the same commit.

---

## 4. Tables, jobs, UI

Tables: `persons`, `person_prototypes`, `face_constraints` (00 section 6.1), revision
`<next>_library_people.py`. Field module `library_face_fields.py` (`faces.person_ids`,
`candidate_person_ids`, `projected_at`). Jobs: `library_person_grouping` (`cluster` queue,
`exclusive_group='library_people'`, pausable, sliced), `library_person_forget`, the assignment tick,
`library_send_to_project` (`exclusive_group='combine_target'`, scope `target_slug`).

Cropwright: `lib/components/library/{PeopleGrid,PersonPage,FaceReview,ForgetPersonDialog,SendToProjectDialog}.svelte`,
routes under `routes/p/[project]/library/people/`, `lib/api_libraryPeople.ts`; review keys follow
`components/review/**`; faces drawn on request in `PhotoViewer.svelte` (P2a-1).

---

## 5. Lanes, order and verify commands

```
wave 1:  L-P1 one face-identity service (AHC, gate, constraints; union-find deleted) || L-PE faces eval kit
wave 2:  L-P2 library People routes, forget, person fields || L-P3 Send to project
wave 3:  L-PF People UI || L-FIX
then:    sweep (core /persons suites included), eval against the bars, owner walk
```

| Lane | Model, effort | Owns | Verify |
|---|---|---|---|
| L-P1 | Sonnet, high (identity, lock rule; independent review before merge) | `src/services/face_identity/**`, `src/domain/person_identity.py`, `src/routers/persons.py` (calls only) | `.venv/bin/python -m pytest tests/ -q --no-cov -k "person or face_identity"` |
| L-PE | Sonnet, medium | `scripts/datasets/fetch_commons_faces.py` (CC-licensed identities from Commons person categories, licence per file, local only; builds the set if it is absent), `scripts/bench/library_eval_people.py` (held-out identity split, registered bars, Clopper-Pearson bounds) | `.venv/bin/python -m pytest tests/scripts/test_people_eval.py -q --no-cov` |
| L-P2 | Sonnet, high | `src/services/library/people/**`, `library_people.py` routes and models, person fields in `src/domain/library_rules.py`, `search.py`, `facets.py` | `.venv/bin/python -m pytest tests/library/test_people.py tests/library/test_person_forget.py tests/library/test_person_assign.py -q --no-cov && .venv/bin/python scripts/codegen/generate_contracts.py --check` |
| L-P3 | Sonnet, high (cross-project copy) | `src/services/library/send_to_project.py`, `src/services/projects/combine/**` (selection source) | `.venv/bin/python -m pytest tests/library/test_send_to_project.py tests/projects -q --no-cov` |
| L-PF | Sonnet, high | People UI | `cd frontend && npm run check && npx vitest run src/lib/library` plus `e2e/stubbed/library/test_people.py`, `test_forget_person.py` |

---

## 6. Tests to write red first

| Test | Proves |
|---|---|
| `tests/services/test_face_identity_one_path.py` | no module other than `src/services/face_identity/` groups, assigns or merges faces (AST); `union_find` is gone |
| `tests/library/test_people.py` | same-photo faces never merge; gated faces never bridge; named persons keep ids and faces across a full run; split writes cannot-links a rerun respects; reject never re-adds; merged ids expand in queries at once |
| `tests/library/test_person_assign.py` | accept, margin, review and unassigned bands |
| `tests/library/test_person_forget.py` | after forget: zero face docs, prototypes, constraints and projections for the person (count queries), no name anywhere, a kNN with one of the deleted vectors returns none of them, `privacy_purges` has counts only |
| `tests/library/test_send_to_project.py` | classes by name; images by content key; dry-run counts equal the job's |
| core `/persons` tests (existing) | updated for the new method's documented results, not weakened |

---

## 7. Acceptance walk (owner, own library)

1. People fills after grouping; name five people; hide one from People.
2. Review suggestions for one person: bulk confirm, reject two; merge a split person; split a mixed one.
3. Full regroup: every correction holds.
4. A smart album "person A and person B, summer"; new photos of them appear.
5. Forget a test person: the dry-run counts, then zero; their photos remain without their face.
6. Send a person's best photos to a new workspace project; label there; the library is unchanged.
7. The eval report: precision on held-out identities, recall per face-side bin, with n and bounds.
