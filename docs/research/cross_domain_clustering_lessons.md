# Cross-domain clustering lessons: speaker linking applied to multi-view objects and faces

Status: research note and experiment plan (2026-10-28). Nothing here is implemented, and nothing
was measured for this note. It feeds the graph refinement epic (#346) and its children: #342
(Leiden prototype), #343 (constrained split/merge, stable ids), #344 (continuous refinement), #345
(GPU and experimental methods). It also touches #354 (person grouping), #372 (order-sensitive IVF)
and the auto-label loop (#286).

The question: how do we cluster objects that are one identity but look different from each view?
Examples are the same vehicle seen from the front, side and rear at different scales and light, or
the same face across poses. A sibling open-source speech-transcription project by the same owner
has solved a similar problem in production. It links speakers across recordings, where the "views"
are microphones, rooms and the speaker's state. This note covers four things:

- what that project does and what it learned
- what transfers to our three kinds of clustering ([cluster_types.md](cluster_types.md)), and what
  does not
- a concrete design for the refinement engine
- an ordered experiment plan

The sibling project is described in general terms. No code was copied, and no private names or
paths appear here.

## 1. Summary

1. **Single linkage is the main cause of false merges, in both domains.** The speech project's
   first cross-recording grouping was union-find at cosine 0.65 (connected components, which is
   single linkage). One bridging pair joined two different speakers. It was replaced by
   complete-linkage agglomerative clustering at 0.75. Our `POST /persons/cluster` is still
   union-find at a default of 0.7 ([cluster_types.md](cluster_types.md) section 4.2). Any graph
   method we adopt has to prove it does not chain. Mutual-kNN edges, CPM resolution and a
   CHAMELEON merge test are the guards #346 already lists. This note adds k-reciprocal edge
   weights and a rule that limits merges to one hop per run.
2. **One centroid per identity is a single-view model.** The speech project keeps one
   L2-normalised mean per profile and per cluster. That works for voice because a speaker's
   embeddings form one blob around the identity. Vehicle and face views are multi-modal: a mean
   of the front and rear views lies between them and matches neither well. **Clusters need several
   prototypes (one per view mode), and assignment should go to the nearest prototype rather than
   the centroid.** The speech project's own notes say "store multiple embeddings per profile, not
   just the centroid", but the code never did it.
3. **Corrections are the strongest signal, but only if they last.** The speech project has three
   correction mechanisms:
   - a cannot-link table
   - a per-speaker profile blacklist
   - a carry-over that re-applies a correction to whichever new speaker "spoke mostly inside" the
     corrected time span after re-processing
   It learned the third one the hard way: corrections tied to row ids disappeared when a file was
   re-processed, and the same wrong match came straight back. Our equivalent is to tie a
   correction to the box (by IoU overlap), not to a `box_id` or a cluster id.
4. **Keep the record of who made an identity.** Its machine auto-accept path (cosine of at least
   0.75) wrote the same `verified` flag a person writes. Privacy and purge code could then not tell
   machine identities from human ones, and it took a migration to undo. Our lock rule
   (`src/clients/occ_locks.py`) already separates human and import validation from machine
   writes. #343 must keep machine merges in their own provenance class and never promote them.
5. **A side attribute can veto a merge, but only together with weak embedding similarity.** The
   speech project uses predicted gender in three ways:
   - a speaker whose gender differs from a cluster's dominant gender must exceed 0.85 to join it,
     where the normal bar is 0.75
   - after a batch run, a member is evicted only if it is in a gender minority (a two-thirds
     majority is required) **and** its similarity to the majority centroid is below 0.70
   - neither signal acts alone
   For objects the attributes would be colour, body type and viewpoint, which the VLM attribute
   prompts can already produce. These are noisier than gender under changing light, so the
   both-signals rule matters even more.
6. **The speech project never measured cross-recording linking.** Its within-recording
   diarization is measured carefully: DER on AMI and VoxConverse, cpWER on Earnings-21, a gate
   with no regression allowed. Its cross-recording thresholds (0.5, 0.65, 0.75, 0.85, 0.90) were
   taken from vendor and community sources and never calibrated on labelled pairs. We already have
   the harness it lacks (#341). What we lack is **identity-level labels and viewpoint strata**:
   - the 3,000-item set labels classes, not identities
   - its near-duplicate stratum is the only same-object signal, and pair recall there is 0.24
7. **The same-recording rule gives a free cannot-link.** In diarization, two different speakers
   in one recording are different people. For identity clustering, two detections in the same
   image are two different objects (apart from reflections and mirrors). This needs no human
   label, and it caps chaining hard. It applies only at identity level. At class level, two cars
   in one image are one class.
8. **Clustering in two stages works in both domains.** The diarization stack seeds with
   agglomerative clustering (centroid linkage, threshold 0.6) and then refines with a
   probabilistic model (PLDA plus VB-EM, "VBx"). The pure-threshold engines it evaluated and
   rejected lost accuracy. That is the shape #346 proposes: an existing partition or Leiden seed,
   then constrained split and merge.

One correction to the #345 plan, checked against the vendor docs on 2026-10-28:
`cugraph.leiden(G, max_iter, resolution, random_state, theta)` optimises **modularity only**. It
has no CPM objective and no initial-partition argument. So "CPM, initialised from the current
partition" exists only on the CPU path (`leidenalg`). The GPU path has to be compared as plain
modularity Leiden, or run as ECG.

## 2. What the speech project does

### 2.1 Two problems, two engines

| | Within one recording (diarization) | Across recordings (speaker linking) |
|---|---|---|
| Question | Who spoke when, with local labels | Is local speaker A in file 1 the same person as local speaker B in file 9? |
| Embedding | WeSpeaker ResNet34, 256-d, one per (window, local speaker) | The same 256-d embedding, aggregated per local speaker from up to 5 of its longest clean segments |
| Algorithm | Filter, L2-normalise, seed by AHC (centroid linkage, threshold 0.6), PLDA transform, VB-EM, centroids, cosine soft-assign, per-window Hungarian matching | Online: kNN against stored cluster centroids and profile vectors. Batch: a two-phase re-cluster (2.3) |
| Constraints | Speaker count (min, max, forced), applied by re-running k-means when VBx's count falls outside the bounds | Cannot-link pairs, profile blacklist, tenant scope, attribute veto |
| Evaluation | DER at collar 0.25 on AMI (count-matched average about 0.22, in line with the published pipeline) and VoxConverse; cpWER on Earnings-21; a parity gate against the reference pipeline | No labelled benchmark. Only internal checks (2.5) |

The diarization side spent most of its effort on making the reference algorithm fast without
changing its output. The design rule written into its plan is "decision parity before speed
claims". A planned native rewrite of cross-recording linking (a warm in-process ANN index, batch
re-resolution) carries the same gate: identical match decisions on a fixture library before any
speed number counts.

### 2.2 Online assignment, tiered confidence

When a new local speaker arrives:

1. Query kNN (k=5) against cluster centroids in the vector store, filtered to the same tenant
   scope.
2. Join the best cluster if cosine is at least 0.75 and no correction forbids it. A candidate whose
   gender differs from the cluster's dominant gender must reach 0.85.
3. Otherwise create a singleton cluster.
4. Separately, profile matching has three tiers:
   - at least 0.75: auto-link (this is the path that wrote `verified`, see 2.4)
   - 0.50 to 0.75: a suggestion for review
   - below 0.50: nothing
5. A margin is stored with every assignment (best minus second-best similarity) for display. It
   is not used to decide anything.

The project's research note lists the threshold ladder it adopted: at least 0.90 for an automated
merge, 0.75 to 0.90 strong, 0.65 to 0.75 cluster formation, 0.50 to 0.65 weak. It cites a speech
toolkit's verification default (0.7) and vendor practice. These are uncalibrated starting points,
not measured operating points.

### 2.3 Batch re-cluster in two phases

1. **Anchors first.** Speakers that already share a person-verified profile are grouped
   deterministically, with no embedding maths. The trusted labels define these clusters.
2. **Discovery on the rest.** For unlabelled speakers, the job builds a full cosine matrix. It is
   chunked on the GPU in fp16 at 500 or more speakers and on the CPU below that. Then it runs
   SciPy complete-linkage AHC cut at distance 1 - 0.75. Each tenant partition is clustered
   separately, so private and organisation speakers never link.
3. **Post-hoc repair:**
   - In a group holding a cannot-link pair, evict the member less similar to the group centroid.
   - Apply the gender veto: with a two-thirds majority, evict a minority member whose similarity
     to the majority centroid is below 0.70.
   - Check that every pair in every cluster is at least the threshold (minus 1%), and log
     violations.
4. **Late joins.** Singletons are matched to the anchor clusters at the auto-link bar, unless a
   correction or an existing identity conflicts.
5. **Quality fields per cluster:**
   - mean member-to-centroid cosine
   - minimum pairwise cosine (the worst pair)
   - separation, which is 1 minus the highest cosine to any other centroid
   - per-member confidence, which is the cosine to the centroid
   - centroids weighted 2x for verified members

The batch job **deletes every cluster and rebuilds**. The only continuity is that anchor clusters
are recreated from their profiles. Unpromoted cluster identities are not stable across runs. This
is the same defect #354 records for our person grouping.

### 2.4 Human input, corrections and provenance

- **Merge, split, promote to profile, unassign.** Unassigning a speaker records a cannot-link
  with the cluster's other members and blacklists the profile. Every automatic path checks both
  before joining.
- **Corrections tied to data rather than rows.** Re-processing a file deletes its speakers, and
  the cascade took the corrections with them. The voiceprint stage then matched the new speakers
  to the old wrong profile again. The fix:
  - stash each correction as time spans before the re-run
  - re-apply it to every new speaker whose own speech falls **mostly** inside those spans
  - give nothing to a speaker that only touches a span, so the real person is not blocked from
    their own profile
- **Provenance.** Each speaker identity now records who made it (a person or a voiceprint), when,
  and from what. Before that, the auto-accept, propagation, retroactive-match and cluster-join
  paths all wrote `verified = True`. The privacy controls treated only unverified suggestions as
  machine-made, so the most common machine identities could be neither hidden nor purged.

### 2.5 Quality gating and space hygiene

- Embeddings come only from segments above a minimum duration, at most 5 per speaker, after
  adjacent segments are merged. Short or overlapped speech does not become identity evidence.
- Two embedding generations (512-d and 256-d) coexisted during a migration. A shared mode flag
  with a short-lived cache stopped workers writing at the wrong dimension. Vectors of mismatched
  dimension are skipped at clustering time. A periodic consistency job re-extracts embeddings
  missing from the vector store.
- Internal checks only: the pairwise-threshold check, quality and separation scores, and a
  coherence monitor idea (flag a profile whose minimum pairwise similarity is below 0.60). There
  is no purity, coverage, BCubed or false-merge rate against labelled identities.

### 2.6 Tuning history and failed attempts, in order

1. Union-find connected components at 0.65 merged different speakers through chains. It was
   replaced by complete-linkage AHC at 0.75, which the commit justifies as "complete linkage
   nearly guarantees same speaker".
2. Re-clustering cleared only unpromoted clusters, so promoted clusters piled up as duplicates. It
   was changed to clear all and rebuild anchors. That fixed duplication and made cluster
   identities unstable.
3. A name-suggestion step (NER over titles) ran per cluster inside the re-cluster. It caused
   timeouts and was removed from the pipeline.
4. A GPU context in every prefork worker held about 1.4 GB. The work was routed to the CPU below
   500 speakers.
5. The community recipe of UMAP then HDBSCAN (from a speaker toolkit) was recommended in research
   but not adopted. Our own experience is the reason not to: UMAP collapsed the pool into one
   cluster and HDBSCAN found no density gaps ([clustering_methods.md](clustering_methods.md)
   sections 2.1 and 2.3).
6. Corrections lost on re-processing and machine identities posing as human ones are the two
   largest defects. Both were found late, both needed migrations, and both are about lasting
   human input, not about the algorithm.

## 3. Where OpenProcessor stands

| | Speech project | OpenProcessor today |
|---|---|---|
| Level | Identity (a person) | Kind 2a/2b: **class level** (group unlabelled crops of one class for review). Kind 3b: **identity level** (faces). The owner's vehicle-identity case is **identity level**, and no mechanism serves it today |
| Space | 256-d identity-trained speaker embedding | PE-Core 1024 (a general semantic encoder, not identity-trained), MobileCLIP 512, ArcFace 512 (identity-trained) |
| Online | kNN to centroids, 0.75 | 2a: nearest of 512 persisted IVF centroids, always assigned, never abstains |
| Batch | Anchors, then complete-linkage AHC | 2a: FAISS k-means K = min(512, n // 6), AHC refine on demand; 3b: union-find at 0.7 |
| Constraints | Cannot-link, blacklist, carried by data overlap | Lock rule (validated, imported and holdout items never move); confident items stay out of the pool; no pairwise constraints |
| Stability | Full rebuild, unstable ids | 2a: ARI 0.306 under row permutation (#372); 3b: fresh ids every run (#354) |
| Evaluation | DER inside a recording only | Harness (#341): on the 3,000-item set, pair precision 0.696, false-merge rate 0.304 (0.400 on low-res), near-duplicate pair recall 0.244 |

The level matters more than the algorithm. PE-Core places two silver sedans of one model close
together whether or not they are the same car, because it was trained for semantics, not for
identity. Viewpoint, in contrast, moves a crop a long way. So for kinds 2a and 2b, a viewpoint
change mostly causes **false splits within a class**: front-view cars in one bucket, rear-view
cars in another. The false-split rate of 0.98 on the 3,000-item set fits this, although that set
cannot separate viewpoint from other causes. At identity level the problem is harder in both
directions. Same-identity views drift apart, and same-model different-identity objects collide.
That is the re-identification problem, and its literature (section 7) gets roughly 80 to 85 mAP
on VeRi-776 only with identity-trained models. **Whether frozen PE-Core can do vehicle identity at
all is the first thing to measure (E5). It should not be assumed.**

## 4. Transferable techniques, mapped

Each row gives the speech evidence, the object and face mapping, and where it lands.

| # | Technique | Speech evidence | Object / face mapping | Lands in |
|---|---|---|---|---|
| T1 | **Separate identity from view, with multi-prototype clusters** | Mean profiles; the notes call for keeping several embeddings | A cluster holds P prototypes (the medoids of its sub-communities or `cluster_subid` groups), each optionally tagged with a view. Membership and merge scores use the best prototype pair, never the mean | #343 cluster record; #344 assignment |
| T2 | **Prevent chaining** | Union-find at 0.65 replaced by complete linkage at 0.75 | Mutual-kNN graph; k-reciprocal (Jaccard) edge reweighting (Zhong et al. 2017); CPM resolution; at most one merge hop per run without verification; replace union-find in `POST /persons/cluster` | #342, #354 |
| T3 | **Link clusters across views with pairwise verification** | Profile-tier matching; set-level evidence (up to 5 segments per speaker) | Score a cluster pair set-to-set (top-m prototype pairs, after normalisation). A candidate merge needs a second, independent check: (a) local-feature geometric verification on the best crop pairs, which works for the same rigid object across moderate view changes and fails near 180 degrees; (b) a re-ID embedding if one is adopted (E5); (c) a VLM "same object?" judgement for the few borderline pairs, under the loop's budget | #343 merge validator |
| T4 | **Constrained clustering** | Anchors first; cannot-link eviction; blacklist; carry-over by span overlap | Must-link: human-confirmed same identity, near-duplicate groups, and video tracklets if video ingest gains tracking. Cannot-link: human splits, validated different classes (class level), and **two boxes in one image (identity level)**. Constraints are soft with a confidence (arXiv 2212.14437). Each is tied to `(crop_id, bbox_norm)` and re-attached after re-detection by IoU of at least 0.5, the analogue of "spoke mostly inside" | #343 |
| T5 | **Online assignment with abstention** | Accept at 0.75 or more, suggest at 0.50 to 0.75, otherwise a new singleton; margin stored | Nearest prototype. Accept if the normalised score is at least t_accept **and** the margin to the second cluster is at least m. Between t_review and t_accept, send to review. Below, start a microcluster. Never write `validated`. Today 2a always assigns | #344 |
| T6 | **Hierarchical merge with stability** | (none; full rebuild) | Run the partition over seeds, row permutations and resolutions; use the co-association matrix or ECG consensus; accept a merge only if it holds in at least 80% of runs; send unstable items to review. Also answers #372 | #342, #343, #372 |
| T7 | **Cluster first, then refine** | AHC seed, then PLDA plus VB-EM refinement | Seed from the current partition (IVF plus refine) or Leiden. Refine with CHAMELEON relative interconnectivity and relative closeness for merges, and dispersion plus sub-community structure for splits | #342, #343 |
| T8 | **Calibrated, normalised thresholds** | Thresholds borrowed from vendor and community sources, never calibrated | Map a raw cosine to a calibrated same-identity probability per space and per level, using labelled pairs (isotonic or logistic fit). Apply cohort score normalisation first (adaptive s-norm against each item's top-k impostors; Matejka et al. 2017), so dense regions and sparse views share one threshold. Optional threshold per view pair, after VANet (Chu et al. 2019) | #342 calibration step; the harness |
| T9 | **Gate on embedding quality** | Minimum segment duration, top-5 segments | Crops below 32 px, blurred, truncated or heavily occluded carry weak identity evidence. Let them join clusters but never form or bridge one. Exclude them from prototypes and from merge evidence. The low-res stratum has the worst false-merge rate (0.400) | #342 graph build; #344 |
| T10 | **Provenance** | The machine-written `verified` defect | Machine merges, propagations and assignments carry their own `label_provenance` and never count as validated. The lock rule stays the only gate | #343, #286 |
| T11 | **Veto by a side attribute** | Gender: 0.85 cross-attribute bar, eviction only with both signals | VLM attributes (colour, body type, view) or a detector class. Raise the bar on an attribute conflict; evict only with both signals; never veto on the attribute alone, because colour shifts with light | #343 merge validator |
| T12 | **Evaluate identity across views** | DER, purity and coverage inside a recording; nothing across recordings | BCubed precision and recall as the headline (Amigo et al. 2009), plus pairwise F and AMI (not ARI alone, because identity sizes are long-tailed). Report false merges apart from false splits. **Report same-identity pairs stratified by view difference** (same view, adjacent, opposite) and by crop size | #341 harness extension (E0) |

### 4.1 Notes on the harder rows

- **T1, how many prototypes.** Use the sub-communities Leiden already finds inside a community,
  or the AHC `cluster_subid` groups. Cap at about 8 per cluster, and keep each prototype's member
  count and dispersion. Each prototype is a medoid, a real crop. That keeps it inspectable in
  Cropwright and gives the auto-label loop one sample per view to send to the VLM.
- **T3, verifier choice.** Local-feature matching on two crops costs milliseconds per pair on a
  GPU. It suits only rigid objects with overlapping visible surfaces, so not front against rear.
  For opposite views no geometric check exists. Such a merge needs human confirmation or an
  identity-trained model. This is a structural limit to state in the UI: "linked by evidence"
  versus "linked by a person".
- **T4, the same-image cannot-link.** It is cheap and strong, and also dangerous at the wrong
  level. Put it only in identity graphs (faces 3b, any future object-identity kind), never in the
  2a or 2b class graphs.
- **T8, calibration data.** Faces get labelled pairs from public cross-pose sets (E6). Objects get
  them from the multi-view sets in E4 and E5. Class-level 2a calibrates against the 3,000-item
  class labels. A threshold is valid only for its (space, level, model version), so it must be
  stored with the partition run.

## 5. What does not transfer

- **Temporal structure.** VBx exploits the order of turns inside a recording. Still images have
  none. Video tracklets would be the only analogue, and only if tracking exists.
- **The threshold numbers.** 0.65, 0.75, 0.85 and 0.90 belong to one 256-d speaker model. For
  PE-Core, MobileCLIP and ArcFace each needs its own calibration (T8). Copying numbers would
  repeat the speech project's uncalibrated ladder.
- **A mean centroid as the profile.** It is fine for a single-mode voice blob and wrong for
  multi-view objects (T1).
- **Identity-trained embeddings.** Speaker embeddings are trained with verification losses.
  PE-Core is not. At class level this hardly matters. At object-identity level it may make the
  task infeasible without a re-ID model (E5).
- **A small number of identities per batch.** A recording holds 1 to 20 speakers, and a library a
  few thousand. A curation project holds hundreds of thousands of crops. Dense n x n matrices,
  which the speech project builds for up to about 15,000 speakers, are ruled out (#354 already
  shows about 40 GB at 100,000 faces). Everything here must be kNN-sparse.
- **Gender as a near-binary attribute.** Gender is fairly reliable from audio. Object attributes
  are noisy and often several attributes at once, so the veto must be soft (T11).
- **Full rebuild.** The speech project's destructive re-cluster is the anti-pattern #343 exists
  to avoid.
- **Biometric privacy rules.** Voiceprints and face embeddings are both biometric. The speech
  project's per-scope "voiceprints off" switch and purge are relevant to faces (kind 3). They do
  not apply to object crops, and they are out of scope here.

## 6. Design proposal

### 6.1 Name the level

Every cluster record (#343) carries `level ∈ {class, identity}` and its space (`pe_embedding`, box
vector, ArcFace). Graph rules depend on the level:

| Rule | Class level (2a, 2b) | Identity level (3b faces; future object identity) |
|---|---|---|
| Same-image cannot-link | no | yes |
| Validated different-class cannot-link | yes | yes |
| Multi-prototype | yes (view modes of a class) | yes (views of one identity) |
| Merge across views | CHAMELEON test plus calibrated set-to-set score | additionally needs a verifier or a human |
| Abstention | to review | to review, and never auto-named |

### 6.2 What the cluster record holds

On top of the #343 fields (`cluster_uuid`, `parent_uuid`, `level`, `cluster_run_id`,
`membership_confidence`, `embedding_model_version`, lineage, `label_provenance`):

- `prototypes[]`: medoid `crop_id` (plus `box_id` for 2b), member count, dispersion, optional view
  tag and its source (VLM, estimator, human)
- per member: best prototype, calibrated score, margin to the second cluster, state
  (`assigned`, `review`, `abstained`)
- `links[]` between clusters: evidence type (`embedding`, `verifier`, `vlm`, `human`), score and
  run id, so a merge can be explained and undone

### 6.3 Refinement pass

This extends the #346 pipeline:

1. Build a kNN graph in one space (k about 30), with cohort-normalised scores (T8) and
   k-reciprocal reweighting (T2). Quality-gated crops (T9) get edges to the graph but none from
   it, so they cannot bridge.
2. Add constraints (T4): must-link edges at high weight, and cannot-link as a hard check after
   each move.
3. Run Leiden CPM at three or more resolutions, seeded from the current partition (CPU
   `leidenalg`). Repeat over seeds and permutations to get co-association (T6).
4. Take prototypes from the sub-communities (T1).
5. Merge proposals: cluster pairs whose best prototype pair is in the calibrated "same" band and
   passes the CHAMELEON relative interconnectivity and relative closeness tests (T7), the
   attribute rule (T11) and no cannot-link. Identity level also needs a verifier. Allow at most
   one hop per run (T2).
6. Split proposals: a cluster whose prototypes do not link to each other above the band, or that
   holds a cannot-link pair.
7. Every proposal goes through the dry-run-first apply path, as assist apply does. Validated,
   imported and holdout items never move. Machine provenance is kept (T10).

### 6.4 Online assignment (#344)

Assign to the nearest prototype using the calibrated score and margin (T5). A crop in the review
band waits for the next local re-evaluation. A crop below the band starts a microcluster
(representative, count, dispersion) that the scheduled pass can promote. Unlike today's 2a,
"no good home" is allowed.

### 6.5 Effect on the auto-label loop (#286)

- **Sample per prototype.** Assist runs already stratify samples by sub-cluster. Sampling at
  least one crop per prototype stops a cluster holding front and rear views from being judged on
  one view only. Agreement is computed per prototype first. A cluster propagates only if every
  prototype passes the gate. Otherwise it splits along the prototypes that failed, which is the
  loop's "split and recurse" step with a better split.
- **Abstained and review crops never propagate.** The gate's 0.9 threshold then applies to
  confident members only.
- **Loop outcomes become constraints.** A VLM disagreement between two prototypes is a soft
  cannot-link candidate. Human review outcomes become hard constraints for the next pass (T4).

## 7. Experiment plan, in priority order

All offline, research scripts only. No project writes. Public data. GPUs 0 and 2 under the lock,
never GPU 1.

| Order | ID | Experiment | Data | Measures | Size |
|---|---|---|---|---|---|
| 1 | E0 | **Harness extension.** Add BCubed P/R, pairwise F, AMI, a risk-coverage curve for partitions with abstention, and an optional `identity` and `view` field per label, with pairs stratified by view difference and crop size | Existing harness, the 3,000-item set, planted synthetic sets | Unit-tested metrics; the baseline re-scored | 1 to 2 days, CPU |
| 2 | E1 | **Chaining test.** Union-find (current 3b), complete-linkage AHC, average linkage, mutual-kNN plus Leiden CPM, plus k-reciprocal reweighting | 3,000-item set (2a), the planted set's bridge items, faces (E6) | False-merge rate, BCubed precision at fixed recall, bridge-item merges | 2 days, CPU; overlaps #342, so run inside its driver |
| 3 | E2 | **Normalisation and calibration.** Raw cosine against cohort s-norm (top-k impostors, k in {50, 200}); isotonic calibration from labelled pairs; one threshold against one per stratum | 3,000-item set; faces | Precision at fixed recall per stratum; spread of the best threshold across strata (smaller is better) | 1 to 2 days, CPU |
| 4 | E3 | **Multi-prototype against centroid assignment.** Hold out 20% of items (seeded), build clusters on the rest, assign held-out items by centroid or by nearest of P prototypes (P in {1, 2, 4, 8}), with and without the margin rule | 3,000-item set (near-duplicate and similar-class strata), E4 data | Accept precision at 60/80/95% coverage; abstention rate | 2 days, CPU |
| 5 | E4 | **Viewpoint-stratified identity on multi-view public data.** Object sequences with camera poses, where identity is the sequence and view difference is the azimuth difference between frames. Bins 0-30, 30-90, 90-150 and 150-180 degrees. PE-Core and MobileCLIP frozen, then E1 to E3 techniques on top. Add hard negatives: other sequences of the same category | CO3D v2 categories car, motorcycle, bicycle (CC BY-NC 4.0; data licence to confirm); Objectron bike and shoe (C-UDA 1.0); CompCars five view labels (non-commercial research only) | Same-identity pair recall per azimuth bin at fixed false-merge rate; BCubed. **Answers: does a frozen PE-Core link a front view to a rear view at all?** | 3 to 4 days, about 2 GPU-hours encoding on GPU 0 |
| 6 | E5 | **Re-ID ceiling.** The same protocol on vehicle and person re-ID sets: frozen PE-Core against a published re-ID model, to measure the gap a dedicated identity embedding would close | VeRi-776 (on request, non-commercial, no redistribution; MIT orientation and keypoint labels exist separately), VehicleID (academic only, signed agreement), Market-1501 (research only per secondary sources). DukeMTMC is withdrawn: do not use it | mAP and Rank-1 in the standard protocol plus the E0 clustering metrics; cross-view pairs from the orientation labels | 3 days after access; **owner decision on licences first**. Results stay local unless the owner clears publishing aggregates |
| 7 | E6 | **Faces across pose (#354).** Union-find at 0.7 against mutual-kNN Leiden against average-linkage AHC, with same-image cannot-link and multi-prototype persons | A public cross-pose face set (CPLFW or CFP-FP; licences to verify) plus faces detected in COCO person crops | BCubed and pairwise F by pose bin; chaining incidents; memory under kNN, which must be O(n k) | 2 to 3 days |
| 8 | E7 | **Value of constraints.** Simulate human answers from ground truth at budgets of 0.5%, 1% and 2% of items. Compare random queries against active queries (A3S-style split/merge queries, Deng et al. 2024). Include the noise case (5% wrong answers) to test soft constraints | 3,000-item set; E4 | False-merge reduction per query; damage from noisy constraints | 2 days, CPU |
| 9 | E8 | **Attribute veto.** Get colour, body type and view from the VLM attribute prompt on a sample, then veto merges only when both signals agree (T11) | 3,000-item vehicle classes; E4 cars | False merges removed against true merges lost; how often the attribute flips on the same object across views | 2 days plus VLM time under the loop's budget |
| 10 | E9 | **Instability as an error detector (#372).** Co-association over 10 permutations and seeds; flag items that change cluster in more than 20% of runs | 3,000-item set | Precision and recall of "unstable" against actual false merges; ARI across permutations before and after sorting the pool | 1 day, CPU |

**Gates.**

- E1 to E3 decide what #342 recommends and what #343 builds: the merge rule, the threshold
  method and the prototype count.
- E4 decides whether object identity is in scope with PE-Core.
  - If cross-view recall in the 150-180 degree bin stays near chance at an acceptable
    false-merge rate, record **no-go for frozen-embedding object identity**. Then either propose
    a re-ID embedding as a model decision (E5 gives the size of the gain) or keep object work at
    class level with multi-prototype views.
- E6 decides the #354 fix.

**What the doc commits.** Aggregate tables from public data in `docs/research/`. Images and
per-item outputs stay in `artifacts_local/`. E5 numbers are committed only after the owner rules
on the licence.

## 8. How the parts connect

- **#342:** add E1, E2 and E9 to the prototype run. Report BCubed and stratified results next to
  the planned false-merge, split-recovery, unassigned and stability numbers. Use CPU `leidenalg`
  for CPM with an initial partition.
- **#343:** the record fields in 6.2, level-aware constraints (6.1), soft constraints with
  confidence, corrections re-attached by IoU, machine provenance, the one-hop merge rule.
- **#344:** assignment to the nearest prototype with margin and abstention; microclusters for
  "no home".
- **#345:** GPU Leiden is modularity-only with no initial partition. Compare it as such, or as
  ECG for consensus. FAISS with cuVS or CAGRA is the likely GPU kNN graph builder at 1M.
- **#286:** per-prototype sampling and agreement; abstained crops never propagate; loop outcomes
  become constraints.
- **#354, #372:** solved by E6 and E9 plus the stable-id rule in #343.

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Frozen PE-Core cannot separate identities of the same model | Object-identity clustering merges look-alikes | E4 and E5 measure it first; the go/no-go is recorded; class level stays the default |
| Opposite views cannot be verified geometrically | Front and rear of one object stay split | Accept the split. Link by a person or a re-ID model, and say so in the UI |
| Calibration overfits the public sets | Thresholds wrong on real projects | Calibrate per space and level, store the calibration with the run, and re-check on a sample of each new project (the owner's private projects stay local) |
| Constraint conflicts (noisy human answers) | Infeasible partitions or oscillation | Soft constraints with confidence; E7 noise case |
| VLM attributes drift with light | Wrong vetoes | Veto only with both signals; E8 measures how often attributes flip |
| More moving parts in the record | Migration and contract churn | Land with #343 in one mapping migration; contracts regenerated in the same commit |
| Dataset licences (VeRi, VehicleID, CompCars and CO3D are non-commercial or gated) | Results cannot be published, or images shown | Local-only runs; publish aggregates only with owner approval; never commit images |
| Confirmation bias if loop outcomes feed back as labels | Errors harden | Only person-confirmed constraints are hard; machine outcomes stay soft and expire on the next run (Arazo et al.) |

## 10. Open questions for the owner

1. Is object identity (the same physical vehicle) an actual product goal, or is multi-view
   grouping at class level the goal? The answer decides whether E5 and a re-ID embedding are
   needed.
2. Licences: may we run VeRi-776, VehicleID, CompCars and CO3D locally (all non-commercial or
   gated), and publish aggregate numbers from them?
3. Faces: do we need a "face embeddings off and purge" control like the speech project's, before
   identity work on faces grows?
4. Should a VLM "same object?" check be an allowed merge verifier within the loop's call budget,
   or should identity merges always need a person?
5. Does video ingest with tracking belong on the roadmap? Tracklets are the strongest free
   must-link source for identity clustering.

## 11. References

Labels: **PR** peer-reviewed, **PP** preprint only, **DOCS** vendor or official documentation,
**COMM** community or blog. Titles, venues and URLs were checked on 2026-10-28. Numbers are quoted
only where the source was read. CLIP-ReID's VeRi-776 table was read from the paper (no
re-ranking): TransReID ViT 80.6 mAP / 96.9 Rank-1, CLIP-ReID ViT 83.3 / 97.4, CLIP-ReID with side
information 84.5 / 97.3.

Graph clustering and validity:
- Traag, Waltman, van Eck. From Louvain to Leiden: guaranteeing well-connected communities. Sci.
  Rep. 2019. PR. https://www.nature.com/articles/s41598-019-41695-z
- Karypis, Han, Kumar. CHAMELEON: hierarchical clustering using dynamic modeling. IEEE Computer
  1999. PR. https://doi.org/10.1109/2.781637
- Sarfraz, Sharma, Stiefelhagen. Efficient parameter-free clustering using first neighbor
  relations (FINCH). CVPR 2019. PR. https://arxiv.org/abs/1902.11266
- Poulin, Theberge. Ensemble clustering for graphs (ECG). Complex Networks 2018 (Springer 2019).
  PR. https://arxiv.org/abs/1809.05578
- Lin et al. Maintaining Leiden communities in large dynamic graphs (HIT-Leiden). 2026. PP.
  https://arxiv.org/abs/2601.08554
- Sahu. Heuristic-based dynamic Leiden for evolving graphs. 2024. PP.
  https://arxiv.org/abs/2410.15451
- Ben-Hur, Elisseeff, Guyon. A stability based method for discovering structure in clustered
  data. PSB 2002. PR. https://psb.stanford.edu/psb-online/proceedings/psb02/benhur.pdf
- von Luxburg. Clustering stability: an overview. Found. Trends ML 2010. PR.
  https://arxiv.org/abs/1007.1075
- Amigo, Gonzalo, Artiles, Verdejo. A comparison of extrinsic clustering evaluation metrics based
  on formal constraints (BCubed). Inf. Retrieval 2009. PR.
  https://link.springer.com/article/10.1007/s10791-008-9066-8
- Vinh, Epps, Bailey. Information theoretic measures for clusterings comparison. JMLR 2010. PR.
  https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf
- Romano et al. Adjusting for chance clustering comparison measures. JMLR 2016. PR.
  https://arxiv.org/abs/1512.01286

Re-identification and viewpoint:
- Zhong et al. Re-ranking person re-identification with k-reciprocal encoding. CVPR 2017. PR.
  https://arxiv.org/abs/1701.08398
- Chu et al. Vehicle re-identification with viewpoint-aware metric learning (VANet). ICCV 2019.
  PR. https://arxiv.org/abs/1910.04104
- Meng et al. Parsing-based view-aware embedding network for vehicle re-identification (PVEN).
  CVPR 2020. PR. https://arxiv.org/abs/2004.05021
- He et al. TransReID: transformer-based object re-identification. ICCV 2021. PR.
  https://arxiv.org/abs/2102.04378
- Li, Sun, Li. CLIP-ReID: exploiting vision-language model for image re-identification without
  concrete text labels. AAAI 2023. PR. https://arxiv.org/abs/2211.13977
- Lu et al. CLIP-SENet for vehicle re-identification. 2025. PP (its best VeRi-776 figure reportedly
  uses re-ranking; not comparable). https://arxiv.org/abs/2502.16815
- Ge et al. Self-paced contrastive learning with hybrid memory (SpCL). NeurIPS 2020. PR.
  https://arxiv.org/abs/2006.02713
- Dai et al. Cluster contrast for unsupervised person re-identification. ACCV 2022. PR.
  https://arxiv.org/abs/2103.11568
- Zhang et al. Implicit sample extension for unsupervised person re-ID (ISE, targets split and
  mixed clusters). CVPR 2022. PR. https://arxiv.org/abs/2204.06892
- Cho et al. Part-based pseudo label refinement (PPLR, cross-space neighbour agreement). CVPR 2022.
  PR. https://arxiv.org/abs/2203.14675

Face clustering:
- Yang et al. Learning to cluster faces on an affinity graph. CVPR 2019. PR.
  https://arxiv.org/abs/1904.02749
- Yang et al. Learning to cluster faces via confidence and connectivity estimation. CVPR 2020. PR.
  https://arxiv.org/abs/2004.00445
- Shen et al. Structure-aware face clustering on a large-scale graph with 10^7 nodes (STAR-FC).
  CVPR 2021. PR. https://arxiv.org/abs/2103.13225
- Wang et al. Ada-NETS: face clustering via adaptive neighbour discovery in the structure space.
  ICLR 2022. PR. https://arxiv.org/abs/2202.03800

Constrained and active clustering, pseudo-label noise:
- Wagstaff et al. Constrained k-means clustering with background knowledge. ICML 2001. PR.
  http://wkiri.com/research/papers/wagstaff-kmeans-01.pdf
- Gonzalez-Almagro et al. Semi-supervised constrained clustering: an in-depth overview. Artif.
  Intell. Rev. 2024. PR. https://link.springer.com/article/10.1007/s10462-024-11103-8
- Clustering with confidence-based must-link and cannot-link constraints. 2022 (rev. 2024). PP.
  https://arxiv.org/abs/2212.14437
- Deng et al. A3S: a general active clustering method with pairwise constraints. ICML 2024. PR.
  https://arxiv.org/abs/2407.10196
- Arazo et al. Pseudo-labeling and confirmation bias in deep semi-supervised learning. IJCNN 2020.
  PR (arXiv page). https://arxiv.org/abs/1908.02983
- Kage et al. A review of pseudo-labeling for computer vision. 2024. PP.
  https://arxiv.org/abs/2408.07221

Speaker diarization and score calibration:
- Landini et al. Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization. 2022.
  PR (Computer Speech and Language). https://arxiv.org/abs/2012.14952
- Park et al. Auto-tuning spectral clustering for speaker diarization using normalized maximum
  eigengap. IEEE SPL 2020. PR. https://arxiv.org/abs/2003.02405
- Matejka et al. Analysis of score normalization in multilingual speaker recognition (adaptive
  s-norm). Interspeech 2017. PR.
  https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html
- Bredin. pyannote.metrics (DER, purity, coverage). Interspeech 2017. PR.
  https://www.isca-archive.org/interspeech_2017/bredin17_interspeech.pdf
- WeSpeaker toolkit (embedding model and a UMAP plus HDBSCAN recipe). COMM/DOCS.
  https://github.com/wenet-e2e/wespeaker

GPU implementations:
- RAPIDS cuGraph Leiden (modularity, no initial partition). DOCS.
  https://docs.nvidia.com/cugraph/latest/api_docs/api/cugraph/cugraph.leiden/
- RAPIDS cuGraph ECG. DOCS.
  https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.ecg/
- cuML HDBSCAN (`approximate_predict`, membership vectors). DOCS.
  https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.cluster.hdbscan.hdbscan/
- cuVS integration with FAISS. DOCS. https://docs.rapids.ai/api/cuvs/stable/integrations/faiss/
- Ootomo et al. CAGRA: GPU graph construction and ANN search. ICDE 2024. PR.
  https://arxiv.org/abs/2308.15136
- Douze et al. The Faiss library. 2024. PP. https://arxiv.org/abs/2401.08281
- leidenalg (CPM, initial membership). DOCS. https://github.com/vtraag/leidenalg

Datasets and terms (check before any use):
- VeRi-776: request by email, non-commercial, no redistribution. https://vehiclereid.github.io/VeRi/
- Orientation and keypoint labels for VeRi-776 (labels only, MIT). COMM.
  https://github.com/Zhongdao/VehicleReIDKeyPointData
- VehicleID: academic use only, signed agreement. https://www.pkuml.org/resources/pku-vehicleid.html
- VERI-Wild: request-gated; exact terms not verified. https://github.com/PKU-IMRE/VERI-Wild
- Market-1501: research use per secondary sources; official terms not verified.
  https://zheng-lab-anu.github.io/Project/project_reid.html
- DukeMTMC: withdrawn by its owner in 2019; do not use. https://exposing.ai/duke_mtmc/
- MSMT17: academic only, no redistribution, no images in publications.
  https://www.pkuvmc.com/agreement/RELEASE_AGREEMENT-MSMT17.pdf
- CompCars (five viewpoint labels): non-commercial research only.
  https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
- CO3D (multi-view sequences with camera poses): repository under CC BY-NC 4.0; data terms to
  confirm. https://github.com/facebookresearch/co3d
- Objectron (camera poses, 3D boxes): C-UDA 1.0.
  https://github.com/google-research-datasets/Objectron

Internal: [cluster_types.md](cluster_types.md), [clustering_methods.md](clustering_methods.md),
[cluster_eval_baseline.md](cluster_eval_baseline.md), [gpu_clustering.md](gpu_clustering.md). The
owner's literature notes on Leiden, CHAMELEON, FINCH, consensus clustering and HIT-Leiden
(untracked, in the main checkout) are reconciled here: their six-step engine is section 6.3 with
T1, T2, T4, T8 and T9 added; their safeguards table is T2, T4, T6 and T9; their caution on
retraining embeddings is kept (E5 measures the gap first and changes nothing).
