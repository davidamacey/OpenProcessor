# v0.7.0 raw result files

Small raw files behind the numbers in [`docs/research/v070_measurement_log.md`](../../research/v070_measurement_log.md).
The log's "Raw-data index" says which file backs which section. Each file is exactly what the bench tool wrote
(one JSON object per run, or one per line), copied unchanged. Files too large to commit (vector dumps, per-request
latency lists, 5 s and 500 ms `nvidia-smi` and `docker stats` samples) stay in the artifact root named in the
index; `MANIFEST.sha256` has the checksums of the vector dumps.

| Directory | Backs |
|---|---|
| `cluster_ladder/` | cluster-pass ladder, memory table, k-NN table, isolated fit (log "Results") |
| `ingest_campaign/` | sections A, B, C and D of the live ingest campaign |
| `live_e/` | E1 to E4 (embed pass, concurrent duplicates, text search restart, instance count) |
| `live_f/` | F1 to F4 (embed pass on the GPU decode) |
| `live_g/` | G1 to G5 (region parity, chunk window tuning, final configuration) |
| `sp0/` | SP0 stage profile: per-repetition reports and summaries of every row, the two rendered tables (log "SP0: stage profile", PERFORMANCE "0.7.0 stage profile"); set P rows hold numbers only |
| `sp0/gpu_prep_concurrency/` | SP0.7 (log "SP0.7"): headroom against full model set, instance count and in-flight settings, thread samples; `metrics.api.counters` cut to the `op_ingest_*` series; set P rows hold numbers only |
| `spf/` | SPF (log "SPF"): memory budget, WORKERS and MAX_IN_FLIGHT, the ingest load profile, acceptance rows, model parity, per-model memory and the endpoint sweep; `metrics.api.counters` cut to the `op_ingest_*` series; set P rows hold numbers only |
