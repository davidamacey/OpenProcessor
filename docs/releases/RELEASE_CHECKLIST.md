# v0.4.1 release checklist

Every step is manual and owner-run. Run from a clean checkout of the release
commit. Nothing here is automated by CI (images are built locally).

## images.lock: what is a placeholder

All 16 entries in `images.lock` are development placeholders (`...dev0` to `...devf`,
not 64-hex digests) and must all be filled; the installer refuses them.

- Built and pushed by step 4 (`davidamacey/...:0.4.1`): `api`, `triton`, `evaluator`,
  `segmenter`, `trainer`.
- Third-party, resolved from upstream tags by step 4: the two `vlm_*` keys,
  `opensearch`, `opensearch_dashboards`, `mlflow`, `prometheus`, `grafana`, `loki`,
  `alloy`, `node_exporter`, `dcgm_exporter`. Each tag must be pulled locally first
  (`docker pull <image>`), because the script reads its RepoDigest.

`make release` rewrites `images.lock` entirely; do not hand-edit digests.

## Release decisions (#44, #63): all DECIDED

All decisions are final (owner, 4 Oct 2026). The only step left is the owner's go to publish.

- **`:latest` tag: DECIDED.** v0.4.1 is the numbered release and `latest` is published
  together with it, so `latest` equals `0.4.1` at release. `make release` pushes
  `:0.4.1` and `:latest` for each image, and refuses a non-X.Y.Z version, so `latest`
  can only ever point at a stable release. A later stable release moves it by running
  `make release` for that version; a backport or an old line is pushed by hand with
  `docker push` of the version tag only. The installer pins digests from `images.lock`
  and never uses `latest`. Rationale: pinned digests keep installs reproducible and
  `latest` is a convenience for manual pulls.
- **Control-plane-only mode: DECIDED.** Kept as an optional installer flag
  (`--cpu --control-plane-only`), never the default and not part of the supported
  production path. It starts only OpenSearch and the API (plus Cropwright if asked), with
  no Triton, workers, VLM, segmenter, trainer or monitoring; inference routes answer 503.
  The default install requires a GPU and builds the full stack for the chosen tiers.
  Rationale: ship what is tested by default and keep the no-GPU path explicit and opt-in.
- **Unverified VLM catalog entries: DECIDED.** Kept, marked `unverified/experimental`
  in the catalog label (`Unverified (experimental)`), the installer warnings and the
  docs. The verified (`tested`) default is unchanged and unverified rows are never
  auto-selected (explicit `--vlm-model-id` only). Rationale: users can opt in, but an
  untested model that fails to load is not mistaken for a product bug.
- **OpenSearch heap and shard budget: DECIDED.** Default heap is half of host RAM, 2 GB
  minimum, 30 GB cap, `-Xms` equal to `-Xmx`, memory locked; the installer also warns
  when `vm.max_map_count` is below 262144. Each project costs 6 shards; the soft budget is
  heap GB x 20 (`OP_SHARDS_PER_HEAP_GB`). Creating past it succeeds with a
  `shard_budget_high` warning; past `cluster.max_shards_per_node` it is refused
  (`409 shard_budget_exceeded`). Documented in `docs-site/docs/deployment/sizing-and-storage.mdx`.
  Rationale: this is OpenSearch's own sizing guidance, and the existing warn/refuse
  behavior already tells operators when to raise the heap.
- **Docs hosting: DECIDED.** Both. A public GitHub Pages site built from `docs-site/` by
  `.github/workflows/docs.yml`, and a local docs container (`docker compose -f docker-compose.yml -f docker-compose.docs.yml up -d --build docs`, port 4613) for source checkouts. Rationale: Pages is free and
  already built in CI; the container serves offline readers. Owner one-time steps (not done):
  Settings > Pages > Source "GitHub Actions"; allow `main` on the `github-pages` environment;
  run the workflow by hand once from `main`. The docs image is not in `images.lock` or
  `make release` (installed stacks read the public site).
- **Trivy remainder (#110): DECIDED.** `linux-libc-dev` kernel-header CVEs (7 ids) and
  libxml2 `CVE-2026-6653` are owner-accepted and allowlisted per CVE with a reason in
  `scripts/release/trivy-allowlist.txt` (headers only, no kernel runs in a container, no
  upstream fix); re-check each release and drop entries when a fixed package ships.
  The trainer `mlflow` 2.x pin stays a documented known item (a 3.x move needs client
  and server together).

## Acceptance status and open items

- Installer acceptance passed 4 Oct 2026 (clean install, lifecycle, upgrade/rollback,
  teardown, installed-stack e2e ingest to delete); fixes #105 to #109 are merged.
- Decided: Grafana keeps the default admin password for LAN-only installs; the docs
  warn to change it before exposing beyond the LAN.
- No open decisions remain except the owner's go to publish.
- Known items shipping in 0.4.1: the trainer `mlflow` 2.x pin (#110, allowlisted).
  #111 (export OOM on a shared 12 GB card) is fixed in 0.4.1; #44 (clean-host install
  from the published artifacts) stays open until run. #113 (`/models/status` sam3) is fixed: the installer sets
  `OP_SEGMENTER_URL` with the segmenter tier. Allowlisted CVEs are re-checked each release.

## Scripted acceptance run (release gate)

Replaces the manual acceptance walk-through (issue #54, plan:
`docs/design/release_acceptance_plan.md`). Install the release candidate with the
installer into an ISOLATED compose project and install directory (non-default
`--port-base`, GPUs nothing else uses), then run:

```bash
python3 scripts/datasets/fetch_coco_subset.py --out data/samples/coco_va   # public COCO, once
scripts/release/acceptance_run.sh \
  --base-url http://localhost:<port> --project-name <compose-project> \
  --install-dir <install-dir> --models-dir <install-dir>/models \
  --project acc-v041 --images-dir data/samples/coco_va --report acceptance.json
# or: make acceptance ACC_ARGS="<the same flags>"
```

It exercises health, self-hosted docs, ingest of ~100 COCO images, clustering, VLM
labeling, human confirmation, holdout, export, a 2-epoch probe train, bake-off,
promote (forced when the gate rejects the undertrained model), inference via
`/detect?model_name=`, model delete, metrics and log-noise checks, and the installer
lifecycle (`--only installer_uninstall` and `--upgrade-version vX` are opt-in). A
cleanup trap always deletes the throwaway project and promoted models. Exit status is
non-zero if any required phase failed. The trainer refuses classes under 20 validated
crops, so the run confirms `--confirm-crops` (default 40) crops into one class and
trains a single-class subset; the model proves the pipeline, not accuracy. A project
slug is retired once deleted, so use a fresh `--project` per run; the JSON report has per-phase status, seconds
and evidence. Attach `acceptance.json` to the go/no-go record. Screenshot review
(plan phase K) stays manual.

## Steps

1. Pre-flight. Confirm `VERSION` is `0.4.1`, the `CHANGELOG.md` `[0.4.1]` date is the
   release day (edit if not), and the tree is clean on the branch to release.
   Re-read `docs/releases/v0.4.1.md` and `docs/design/README.md` (plan statuses) against what is merged, and run the full gate: `pytest tests/ -q --ignore=tests/live -n 16`,
   `.venv/bin/pre-commit run --all-files`,
   `.venv/bin/python scripts/codegen/generate_contracts.py --check`.
2. Prove the asset set: `make release-verify` (runs `scripts/release/verify_release_assets.sh`;
   it touches no running stack; on a host with a live stack run it as
   `OP_PROJECT=op-relverify make release-verify` so the project name does not collide;
   builds the assets to a temp dir, dry-runs the installer against them, checks
   that a tampered tarball and installer are refused).
3. Build, Trivy-scan and dry-run the images, nothing pushed: `make release-dry-run`.
   Trivy CRITICAL gate must pass (allowlist: `scripts/release/trivy-allowlist.txt`).
4. Push images and pin them: `docker login`, pull the 11 third-party images, then
   `make release`. It pushes `davidamacey/openprocessor{,-triton,-evaluator,-segmenter,-trainer}:0.4.1`
   (plus `latest`), resolves the third-party digests, and writes `images.lock` and
   `images.lock.sha256`. Check `rg 'dev[0-9a-f]$' images.lock` finds nothing.
5. Commit the lock: `git add images.lock` then
   `git commit -m "chore(release): pin v0.4.1 image digests"`. (`images.lock.sha256` is a
   local artifact, not committed.)
6. Build the release assets from the committed tree:
   `CW_RELEASE_DIR=<dir> scripts/release/build_deploy_bundle.sh v0.4.1`. For v0.4.1 the
   Cropwright files named in `cropwright.lock` MUST be staged: they ship inside the tarball
   (`cropwright-release/v0.1.1/`) because the standalone Cropwright repository is private
   until the 0.5.0 monorepo. Cropwright's `docs` service is behind compose profile `docs`;
   the installer does not enable it, so `/cropwright/` docs 502 until the profile is on
   (documented in the installer docs and release notes). It refuses an unpinned lock. Output:
   `dist/release-v0.4.1/` holding `openprocessor-deploy-v0.4.1.tar.gz`, `SHA256SUMS`,
   `setup-openprocessor.sh`, `release-manifest.txt`.
7. Check the real assets before publishing:
   `cd dist/release-v0.4.1 && sha256sum --check --ignore-missing SHA256SUMS` (the tarball
   and installer lines; the other lines name files inside the tarball).
8. Merge to the release branch with a merge commit (`--no-ff`, never squash), then push it.
9. Tag: `git tag -a v0.4.1 -m "OpenProcessor v0.4.1"` then `git push origin v0.4.1`.
   (`make release` refuses a tag that is not at HEAD, so tag after step 5.)
10. Publish (immediately, not a draft):
    `gh release create v0.4.1 --title "OpenProcessor v0.4.1" --notes-file docs/releases/v0.4.1.md dist/release-v0.4.1/openprocessor-deploy-v0.4.1.tar.gz dist/release-v0.4.1/SHA256SUMS dist/release-v0.4.1/setup-openprocessor.sh dist/release-v0.4.1/release-manifest.txt`
11. GitHub org transfer: the docs name no transfer plan. Every link and installer URL uses
    `github.com/davidamacey/OpenProcessor`. If the repo moves, update those URLs
    (`rg 'davidamacey/OpenProcessor'`) and re-run `check_docs_vs_code.py` before step 10.

## Post-release verification

- `gh release view v0.4.1` lists the four assets and is not a draft.
- On a scratch directory: `curl -fsSLO` the release `setup-openprocessor.sh` and
  `SHA256SUMS`, `sha256sum --check --ignore-missing`, then
  `bash setup-openprocessor.sh --version v0.4.1 --dry-run --dir /tmp/op-verify`.
- On a scratch host or `--dir`/`--project` pair that does not touch a live stack:
  real install with `--version v0.4.1`, health check passes.
- `docker pull davidamacey/openprocessor:0.4.1` digest equals the `api` line in `images.lock`.
- Docker Hub shows the five repositories at `0.4.1` and `latest`.
