# v0.5.0 release checklist

Every step is manual and owner-run. Run from a clean checkout of the release
commit. Nothing here is automated by CI (images are built locally).

## images.lock: what is pinned, and what `make release` changes

`images.lock` pins every image an install can pull, one `key=repo:tag@sha256:digest`
line per key. The keys come from `scripts/lib/image_keys.sh`, the one table the release
script and the installer share. After `make release` there are 17 lines:

- Six images built from this repository and pushed by step 4
  (`davidamacey/...:0.5.0`, plus `latest`): `api` (`openprocessor`), `triton`
  (`openprocessor-triton`), `evaluator` (`openprocessor-evaluator`), `segmenter`
  (`openprocessor-segmenter`), `trainer` (`openprocessor-trainer`) and `cropwright`
  (`cropwright`, built from `frontend/Dockerfile` with `frontend/` as its build context).
- Eleven third-party images, resolved from upstream tags by step 4: the two `vlm_*` keys,
  `opensearch`, `opensearch_dashboards`, `mlflow`, `prometheus`, `grafana`, `loki`,
  `alloy`, `node_exporter`, `dcgm_exporter`. Each tag must be pulled locally first
  (`docker pull <image>`), because the script reads its RepoDigest.

The committed `images.lock` still holds the 0.4.1 digests (16 lines, no `cropwright` row)
until step 4 rewrites it entirely; do not hand-edit it, and do not bump it ahead of the
release. In lock mode the installer dies with "images.lock has no 'cropwright' entry" for
every tier, because it requires a row for each build key, so a lock-mode install of this
branch is impossible until step 4. That is expected: `make release-verify` builds its bundle
with `ALLOW_UNPINNED_LOCK=1` and installs by `--image-tag`, and the installer tests pass with the
committed lock as it is. The version-consistency test does not compare lock tags with `VERSION`;
the bundle builder checks only that every line is digest-pinned and not `latest`.

Cropwright is no longer released from its own repository or its own lock. There is no
`cropwright.lock`, no `CW_RELEASE_DIR` staging and no `cropwright-release/` directory in the
tarball; do not run `frontend/scripts/release/*` for 0.5.0. The `cropwright` tier is a compose
profile of the main project and takes its image from `images.lock`.

## Release decisions (#44, #63): all DECIDED

Carried forward from 0.4.1; nothing here changes for 0.5.0 unless noted.

- **`:latest` tag: DECIDED.** v0.5.0 is the numbered release and `latest` is published
  together with it, so `latest` equals `0.5.0` at release for all six images, including
  `davidamacey/cropwright` (whose `latest` previously followed the standalone 0.1.x line).
  `make release` pushes `:0.5.0` and `:latest` for each image, and refuses a non-X.Y.Z
  version, so `latest` can only ever point at a stable release. A backport or an old line is
  pushed by hand with `docker push` of the version tag only. The installer pins digests from
  `images.lock` and never uses `latest`.
- **Control-plane-only mode: DECIDED.** Kept as an optional installer flag
  (`--cpu --control-plane-only`), never the default and not part of the supported
  production path. It starts only OpenSearch and the API (plus Cropwright if asked).
- **Unverified VLM catalog entries: DECIDED.** Kept, marked `unverified/experimental`; never
  auto-selected (explicit `--vlm-model-id` only).
- **OpenSearch heap and shard budget: DECIDED.** Default heap is half of host RAM, 2 GB
  minimum, 30 GB cap; each project costs 6 shards; the soft budget is heap GB x 20
  (`OP_SHARDS_PER_HEAP_GB`). See `docs-site/docs/deployment/sizing-and-storage.mdx`.
- **Docs hosting: DECIDED.** Both. A public GitHub Pages site built from `docs-site/` by
  `.github/workflows/docs.yml`, and a local docs container
  (`docker compose -f docker-compose.yml -f docker-compose.docs.yml up -d --build docs`,
  port 4613) for source checkouts. The Cropwright docs are a section of this one site. The
  docs image is not in `images.lock` or `make release`. Owner one-time steps if not already
  done: Settings > Pages > Source "GitHub Actions"; allow `main` on the `github-pages`
  environment; run the workflow by hand once from `main`.
- **Trivy remainder (#110): DECIDED.** `linux-libc-dev` kernel-header CVEs and libxml2
  `CVE-2026-6653` are owner-accepted and allowlisted per CVE with a reason in
  `scripts/release/trivy-allowlist.txt`; re-check each release and drop entries when a fixed
  package ships. The mlflow 2.x entries are gone (MLflow 3.17, #120).

## Acceptance status and open items

- Decided: Grafana keeps the default admin password for LAN-only installs; the docs warn to
  change it before exposing the stack beyond the LAN. Gateway mode refuses the default.
- The API has no authentication (documented in `SECURITY.md`).
- #44 (clean-host install from the published artifacts) stays open until run against the
  published 0.5.0 assets.
- The 430 px UI fix (#184) is covered by the stubbed e2e suite only.
- #119 and #61 live acceptance: record the results in `docs/releases/v0.5.0.md` (every
  `TODO-LIVE` marker there must be replaced before step 10; `rg TODO-LIVE docs/releases`
  must find nothing).
- No open decisions remain except the owner's go to publish.

## Scripted acceptance run (release gate)

Install the release candidate with the installer into an ISOLATED compose project and
install directory (non-default `--port-base`, GPUs nothing else uses; tests use GPUs 0 and 2),
then run:

```bash
python3 scripts/datasets/fetch_coco_subset.py --out data/samples/coco_va   # public COCO, once
scripts/release/acceptance_run.sh \
  --base-url http://localhost:<port> --project-name <compose-project> \
  --install-dir <install-dir> --models-dir <install-dir>/models \
  --project acc-v050 --images-dir data/samples/coco_va --report acceptance.json
# or: make acceptance ACC_ARGS="<the same flags>"
```

It exercises health, self-hosted docs, ingest of ~100 COCO images, clustering, VLM
labeling, human confirmation, holdout, export, a 2-epoch probe train, bake-off, promote
(forced when the gate rejects the undertrained model), inference via `/detect?model_name=`,
model delete, metrics and log-noise checks, and the installer lifecycle (`--only
installer_uninstall` and `--upgrade-version vX` are opt-in). A cleanup trap always deletes
the throwaway project and promoted models. Exit status is non-zero if any required phase
failed. A project slug is retired once deleted, so use a fresh `--project` per run (`acc-v050`
for this release); the JSON report has per-phase status, seconds and evidence. Attach
`acceptance.json` to the go/no-go record. Screenshot review (plan phase K) stays manual:
open the full-page desktop and 430 px captures and look at them. For 0.5.0 also exercise
label confirmation by hand (VLM scope `off`, an audit draw and report, the
`detector_disagreements` tab) and the gateway (`OP_GATEWAY_SUBPATHS=true` on loopback), and
write the results into the `TODO-LIVE` lines of `docs/releases/v0.5.0.md`.

## Steps

1. Pre-flight. Confirm `VERSION` is `0.5.0` (also `pyproject.toml`, `docs-site/package.json`
   and its lock, `frontend/package.json` and its lock, the `OP_IMAGE_TAG` defaults in
   `docker-compose.yml`, `SCRIPT_VERSION` in `setup-openprocessor.sh`, the CLI default and the
   roadmap; `tests/test_version_consistency.py` checks most of them), the `CHANGELOG.md` and
   `frontend/CHANGELOG.md` `[0.5.0]` dates are the release day (edit if not), and the tree is
   clean on the branch to release. Re-read `docs/releases/v0.5.0.md` and `docs/design/README.md`
   (plan statuses) against what is merged, and run the full gate:
   `.venv/bin/python -m pytest tests/ -q --ignore=tests/live -n 16`,
   `.venv/bin/pre-commit run --all-files`,
   `.venv/bin/python scripts/codegen/generate_contracts.py --check`, and from `frontend/`
   `npm ci && npm run check && npm test && npm run build`, plus `npm run test:e2e` (stubbed).
   Also `cd docs-site && npm ci && npm run build` (it fails on a broken link).
2. Prove the asset set: `make release-verify` (runs `scripts/release/verify_release_assets.sh`;
   it touches no running stack; on a host with a live stack run it as
   `OP_PROJECT=op-relverify make release-verify` so the project name does not collide;
   builds the assets to a temp dir, dry-runs the installer against them, checks that a
   tampered tarball and installer are refused).
3. Build, Trivy-scan and dry-run the images, nothing pushed: `make release-dry-run`. It
   builds all six images, including Cropwright from `frontend/`. The Trivy CRITICAL gate must
   pass (allowlist: `scripts/release/trivy-allowlist.txt`).
4. Push images and pin them: `docker login`, pull the 11 third-party images, then
   `make release`. It pushes
   `davidamacey/{openprocessor,openprocessor-triton,openprocessor-evaluator,openprocessor-segmenter,openprocessor-trainer,cropwright}:0.5.0`
   (plus `latest`), resolves the third-party digests, and writes `images.lock` (17 lines) and
   `images.lock.sha256`. Check `rg 'dev[0-9a-f]$' images.lock` finds nothing and that
   `rg -c . images.lock` is 17 with a `cropwright=` line.
5. Commit the lock: `git add images.lock` then
   `git commit -m "chore(release): pin v0.5.0 image digests"`. (`images.lock.sha256` is a
   local artifact, not committed.)
6. Build the release assets from the committed tree:
   `scripts/release/build_deploy_bundle.sh v0.5.0`. It refuses an unpinned lock and needs no
   Cropwright files staged: Cropwright is in `images.lock`. Output: `dist/release-v0.5.0/`
   holding `openprocessor-deploy-v0.5.0.tar.gz`, `SHA256SUMS`, `setup-openprocessor.sh`,
   `release-manifest.txt`.
7. Check the real assets before publishing:
   `cd dist/release-v0.5.0 && sha256sum --check --ignore-missing SHA256SUMS` (the tarball
   and installer lines; the other lines name files inside the tarball).
8. Merge to the release branch with a merge commit (`--no-ff`, never squash), then push it.
9. Tag: `git tag -a v0.5.0 -m "OpenProcessor v0.5.0"` then `git push origin v0.5.0`.
   (`make release` refuses a tag that is not at HEAD, so tag after step 5.)
10. Publish (immediately, not a draft):
    `gh release create v0.5.0 --title "OpenProcessor v0.5.0" --notes-file docs/releases/v0.5.0.md dist/release-v0.5.0/openprocessor-deploy-v0.5.0.tar.gz dist/release-v0.5.0/SHA256SUMS dist/release-v0.5.0/setup-openprocessor.sh dist/release-v0.5.0/release-manifest.txt`
11. After the release: move the repository to a GitHub organisation. Do this only once the
    release is published, so 0.5.0 ships with the URLs the tag was built with; GitHub
    redirects the old repository URL, but Pages, Docker Hub and the raw installer URL do not
    all follow. The target organisation is not named in this tree. Work through every place
    that carries the owner, in a follow-up change, then re-run
    `check_docs_vs_code.py` and `scripts/codegen/check_naming_leaks.py`:
    - `rg 'davidamacey/OpenProcessor'` for repository URLs: `setup-openprocessor.sh` (the
      raw installer URL and release download URLs), `INSTALLATION.md`, `README.md`,
      `SECURITY.md`, `CONTRIBUTING.md`, `docs-site/site.config.ts` (`githubRepo`, `url`,
      `baseUrl`), `docs-site/docs/**`, `docs-site/src/data/roadmap.json` (`repo`),
      `.github/ISSUE_TEMPLATE/config.yml` (security advisory link), the OCI `source` labels in
      the Dockerfiles, `frontend/package.json`, and the tests that assert the URL
      (`tests/installer/`, `tests/test_release_build_and_publish.py`).
    - Docker Hub: the images are named `davidamacey/<name>` (`OP_IMAGE_NAMESPACE` in
      `scripts/release/build_and_publish.sh`, the `OP_IMAGE_REPO` defaults in the compose
      files, `scripts/lib/image_keys.sh` consumers, `scripts/docker-build-push.sh`). A new
      namespace needs the six repositories created, the images re-pushed by a new `make
      release`, and a fresh `images.lock` (the installer checks each lock entry's repo against
      the table). Old tags stay pullable under the old name.
    - `CODEOWNERS` (a user handle becomes an organisation team) and `.github/dependabot.yml`
      (reviewers, assignees).
    - Repository secrets and variables (Docker Hub token, any Actions secrets) and the
      `github-pages` environment, which do not transfer with every setting; re-create them
      and re-run the docs workflow by hand so Pages serves from the new location.
    - GitHub Pages: the published site URL changes with the owner, so update every link to it
      and `site.config.ts` together.
    - Local clones and the `origin` remote of any release worktree.

## Post-release verification

- `gh release view v0.5.0` lists the four assets and is not a draft.
- On a scratch directory: `curl -fsSLO` the release `setup-openprocessor.sh` and
  `SHA256SUMS`, `sha256sum --check --ignore-missing`, then
  `bash setup-openprocessor.sh --version v0.5.0 --dry-run --dir /tmp/op-verify`.
- On a scratch host or `--dir`/`--project` pair that does not touch a live stack: real
  install with `--version v0.5.0 --all`, health check passes, Cropwright answers on its port
  (this is the clean-host test of #44; close it when it passes).
- `docker pull davidamacey/openprocessor:0.5.0` digest equals the `api` line in
  `images.lock`, and `docker pull davidamacey/cropwright:0.5.0` equals the `cropwright` line.
- Docker Hub shows the six repositories at `0.5.0` and `latest`.
- Upgrade check on a 0.4.1 install: `./openprocessor upgrade` replaces the `yolo-api`
  container with `api` (no name-conflict error) and `./openprocessor status` is healthy.
