# v0.4.0 release checklist

Every step is manual and owner-run. Run from a clean checkout of the release
commit. Nothing here is automated by CI (images are built locally).

## images.lock: what is a placeholder

All 16 entries in `images.lock` are development placeholders (`...dev0` to `...devf`,
not 64-hex digests) and must all be filled; the installer refuses them.

- Built and pushed by step 4 (`davidamacey/...:0.4.0`): `api`, `triton`, `evaluator`,
  `segmenter`, `trainer`.
- Third-party, resolved from upstream tags by step 4: the two `vlm_*` keys,
  `opensearch`, `opensearch_dashboards`, `mlflow`, `prometheus`, `grafana`, `loki`,
  `alloy`, `node_exporter`, `dcgm_exporter`. Each tag must be pulled locally first
  (`docker pull <image>`), because the script reads its RepoDigest.

`make release` rewrites `images.lock` entirely; do not hand-edit digests.

## Steps

1. Pre-flight. Confirm `VERSION` is `0.4.0`, the `CHANGELOG.md` `[0.4.0]` date is the
   release day (edit if not), and the tree is clean on the branch to release.
   Run the full gate: `pytest tests/ -q --ignore=tests/live -n 16`,
   `.venv/bin/pre-commit run --all-files`,
   `.venv/bin/python scripts/codegen/generate_contracts.py --check`.
2. Prove the asset set: `scripts/release/verify_release_assets.sh v0.4.0`
   (builds the assets to a temp dir, dry-runs the installer against them, checks
   that a tampered tarball and installer are refused).
3. Build, Trivy-scan and dry-run the images, nothing pushed: `make release-dry-run`.
   Trivy CRITICAL gate must pass (allowlist: `scripts/release/trivy-allowlist.txt`).
4. Push images and pin them: `docker login`, pull the 11 third-party images, then
   `make release`. It pushes `davidamacey/openprocessor{,-triton,-evaluator,-segmenter,-trainer}:0.4.0`
   (plus `latest`), resolves the third-party digests, and writes `images.lock` and
   `images.lock.sha256`. Check `rg 'dev[0-9a-f]$' images.lock` finds nothing.
5. Commit the lock: `git add images.lock` then
   `git commit -m "chore(release): pin v0.4.0 image digests"`. (`images.lock.sha256` is a
   local artifact, not committed.)
6. Build the release assets from the committed tree:
   `scripts/release/build_deploy_bundle.sh v0.4.0` (add `CW_RELEASE_DIR=<dir>` to stage the
   Cropwright files named in `cropwright.lock`). It refuses an unpinned lock. Output:
   `dist/release-v0.4.0/` holding `openprocessor-deploy-v0.4.0.tar.gz`, `SHA256SUMS`,
   `setup-openprocessor.sh`, `release-manifest.txt`.
7. Check the real assets before publishing:
   `cd dist/release-v0.4.0 && sha256sum --check --ignore-missing SHA256SUMS` (the tarball
   and installer lines; the other lines name files inside the tarball).
8. Merge to the release branch with a merge commit (`--no-ff`, never squash), then push it.
9. Tag: `git tag -a v0.4.0 -m "OpenProcessor v0.4.0"` then `git push origin v0.4.0`.
   (`make release` refuses a tag that is not at HEAD, so tag after step 5.)
10. Publish (immediately, not a draft):
    `gh release create v0.4.0 --title "OpenProcessor v0.4.0" --notes-file docs/releases/v0.4.0.md dist/release-v0.4.0/openprocessor-deploy-v0.4.0.tar.gz dist/release-v0.4.0/SHA256SUMS dist/release-v0.4.0/setup-openprocessor.sh dist/release-v0.4.0/release-manifest.txt`
11. GitHub org transfer: the docs name no transfer plan. Every link and installer URL uses
    `github.com/davidamacey/OpenProcessor`. If the repo moves, update those URLs
    (`rg 'davidamacey/OpenProcessor'`) and re-run `check_docs_vs_code.py` before step 10.

## Post-release verification

- `gh release view v0.4.0` lists the four assets and is not a draft.
- On a scratch directory: `curl -fsSLO` the release `setup-openprocessor.sh` and
  `SHA256SUMS`, `sha256sum --check --ignore-missing`, then
  `bash setup-openprocessor.sh --version v0.4.0 --dry-run --dir /tmp/op-verify`.
- On a scratch host or `--dir`/`--project` pair that does not touch a live stack:
  real install with `--version v0.4.0`, health check passes.
- `docker pull davidamacey/openprocessor:0.4.0` digest equals the `api` line in `images.lock`.
- Docker Hub shows the five repositories at `0.4.0` and `latest`.
