# Installer drive log (v0.7.0 hardening pass)

Hand-driven runs of `setup-openprocessor.sh` from the `release/0.7.0` branch against the published
v0.5.0 and v0.6.0 release assets. No GPU container was started: every run that starts anything uses
`--cpu --control-plane-only`. Each drive used a fresh `<scratch dir>`, its own compose project name
and a shifted `--port-base`, and was removed afterwards. Times are wall clock on a machine with
images already pulled (a cold pull adds the download time).

## Results

| # | Drive | Result | Time |
|---|---|---|---|
| 1 | `--cpu --control-plane-only --unattended`, health, uninstall | Found two defects (below); passes after the fixes. Uninstall keeps the named volumes, as documented | install 4m17s (all spent waiting for a health state that cannot happen), repair 8s, uninstall 12s |
| 2 | `--no-start`, then `./openprocessor start` | Passes after fix: start no longer brings up Triton | install 6s, start 41s |
| 3 | `--branch release/0.7.0 --no-start` | Resolves the pushed head SHA, writes `mode: branch`, `image_mode: tag`, tag `sha-<12 hex>`. Needs images tagged `sha-<12 hex>` (CI publishes them for branch builds); without them it stops with "neither local nor pullable". With a locally tagged image the file and config path completes | 10s / 6s |
| 4 | Install v0.5.0 `--no-start`, upgrade to v0.6.0, `--rollback --yes` | Files and `.env` return byte-for-byte (including a hand-added key); backup kept in `backups/`, replaced files in `.install/rollback-undo/`. Found the in-place script overwrite defect (below) | upgrade 6s, rollback 53s |
| 5 | Edge cases | See the table below | |
| 6 | Message review | See "Messages" | |

### Drive 5 edge cases

| Case | Exit | Message |
|---|---|---|
| unknown flag | 2 | `unknown flag: --bogus`, then usage |
| flag missing its value | 2 | `--dir needs a value` |
| bad `--project` / bad `--version` | 2 | names the rule and the value |
| public `--bind` or external `--vlm-remote`, no terminal, no env consent | 9 | names the env var to set |
| uninstall or upgrade, no terminal, no `--yes` | 9 | says a missing terminal is not consent |
| non-empty dir, no `--force-existing-dir` | 3 | nothing changed; names the flag |
| uninstall in a dir this installer did not create | 3 | no `.install/state.json`; nothing changed |
| unwritable parent dir | 1 | `cannot create <dir>` |
| `--cpu` alone | 4 | now says `--cpu on its own cannot work` |
| low disk (`df` stub, `--dry-run`) | 1 | `need ~N GB free ... have 4 GB (use --force ...)` |
| two installers in one dir at once | was 0 and 1 | see defect 4; now 3 |
| SIGINT mid-run, then re-run | killed by the signal | re-run resumes cleanly; no message on interrupt (open) |

## Defects found and fixed

1. **Control-plane-only install always ended `failed` (exit 8) and waited 4 minutes first.** The API's
   `/health` answers 503 without Triton, so the "API up but not ready" fallback, which used a
   `curl -f` call, could never succeed. The installer now waits for a `/health` body with OpenSearch
   ok, accepting any HTTP status.
2. **`./openprocessor start` after a control-plane-only install started Triton.** The overlay only
   dropped the API's GPU reservation, so a plain `up -d` started every default service. The overlay
   now disables `triton-server`, removes the API's dependency on it, and probes `/` for the container
   healthcheck (the API was reported `unhealthy` forever).
3. **Rollback and upgrade rewrote the running installer in place.** Bash reads a script as it runs;
   after a rollback to a different release the shell resumed at its old byte offset inside the new
   text and executed a fragment of it (here: it started the whole stack, including Triton, after
   `--rollback --no-start`). Release files are now replaced by rename, so the running copy keeps its
   old file.
4. **No lock between installer runs.** Two runs in one dir raced on `.install/staging` (one died with
   a `cp` error). Mutating runs now take `flock` on `.install/lock` and the second exits 3.
5. **`--rollback` ignored `--no-start`.** It now restores the files and starts nothing.
6. **`secrets/vlm` and `data/source` were left for Docker to create.** Docker makes a missing bind
   source root-owned, which blocked `openprocessor vlm key set` and removing the install dir without
   `sudo`. The installer now creates both as the invoking user (`secrets/` mode 700).

Message fixes: `--cpu` alone no longer claims there is no GPU; a bad `--version` is refused before the
preflight; the nvidia-runtime line and the "Triton image alone is ~30 GB" line are not printed for a
control-plane-only install; a branch install says "downloaded" rather than "downloaded and verified"
(branch files are not checksum-verified); the `--no-start` hint omits `models install` when there are
no models; the low-disk error names the install dir, not the dry-run scratch copy.

## Messages a first-time user may find confusing (not changed)

- Summary and `status` print `API health http://...:<port>/health` for a control-plane-only install,
  where it is 503 by design (the NOTE line explains it below the URLs).
- `./openprocessor status` probes the default Triton (4600) and Grafana (4605) ports regardless of
  the install's tiers or `--port-base`, so on a shared host it can report another stack's service;
  a failed probe prints `HTTP 000000`.
- The 0.5.0 installer, when run through `./openprocessor upgrade`, logs `action install`.
- Branch installs print the 40-hex SHA as the release name.
- A dry run checks free disk on its temporary scratch copy, not on the target filesystem.
- Ctrl-C prints nothing; the next run resumes.
