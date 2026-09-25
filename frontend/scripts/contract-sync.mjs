#!/usr/bin/env node
/**
 * Vendor OpenProcessor's generated API contract files into
 * `contracts/openprocessor/` so the frontend's contract tests
 * (`src/lib/contract/*.test.ts`) can assert against the backend's real
 * wire format instead of a hand-copied fact.
 *
 * The backend (OpenProcessor) generates these under
 * `contracts/` on its own repo and gates them with a pre-commit `--check`
 * hook (see `contracts/README.md` there) — this script is the frontend
 * half of that pattern: vendor at a known revision, and fail loudly when
 * the vendored copy drifts from the backend ref.
 *
 * Usage:
 *   node scripts/contract-sync.mjs          # sync (overwrite vendored copies)
 *   node scripts/contract-sync.mjs --check  # exit 1 if the vendored copies differ
 *
 * Env:
 *   OPENPROCESSOR_REPO   path to a local OpenProcessor checkout (default:
 *                        ../OpenProcessor, relative to this repo's root)
 *   OPENPROCESSOR_REF    git ref to read from (default: main)
 *   OPENPROCESSOR_URL    public repo URL recorded in SOURCE.md (default:
 *                        https://github.com/davidamacey/OpenProcessor). Only
 *                        the URL and the sha are recorded, never a local path.
 */
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..');
const vendorRoot = path.join(repoRoot, 'contracts', 'openprocessor');

const OP_REPO = path.resolve(
  repoRoot,
  process.env.OPENPROCESSOR_REPO || '../OpenProcessor',
);
const OP_REF = process.env.OPENPROCESSOR_REF || 'main';
const OP_URL =
  process.env.OPENPROCESSOR_URL || 'https://github.com/davidamacey/OpenProcessor';

/** [backend-relative path (under contracts/), vendored-relative path]. Same
 *  subpath on both sides — only the `contracts/` vs `contracts/openprocessor/`
 *  root differs — so adding a new vendored file is a one-line change. */
// A git hook runs with GIT_DIR/GIT_INDEX_FILE/GIT_WORK_TREE pointing at
// THIS repo (always, in a worktree), and `git -C` does not override them,
// so without this every call below silently reads the frontend repo.
const GIT_ENV = Object.fromEntries(
  Object.entries(process.env).filter(
    ([k]) =>
      !['GIT_DIR', 'GIT_INDEX_FILE', 'GIT_WORK_TREE', 'GIT_COMMON_DIR'].includes(k),
  ),
);

const FILES = [
  'json/item_wire.json',
  'ts/itemWire.ts',
  'ts/classSources.ts',
  'ts/regionStatus.ts',
  'openapi/curation.json',
];

function backendAvailable() {
  try {
    execFileSync('git', ['-C', OP_REPO, 'rev-parse', '--git-dir'], {
      stdio: 'ignore',
      env: GIT_ENV,
    });
    return true;
  } catch {
    return false;
  }
}

function readBackendFile(relPath) {
  return execFileSync('git', ['-C', OP_REPO, 'show', `${OP_REF}:contracts/${relPath}`], {
    encoding: 'utf-8',
    maxBuffer: 1024 * 1024 * 16,
    env: GIT_ENV,
  });
}

function backendSha() {
  return execFileSync('git', ['-C', OP_REPO, 'rev-parse', OP_REF], {
    encoding: 'utf-8',
    env: GIT_ENV,
  }).trim();
}

function sourceMd(sha) {
  return `# Contract source

Vendored from OpenProcessor via \`git -C <repo> show <ref>:contracts/...\`.

- repo: \`${OP_URL}\`
- ref: \`${OP_REF}\`
- sha: \`${sha}\`
- synced_at: \`${new Date().toISOString()}\`

Regenerate with \`npm run contract:sync\`. Verify with \`npm run contract:check\`
(non-blocking in CI — the backend repo isn't checked out there).

Do not hand-edit the files in this directory; they are generated on the
backend and copied verbatim. See \`contracts/README.md\` on the backend repo
for what each file is and where it comes from.
`;
}

function writeAll(destRoot, sha) {
  for (const rel of FILES) {
    const content = readBackendFile(rel);
    const dest = path.join(destRoot, rel);
    mkdirSync(path.dirname(dest), { recursive: true });
    writeFileSync(dest, content);
  }
  writeFileSync(path.join(destRoot, 'SOURCE.md'), sourceMd(sha));
}

function sync() {
  const sha = backendSha();
  writeAll(vendorRoot, sha);
  console.log(`Synced contracts from ${OP_REPO} @ ${OP_REF} (${sha}) into ${vendorRoot}`);
}

/** Compare SOURCE.md by its `sha:` line only — synced_at always differs. */
function sourceShaLine(text) {
  return (text.match(/- sha: `([^`]+)`/) || [])[1] ?? null;
}

function check() {
  const sha = backendSha();
  const tmp = mkdtempSync(path.join(tmpdir(), 'contract-sync-'));
  try {
    writeAll(tmp, sha);
    const diffs = [];
    for (const rel of FILES) {
      const a = readFileSync(path.join(vendorRoot, rel), 'utf-8');
      const b = readFileSync(path.join(tmp, rel), 'utf-8');
      if (a !== b) diffs.push(rel);
    }
    let existingSourceMd;
    try {
      existingSourceMd = readFileSync(path.join(vendorRoot, 'SOURCE.md'), 'utf-8');
    } catch {
      existingSourceMd = '';
    }
    if (sourceShaLine(existingSourceMd) !== sha) diffs.push('SOURCE.md (sha)');

    if (diffs.length > 0) {
      console.error('Vendored contracts are out of sync with the backend:');
      for (const d of diffs) console.error(`  - ${d}`);
      console.error('\nRun `npm run contract:sync` to refresh, then review the diff.');
      process.exitCode = 1;
    } else {
      console.log(`Vendored contracts match ${OP_REPO} @ ${OP_REF} (${sha}).`);
    }
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

const mode = process.argv.includes('--check') ? 'check' : 'sync';

if (!backendAvailable()) {
  console.log(
    `contract:${mode}: skipping — backend repo not found at ${OP_REPO} ` +
      `(set OPENPROCESSOR_REPO to override). This is expected in CI.`,
  );
  process.exit(0);
}

if (mode === 'check') check();
else sync();
