#!/usr/bin/env node
/**
 * `npm run test:e2e` — creates or reuses the e2e/.venv, installs the pinned
 * Python deps + a chromium browser if missing, then runs the stubbed
 * pytest+Playwright suite. The suite's own `app_url` fixture
 * (e2e/conftest.py) does `npm run build` + `vite preview`, so this script
 * does not build itself.
 *
 * See docs/design/test-audit-2026-09-24.md recommendation 5 and CLAUDE.md's
 * "Development" section.
 */
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const ROOT = new URL('..', import.meta.url).pathname;
const VENV = join(ROOT, 'e2e', '.venv');
const VENV_PY = join(VENV, 'bin', 'python3');
const VENV_PIP = join(VENV, 'bin', 'pip');
const VENV_PYTEST = join(VENV, 'bin', 'pytest');
const REQUIREMENTS = join(ROOT, 'e2e', 'requirements.txt');
const STAMP = join(VENV, '.requirements.sha256');

function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, { stdio: 'inherit', cwd: ROOT, ...opts });
  if (res.status !== 0) {
    process.exit(res.status ?? 1);
  }
}

function sha256(text) {
  // No extra deps: shell out to a stable hash rather than pulling in
  // node:crypto ceremony for one string.
  const res = spawnSync('sha256sum', [], { input: text, encoding: 'utf8' });
  return res.stdout.trim().split(/\s+/)[0];
}

if (!existsSync(VENV_PY)) {
  console.log('[test:e2e] creating e2e/.venv …');
  let python3 = 'python3.12';
  if (spawnSync(python3, ['--version']).status !== 0) python3 = 'python3';
  run(python3, ['-m', 'venv', VENV]);
}

const reqHash = sha256(readFileSync(REQUIREMENTS, 'utf8'));
const stamped = existsSync(STAMP) ? readFileSync(STAMP, 'utf8').trim() : null;
if (stamped !== reqHash) {
  console.log('[test:e2e] installing e2e/requirements.txt …');
  run(VENV_PIP, ['install', '-q', '--upgrade', 'pip']);
  run(VENV_PIP, ['install', '-q', '-r', REQUIREMENTS]);
  run('sh', ['-c', `printf '%s' "${reqHash}" > "${STAMP}"`]);
}

// A cheap existence check — playwright's own cache lives outside the venv
// (~/.cache/ms-playwright), so a fresh venv doesn't always mean a missing
// browser.
const cacheGlobRoot = join(homedir(), '.cache', 'ms-playwright');
const hasChromium =
  existsSync(cacheGlobRoot) &&
  spawnSync('sh', ['-c', `ls "${cacheGlobRoot}" | grep -q '^chromium-'`]).status === 0;
if (!hasChromium) {
  console.log('[test:e2e] installing chromium …');
  run(VENV_PY, ['-m', 'playwright', 'install', 'chromium']);
}

console.log('[test:e2e] running pytest e2e/stubbed …');
run(VENV_PYTEST, ['-c', 'e2e/pytest.ini', 'e2e/stubbed', '-v']);
