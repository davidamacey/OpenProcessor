#!/usr/bin/env node
/**
 * `npm run test:e2e` — creates or reuses the e2e/.venv, installs the pinned
 * Python deps + a chromium browser if missing, then runs the stubbed
 * pytest+Playwright suite. The suite's own `app_url` fixture
 * (e2e/conftest.py) does `npm run build` + `vite preview`, so this script
 * does not build itself.
 *
 * `npm run test:live` reuses this exact venv/browser setup, just pointed
 * at `e2e/live` instead — pass the target directory as argv[2] (see
 * package.json). The live tier never builds/serves anything itself; it
 * talks to whatever `CROPWRIGHT_LIVE_URL` points at (its own conftest.py
 * skips the whole tier when that's unset), and gets `--screenshot=
 * only-on-failure` so a failure leaves a screenshot in
 * artifacts_local/cw-live/live-tier/ instead of nothing.
 *
 * See docs/design/test-audit-2026-09-24.md recommendation 5 and CLAUDE.md's
 * "Development" section.
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { availableParallelism, homedir } from 'node:os';
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

const target = process.argv[2] || 'e2e/stubbed';
const isLive = target === 'e2e/live';
const pytestArgs = ['-c', 'e2e/pytest.ini', target, '-v'];
if (isLive) {
  pytestArgs.push(
    '--screenshot=only-on-failure',
    '--output=artifacts_local/cw-live/live-tier',
  );
}

if (isLive || process.env.E2E_APP_URL) {
  console.log(`[test:e2e] running pytest ${target} …`);
  run(VENV_PYTEST, pytestArgs);
} else {
  // Build and serve once here, then fan the stubbed suite out across xdist
  // workers. Left to conftest's session fixture, every worker would run its
  // own build and preview server. Every test stubs its own page, so the
  // tests share nothing but this static server.
  console.log('[test:e2e] building …');
  run('npm', ['run', '-s', 'build']);
  const port = await freePort();
  const preview = spawn(
    'npx',
    ['vite', 'preview', '--port', String(port), '--strictPort'],
    {
      cwd: ROOT,
      stdio: 'ignore',
    },
  );
  const stop = () => preview.kill('SIGTERM');
  process.on('exit', stop);
  process.on('SIGINT', () => process.exit(130));
  const url = `http://localhost:${port}`;
  await waitForServer(url, preview);
  const workers =
    process.env.E2E_WORKERS ??
    String(Math.min(6, Math.max(1, Math.floor(availableParallelism() / 2))));
  console.log(
    `[test:e2e] running pytest ${target} with ${workers} workers against ${url} …`,
  );
  const res = spawnSync(
    VENV_PYTEST,
    [...pytestArgs, '-n', workers, '--dist', 'loadfile'],
    {
      stdio: 'inherit',
      cwd: ROOT,
      env: { ...process.env, E2E_APP_URL: url },
    },
  );
  stop();
  process.exit(res.status ?? 1);
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.once('error', reject);
    srv.listen(0, () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

async function waitForServer(url, proc, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (proc.exitCode !== null) {
      console.error(`[test:e2e] vite preview exited early (code ${proc.exitCode})`);
      process.exit(1);
    }
    try {
      await fetch(url);
      // One throwaway request so the first test doesn't pay the static
      // handler's first-request cost against its own timeout.
      await fetch(url);
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 500));
    }
  }
  console.error(
    `[test:e2e] vite preview did not come up within ${timeoutMs / 1000}s at ${url}`,
  );
  process.exit(1);
}
