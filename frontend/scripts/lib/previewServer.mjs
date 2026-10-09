/**
 * Process-group helpers for the e2e runner.
 *
 * A child started through a wrapper (`npx vite preview`) is a grandchild of
 * the runner, so signalling the wrapper alone leaves the real server running
 * forever. Everything here spawns into its OWN process group
 * (`detached: true`) and signals the whole group (`process.kill(-pid, ...)`),
 * and only ever signals groups this module created.
 */
import { spawn } from 'node:child_process';
import { join } from 'node:path';

const GRACE_MS = 3000;

function groupAlive(pid) {
  try {
    process.kill(-pid, 0);
    return true;
  } catch (e) {
    return e.code === 'EPERM';
  }
}

function signalGroup(pid, sig) {
  try {
    process.kill(-pid, sig);
  } catch {
    // group already gone
  }
}

/** Spawn `cmd` as the leader of a new process group. */
export function spawnGroup(cmd, args, opts = {}) {
  const child = spawn(cmd, args, { ...opts, detached: true });
  child.once('error', () => {});
  return child;
}

/**
 * SIGTERM the child's whole group, then SIGKILL it if anything is still
 * alive after `graceMs`. Resolves once the group is gone.
 */
export async function stopGroup(child, graceMs = GRACE_MS) {
  const pid = child.pid;
  if (!pid) return;
  signalGroup(pid, 'SIGTERM');
  const deadline = Date.now() + graceMs;
  // The leader stays a zombie (and so keeps the group "alive") until its
  // exit event has been delivered, so wait for that before polling.
  if (child.exitCode === null && child.signalCode === null) {
    await Promise.race([
      new Promise((r) => child.once('exit', r)),
      new Promise((r) => setTimeout(r, graceMs)),
    ]);
  }
  while (groupAlive(pid) && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 50));
  }
  if (groupAlive(pid)) {
    signalGroup(pid, 'SIGKILL');
  }
}

/**
 * Synchronous variant for `process.on('exit')` handlers, where nothing can
 * be awaited: SIGTERM, a short bounded pause, then SIGKILL.
 */
export function stopGroupSync(child, pauseMs = 200) {
  const pid = child.pid;
  if (!pid) return;
  signalGroup(pid, 'SIGTERM');
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, pauseMs);
  signalGroup(pid, 'SIGKILL');
}

/**
 * Start `vite preview` directly (this node binary running vite's own entry
 * point, no `npx` wrapper) in its own process group.
 */
export function startPreview(root, port) {
  const vite = join(root, 'node_modules', 'vite', 'bin', 'vite.js');
  return spawnGroup(
    process.execPath,
    [vite, 'preview', '--port', String(port), '--strictPort'],
    { cwd: root, stdio: 'ignore' },
  );
}
