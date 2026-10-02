// @vitest-environment node
import { spawnSync } from 'node:child_process';
import { afterEach, describe, expect, it } from 'vitest';
import {
  spawnGroup,
  stopGroup,
  stopGroupSync,
} from '../../../scripts/lib/previewServer.mjs';

function marked(marker: string): number[] {
  const res = spawnSync('pgrep', ['-f', marker], { encoding: 'utf8' });
  return res.stdout.split('\n').filter(Boolean).map(Number);
}

async function until(pred: () => boolean, ms = 5000): Promise<boolean> {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (pred()) return true;
    await new Promise((r) => setTimeout(r, 25));
  }
  return pred();
}

const marker = () => `300.${process.pid}${Math.floor(Math.random() * 1e9)}`;
// A wrapper that starts a grandchild, like `npx vite preview` does.
const wrapper = (m: string) => ['-c', `sleep ${m} & wait`];

describe('previewServer group stop', () => {
  const live: number[] = [];
  afterEach(() => {
    for (const pid of live.splice(0)) {
      try {
        process.kill(-pid, 'SIGKILL');
      } catch {
        // already gone
      }
    }
  });

  it('stopGroup removes the wrapper AND its grandchild', async () => {
    const m = marker();
    const child = spawnGroup('sh', wrapper(m), { stdio: 'ignore' });
    live.push(child.pid!);
    expect(await until(() => marked(`^sleep ${m}$`).length === 1)).toBe(true);

    await stopGroup(child);

    expect(await until(() => marked(`^sleep ${m}$`).length === 0)).toBe(true);
  });

  it('stopGroupSync removes the grandchild too', async () => {
    const m = marker();
    const child = spawnGroup('sh', wrapper(m), { stdio: 'ignore' });
    live.push(child.pid!);
    expect(await until(() => marked(`^sleep ${m}$`).length === 1)).toBe(true);

    stopGroupSync(child);

    expect(await until(() => marked(`^sleep ${m}$`).length === 0)).toBe(true);
  });
});
