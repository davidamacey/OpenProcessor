/**
 * AugmentationPanel's preset ids must be exactly the trainer's PRESETS keys
 * (OpenProcessor docker/trainer/augment.py). An unknown id isn't rejected at
 * /train/start; the trainer raises "unknown augmentation preset" only after
 * the run started (and stopped the VLM). The list isn't in the vendored
 * contract yet, so this reads the local backend checkout and skips when it
 * isn't present (CI), the same way `npm run contract:check` does.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '../../..');
const backendRepo = path.resolve(
  repoRoot,
  process.env.OPENPROCESSOR_REPO ?? '../openprocessor',
);
const backendRef = process.env.OPENPROCESSOR_REF ?? 'main';

function trainerPresetIds(): string[] | null {
  if (!existsSync(path.join(backendRepo, '.git'))) return null;
  let src: string;
  try {
    src = execFileSync(
      'git',
      ['-C', backendRepo, 'show', `${backendRef}:docker/trainer/augment.py`],
      { encoding: 'utf-8' },
    );
  } catch {
    return null;
  }
  const table = src.match(/^PRESETS[^=]*=\s*\{([\s\S]*?)^\}/m)?.[1];
  if (!table) throw new Error('PRESETS table not found in augment.py');
  return [...table.matchAll(/^\s*'([a-z0-9_]+)'\s*:/gm)].map((m) => m[1]);
}

function panelPresetIds(): string[] {
  const src = readFileSync(
    path.join(repoRoot, 'src/lib/components/AugmentationPanel.svelte'),
    'utf-8',
  );
  const list = src.match(/const PRESETS = \[([\s\S]*?)\] as const;/)?.[1];
  if (!list) throw new Error('PRESETS list not found in AugmentationPanel.svelte');
  return [...list.matchAll(/'([a-z0-9_]+)'/g)].map((m) => m[1]);
}

const trainer = trainerPresetIds();

describe.skipIf(trainer === null)('augmentation presets match the trainer', () => {
  it('AugmentationPanel offers exactly the trainer PRESETS ids', () => {
    expect([...panelPresetIds()].sort()).toEqual([...trainer!].sort());
  });
});
