/**
 * Promote must stay on the 202 job flow: nothing in the UI source may send
 * or suggest `wait=true` (the blocking mode) to users.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const FILES = [
  'src/lib/api.ts',
  'src/lib/promote.ts',
  'src/lib/promoteJobController.svelte.ts',
  'src/lib/components/PromoteModal.svelte',
  'src/routes/p/[project]/train/+page.svelte',
  'src/routes/p/[project]/models/+page.svelte',
];

describe('promote never uses wait=true', () => {
  for (const f of FILES) {
    it(f, () => {
      const src = readFileSync(f, 'utf8');
      expect(src).not.toMatch(/wait\s*=\s*true/i);
      expect(src).not.toMatch(/\bwait\s*:\s*true/);
    });
  }
});
