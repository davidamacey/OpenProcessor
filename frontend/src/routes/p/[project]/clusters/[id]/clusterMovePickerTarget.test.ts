/**
 * The cluster move picker offers only served clusters of the serving partition
 * and sends the chosen cluster uid; the old typed-integer rule (negative ids,
 * NaN) is gone. Static source scan (no component-mount harness; see
 * clusterMoveRace.test.ts's doc comment).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(here, '../../../../../lib');
const src = readFileSync(
  path.join(LIB, 'clusters/clusterDetailActions.svelte.ts'),
  'utf-8',
);
const pickerSrc = readFileSync(
  path.join(LIB, 'components/clusters/MoveClusterPicker.svelte'),
  'utf-8',
);

describe('the move target is a served cluster uid', () => {
  it('confirmMovePicker sends the picked uid and applies no id arithmetic of its own', () => {
    const fn = src.match(/async function confirmMovePicker\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/moveCropIds\(\[\.\.\.sel\.ids\], movePickerValue\)/);
    expect(fn).not.toMatch(/Number\(|isFinite|id < 0|non-negative/);
  });

  it('the picker lists the served cluster cards, not a free-typed number', () => {
    expect(pickerSrc).toMatch(/<select/);
    expect(pickerSrc).toMatch(/actions\.moveCandidates/);
    expect(pickerSrc).not.toMatch(/type="number"/);
  });
});
