/**
 * Test-only loader for the operator-facing example profiles under
 * `examples/annotation-profiles/` (outside `src/`, never bundled). Parses
 * them through the same tier-2 parser a deployment's
 * `annotation-profiles.json` goes through.
 */
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  parseProfileDocument,
  type ProfileDocumentResult,
} from '$lib/annotations/config/parseSlotConfig';
import { callsRegionRoutes } from '$lib/annotations/registeredSlots';
import type { SlotSpec } from '$lib/annotations/types';
import { widgetTagServedSlot } from './regionSlot';

const here = path.dirname(fileURLToPath(import.meta.url));

export const EXAMPLES_DIR = path.resolve(
  here,
  '../../../../examples/annotation-profiles',
);

export function exampleProfileFiles(): string[] {
  return readdirSync(EXAMPLES_DIR)
    .filter((f) => f.endsWith('.json'))
    .sort();
}

export function readExampleDocument(file: string): unknown {
  return JSON.parse(readFileSync(path.join(EXAMPLES_DIR, file), 'utf-8'));
}

export function loadExampleProfile(file: string): ProfileDocumentResult {
  return parseProfileDocument(readExampleDocument(file));
}

/** Every region slot the app can register: the one synthesized from a
 *  served profile, plus every example profile slot that uses the region
 *  routes (a tier-2 customization of the served slot). */
export function regionContractSlots(): SlotSpec[] {
  const examples = exampleProfileFiles().flatMap((f) => loadExampleProfile(f).slots);
  return [widgetTagServedSlot, ...examples.filter(callsRegionRoutes)];
}
