/**
 * `ConfigCloneRequest` (types_config.ts) vs the pack and profile clone
 * schemas of the vendored OpenAPI, including `from_project`.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type { ConfigCloneRequest } from '$lib/types_config';

type Schema = { properties?: Record<string, unknown> };
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;

const ours = Object.keys({
  new_name: true,
  revision: true,
  source: true,
  description: true,
  from_project: true,
} satisfies Record<keyof ConfigCloneRequest, true>).sort();

describe('ConfigCloneRequest matches the vendored clone schemas', () => {
  for (const name of ['PromptPackCloneRequest', 'RegionProfileCloneRequest']) {
    it(name, () => {
      expect(Object.keys(schemas[name]!.properties!).sort()).toEqual(ours);
    });
  }
});
