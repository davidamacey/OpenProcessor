/**
 * The readiness wire types (types_quality.ts) vs the readiness schemas of the vendored OpenAPI:
 * every key matches in both directions.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type {
  ClassCoverage,
  DatasetReadiness,
  ReadinessCheck,
  ReadinessClassRow,
  ReadinessLink,
} from '$lib/types_quality';

type Schema = { properties?: Record<string, unknown> };
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;

const CASES: [string, Record<string, true>][] = [
  [
    'DatasetReadiness',
    {
      verdict: true,
      checks: true,
      coverage: true,
      computed_at: true,
    } satisfies Record<keyof DatasetReadiness, true>,
  ],
  [
    'ReadinessCheck',
    {
      id: true,
      label: true,
      severity: true,
      message: true,
      count: true,
      link: true,
      classes: true,
    } satisfies Record<keyof ReadinessCheck, true>,
  ],
  [
    'ReadinessLink',
    { section: true, query: true } satisfies Record<keyof ReadinessLink, true>,
  ],
  [
    'ReadinessClassRow',
    {
      class_name: true,
      count: true,
      precision: true,
      ci_low: true,
      ci_high: true,
    } satisfies Record<keyof ReadinessClassRow, true>,
  ],
  [
    'ClassCoverage',
    {
      class_name: true,
      validated: true,
      machine_only: true,
      detector_only: true,
      test: true,
    } satisfies Record<keyof ClassCoverage, true>,
  ],
];

describe('readiness wire types match the contract', () => {
  for (const [name, keys] of CASES) {
    it(name, () => {
      expect(Object.keys(schemas[name]!.properties!).sort()).toEqual(
        Object.keys(keys).sort(),
      );
    });
  }
});
