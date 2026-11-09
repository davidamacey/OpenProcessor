/**
 * `types_jobs.ts` vs the generated contract: the `JobView` / `JobPage` keys and
 * the status ids, so a served rename fails here instead of leaving a job
 * "queued" forever in the UI.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_jobs';
import { JOB_STATUSES } from '$lib/types_jobs';

type Schema = { properties?: Record<string, { enum?: string[] }>; required?: string[] };
const doc = spec as unknown as {
  components: { schemas: Record<string, Schema> };
};
const schemas = doc.components.schemas;
const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();

describe('job runtime contract', () => {
  it('JobView has exactly the keys the frontend type declares', () => {
    const expected = keys<keyof T.JobView>({
      id: true,
      kind: true,
      status: true,
      project: true,
      scope: true,
      queue: true,
      params: true,
      progress: true,
      progress_updated_at: true,
      result: true,
      error_code: true,
      error_message: true,
      pause_reason: true,
      pause_requested: true,
      cancel_requested: true,
      attempt: true,
      max_attempts: true,
      retry_count: true,
      requested_by: true,
      created_at: true,
      queued_at: true,
      started_at: true,
      finished_at: true,
      updated_at: true,
    });
    expect(Object.keys(schemas.JobView!.properties ?? {}).sort()).toEqual(expected);
  });

  it('JobPage has exactly the keys the frontend type declares', () => {
    expect(Object.keys(schemas.JobPage!.properties ?? {}).sort()).toEqual(
      keys<keyof T.JobPage>({ items: true, next_cursor: true }),
    );
  });

  it('the status ids are the served enum, in the served order', () => {
    expect(schemas.JobView!.properties!.status!.enum).toEqual([...JOB_STATUSES]);
  });
});
