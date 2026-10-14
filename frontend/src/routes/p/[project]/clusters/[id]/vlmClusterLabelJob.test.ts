/**
 * /clusters/[id]'s "Run VLM" uses `POST /vlm/label_cluster/{id}` plus status
 * polling instead of fetching crops and looping over a batch route, polls
 * only the job it just started, and confirms first. Static source scan (no
 * component-mount harness; see clusterMoveRace.test.ts's doc comment).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('runVlm uses POST /vlm/label_cluster/{id} + status polling, not the old batch loop', () => {
  it('imports pollAutoLabelJob alongside runVlmOnCluster', () => {
    expect(src).toMatch(/pollAutoLabelJob/);
    expect(src).toMatch(/runVlmOnCluster/);
  });

  it('runVlm calls runVlmOnCluster then polls via pollAutoLabelJob, never fetching a crop page itself', () => {
    const fn = src.match(/async function runVlm\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/await runVlmOnCluster\(clusterId,/);
    expect(fn).toMatch(/await pollAutoLabelJob\(/);
    // The old implementation fetched {API_PREFIX}/crops itself before
    // chunking into {API_PREFIX}/vlm/label_batch — neither should survive.
    expect(fn).not.toMatch(/getCrops\(/);
    expect(fn).not.toMatch(/label_batch/);
  });

  it('renders the job stage/progress from the polled state, not a client-computed percentage', () => {
    expect(src).toMatch(/vlmJob\.stage/);
    expect(src).toMatch(/vlmJob\.processed/);
    expect(src).toMatch(/vlmJob\.total/);
  });
});

describe('Run VLM polls only the job it just started, and confirms first', () => {
  it('runVlm asks for confirmation before starting the job', () => {
    const fn = src.match(/async function runVlm\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/window\.confirm\(/);
  });

  it('passes the just-started job’s own job_id into pollAutoLabelJob, not a bare call', () => {
    const fn = src.match(/async function runVlm\(\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    // Must read the id off the response of runVlmOnCluster (`vlmJob =
    // await runVlmOnCluster(...)`) and thread it into pollAutoLabelJob —
    // a call with no 4th arg would silently resume the M7 bug (reading
    // whatever job happens to be in the status slot).
    expect(fn).toMatch(/vlmJob = await runVlmOnCluster\(/);
    expect(fn).toMatch(/pollAutoLabelJob\(\s*\([^)]*\)\s*=>[\s\S]*?vlmJob\.job_id/);
  });
});
