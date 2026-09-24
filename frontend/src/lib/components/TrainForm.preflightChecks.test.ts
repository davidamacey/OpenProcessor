/**
 * dq-queues cutover (2026-09-24): `POST {API_PREFIX}/train/preflight` adds
 * two new checks, `export_not_empty` and `export_generation` (a stale
 * export blocks). `TrainForm`'s preflight panel (`{#each preflight.checks
 * as c (c.name)}`, `TrainForm.svelte`) renders every check generically by
 * `name`/`severity`/`message` — no per-check-id branching — so a new check
 * id needs no component change, only proof that the existing generic
 * render actually surfaces it. Mount-based (Svelte 5 mount/unmount/
 * flushSync under jsdom), same harness as TrainForm.gpuPicker.test.ts.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import TrainForm from './TrainForm.svelte';
import type { PreflightReport } from '$lib/types_train';

let target: HTMLDivElement;
let instance: unknown;

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

function baseProps(preflight: PreflightReport | null) {
  return {
    datasetExportDir: '/exports/x',
    profiles: [],
    presets: [],
    preflight,
    preflighting: false,
    starting: false,
    onPreflight: () => {},
    onStart: () => {},
    onStartCampaign: () => {},
  };
}

afterEach(() => {
  if (instance) {
    unmount(instance);
    instance = undefined;
  }
  target?.remove();
  vi.unstubAllGlobals();
});

describe('TrainForm — preflight panel renders new check ids generically', () => {
  it('renders export_not_empty and export_generation with their served severity/message, no special-casing needed', () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ options: [], default: null })),
    );
    target = document.createElement('div');
    document.body.appendChild(target);
    instance = mount(TrainForm, {
      target,
      props: baseProps({
        blocked: true,
        summary: '2 checks failed',
        checks: [
          {
            name: 'export_not_empty',
            severity: 'block',
            message: 'export has 0 eligible items',
          },
          {
            name: 'export_generation',
            severity: 'block',
            message: 'export is stale against the current index',
          },
        ],
      }),
    } as never);
    flushSync();

    expect(target.textContent).toContain('export_not_empty');
    expect(target.textContent).toContain('export has 0 eligible items');
    expect(target.textContent).toContain('export_generation');
    expect(target.textContent).toContain('export is stale against the current index');
  });
});
