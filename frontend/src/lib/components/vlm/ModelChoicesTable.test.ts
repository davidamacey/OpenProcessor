/**
 * The read-only "all model choices" table: each served row's label, scope
 * through the served scope label (raw when unlabeled), current value,
 * dimensions, choices and the served settable / reason help; a role with
 * an editor links to it, a role without one does not.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { ModelChoice } from '$lib/types_profiles';
import ModelChoicesTable from './ModelChoicesTable.svelte';

const ROWS: ModelChoice[] = [
  {
    role: 'vlm',
    label: 'VLM endpoint',
    scope: 'per_run',
    current: 'local_vlm',
    dims: null,
    choices: [
      { id: 'local_vlm', label: 'Local VLM' },
      { id: 'off', label: 'Off' },
    ],
    settable: true,
    settable_via: 'Settings → Models',
  },
  {
    role: 'embedding',
    label: 'Embedding model',
    scope: 'deployment',
    current: 'clip_b',
    dims: 512,
    choices: [],
    settable: false,
    reason: 'Changing it re-embeds every crop.',
  },
  {
    role: 'region_ocr',
    label: 'Region OCR',
    scope: 'mystery_scope' as never,
    current: null,
    choices: [],
    settable: true,
  },
];

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

function render(choices = ROWS) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(ModelChoicesTable, {
    target,
    props: {
      choices,
      scopeLabels: { per_run: 'Chosen per run', deployment: 'Deployment config' },
    },
  });
  flushSync();
}

const row = (role: string) =>
  target.querySelector<HTMLElement>(
    `[data-testid="model-choice-row"][data-role="${role}"]`,
  )!;

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

describe('ModelChoicesTable', () => {
  it('prints the served facts, scope through the served label (raw when unlabeled)', () => {
    render();
    expect(row('vlm').textContent).toContain('Chosen per run');
    expect(row('vlm').textContent).toContain('local_vlm');
    expect(row('vlm').textContent).toContain('Local VLM');
    expect(row('vlm').textContent).toContain('settable');
    expect(row('vlm').textContent).toContain('Settings → Models');
    expect(row('embedding').textContent).toContain('Deployment config');
    expect(row('embedding').textContent).toContain('512');
    expect(row('embedding').textContent).toContain('not settable');
    expect(row('embedding').textContent).toContain('Changing it re-embeds every crop.');
    expect(row('region_ocr').textContent).toContain('mystery_scope');
    expect(row('region_ocr').textContent).toContain('—');
  });

  it('links a role with an editor, never one without', () => {
    render();
    expect(row('vlm').querySelector('a')?.getAttribute('href')).toMatch(
      /\/settings\/models#endpoints$/,
    );
    expect(row('region_ocr').querySelector('a')?.getAttribute('href')).toMatch(
      /\/settings\/region-profiles$/,
    );
    expect(row('embedding').querySelector('a')).toBeNull();
  });

  it('says so when nothing is served', () => {
    render([]);
    expect(target.querySelector('[data-testid="model-choices-empty"]')).not.toBeNull();
  });
});
