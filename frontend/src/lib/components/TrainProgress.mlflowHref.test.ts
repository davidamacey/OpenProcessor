import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { TrainJobStatus } from '$lib/types_train';
import { trainStatusFixtureW1 } from '$lib/test/fixtures/trainRun';
import TrainProgress from './TrainProgress.svelte';

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function hrefs(url: string): string[] {
  target = document.createElement('div');
  document.body.appendChild(target);
  const status = { ...trainStatusFixtureW1, mlflow_run_url: url } as TrainJobStatus;
  instance = mount(TrainProgress, { target, props: { status } });
  flushSync();
  return Array.from(target.querySelectorAll('a')).map(
    (a) => a.getAttribute('href') ?? '',
  );
}

describe('TrainProgress MLflow link', () => {
  it('renders no link for a served non-http(s) URL', () => {
    expect(hrefs('javascript:alert(1)')).toEqual([]);
  });
  it('control: an http URL still renders the link', () => {
    expect(hrefs('http://op-mlflow:5000/#/experiments/1/runs/r')).toEqual([
      'http://op-mlflow:5000/#/experiments/1/runs/r',
    ]);
  });
});
