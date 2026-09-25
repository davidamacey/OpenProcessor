/**
 * T1 (visual audit 2026-09-24): the MLflow link pointed at a hardcoded
 * `:5000` that has none of the runs. It now follows the served runs'
 * `mlflow_run_url` origin, and is absent when nothing is served.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import MonitoringLinks from './MonitoringLinks.svelte';

let instance: unknown;
let target: HTMLDivElement;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function render(props: Record<string, unknown>): HTMLDivElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(MonitoringLinks, { target, props } as never);
  flushSync();
  return target;
}

describe('MonitoringLinks MLflow link (T1)', () => {
  it("links to the served run URL's MLflow origin", () => {
    const el = render({
      mlflowRunUrls: [null, 'http://localhost:4731/#/experiments/1/runs/abc'],
    });
    const a = el.querySelector('[data-testid="mlflow-link"]');
    expect(a?.getAttribute('href')).toBe('http://localhost:4731');
    expect(el.innerHTML).not.toContain(':5000');
  });

  it('shows no MLflow link when no run URL is served', () => {
    const el = render({ mlflowRunUrls: [null] });
    expect(el.querySelector('[data-testid="mlflow-link"]')).toBeNull();
    expect(el.textContent).not.toContain('MLflow');
  });
});
