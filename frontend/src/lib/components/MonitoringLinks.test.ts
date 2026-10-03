/**
 * The MLflow link is the served `mlflow_public_url` of `GET /health`
 * (`OP_MLFLOW_PUBLIC_URL`); nothing is derived from run URLs or a port.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import MonitoringLinks from './MonitoringLinks.svelte';
import { healthStore } from '$stores/health.svelte';
import { curationSettingsStore } from '$stores/curationSettings.svelte';
import { EMPTY_CURATION_SETTINGS } from '$lib/curationSettings';

const getCurationSettings = vi.fn();
vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    getCurationSettings: (...a: unknown[]) => getCurationSettings(...a),
  };
});

beforeEach(() => {
  healthStore.health = null;
  curationSettingsStore.reset();
  getCurationSettings.mockResolvedValue(EMPTY_CURATION_SETTINGS);
});

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

describe('MonitoringLinks MLflow link', () => {
  it('links to the served mlflow_public_url, verbatim', () => {
    healthStore.health = {
      status: 'ok',
      mlflow_public_url: 'https://mlflow.example/base',
    };
    const el = render({});
    const a = el.querySelector('[data-testid="mlflow-link"]');
    expect(a?.getAttribute('href')).toBe('https://mlflow.example/base');
  });

  it('ignores run URLs: the base comes only from the served health', () => {
    healthStore.health = { status: 'ok', mlflow_public_url: null };
    const el = render({
      mlflowRunUrls: ['http://localhost:4731/#/experiments/1/runs/abc'],
    });
    expect(el.querySelector('[data-testid="mlflow-link"]')).toBeNull();
  });

  it('shows no MLflow link when none is served or the URL is not http(s)', () => {
    healthStore.health = { status: 'ok', mlflow_public_url: 'javascript:alert(1)' };
    const el = render({});
    expect(el.querySelector('[data-testid="mlflow-link"]')).toBeNull();
    expect(el.textContent).not.toContain('MLflow');
  });
});

describe('MonitoringLinks served dashboards', () => {
  const served = (links: Record<string, string | null>) =>
    getCurationSettings.mockResolvedValue({
      ...EMPTY_CURATION_SETTINGS,
      monitoring_links: {
        grafana: null,
        prometheus: null,
        opensearch_dashboards: null,
        ...links,
      },
    });

  it('renders only the served http(s) URLs, verbatim, and no guessed host:port', async () => {
    served({
      grafana: 'https://grafana.example/d/x',
      prometheus: 'javascript:alert(1)',
      opensearch_dashboards: null,
    });
    const el = render({});
    await vi.waitFor(() => {
      flushSync();
      expect(el.querySelectorAll('[data-testid="monitoring-link"]')).toHaveLength(1);
    });
    const a = el.querySelector('[data-testid="monitoring-link"]');
    expect(a?.getAttribute('href')).toBe('https://grafana.example/d/x');
    expect(el.textContent).not.toContain('Prometheus');
    expect(el.textContent).not.toContain('OpenSearch');
    expect(el.innerHTML).not.toMatch(/:46\d\d/);
  });

  it('shows no row at all when nothing is served', async () => {
    const el = render({});
    await vi.waitFor(() => expect(getCurationSettings).toHaveBeenCalled());
    flushSync();
    expect(el.querySelector('[data-testid="monitoring-link"]')).toBeNull();
    expect(el.textContent).not.toContain('Dashboards');
  });
});
