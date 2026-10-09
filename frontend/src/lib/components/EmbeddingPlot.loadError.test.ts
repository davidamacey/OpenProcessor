import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { ApiError } from '$lib/api';

const getVizProjection = vi.fn();

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, getVizProjection: (...a: unknown[]) => getVizProjection(...a) };
});

const { default: EmbeddingPlot } = await import('./EmbeddingPlot.svelte');

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
    },
  );
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
  vi.unstubAllGlobals();
  getVizProjection.mockReset();
});

async function settle(): Promise<void> {
  await vi.waitFor(() => {
    flushSync();
    expect(target.textContent).not.toContain('Loading embedding projection');
  });
}

describe('EmbeddingPlot read failures', () => {
  it('shows the served message of a failed read, not "not built yet"', async () => {
    getVizProjection.mockRejectedValue(
      new ApiError(503, '/viz/projection', {
        detail: { error: 'projection_unavailable', message: 'Index is down.' },
      }),
    );
    instance = mount(EmbeddingPlot, { target });
    await settle();
    expect(target.textContent).toContain('Index is down.');
    expect(target.textContent).not.toContain('No projection has been built yet');
  });

  it('shows "not built yet" when the read resolves not built', async () => {
    getVizProjection.mockResolvedValue({ built: false, points: [] });
    instance = mount(EmbeddingPlot, { target });
    await settle();
    expect(target.textContent).toContain('No projection has been built yet');
  });
});
