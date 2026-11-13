import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';
import { createReviewTabsController } from './reviewTabsController.svelte';

interface FakeUrl {
  queue: string | null;
  focus: { kind: 'item'; id: string } | null;
}

function fakeWs(url: FakeUrl) {
  const state = $state({ url });
  const setQueue = vi.fn((id: string | null) => {
    state.url.queue = id;
  });
  return {
    ws: {
      get url() {
        return state.url;
      },
      setQueue,
    } as unknown as WorkspaceController,
    setQueue,
  };
}

function serve(defaultTab: string | null): void {
  reviewTabsVocabularyStore.loaded = true;
  reviewTabsVocabularyStore.defaultTab = defaultTab;
}

afterEach(() => {
  reviewTabsVocabularyStore.resetForProjectChange();
});

describe('Review opens the served default queue', () => {
  it('replaces the empty queue with the served default once the vocabulary is loaded', () => {
    const { ws, setQueue } = fakeWs({ queue: null, focus: null });
    const cleanup = $effect.root(() => {
      const tabs = createReviewTabsController(ws);
      expect(tabs.defaultPending).toBe(true);
      serve('check_machine');
      flushSync();
      expect(setQueue).toHaveBeenCalledWith('check_machine', { replace: true });
      expect(tabs.defaultPending).toBe(false);
    });
    cleanup();
  });

  it('reads nothing when every queue is empty', () => {
    const { ws, setQueue } = fakeWs({ queue: null, focus: null });
    const cleanup = $effect.root(() => {
      const tabs = createReviewTabsController(ws);
      serve(null);
      flushSync();
      expect(setQueue).not.toHaveBeenCalled();
      expect(tabs.nothingToReview).toBe(true);
      expect(tabs.defaultPending).toBe(false);
    });
    cleanup();
  });

  it('keeps walking the grid list after the focused item was located', () => {
    serve('to_label');
    const { ws, setQueue } = fakeWs({
      queue: null,
      focus: { kind: 'item', id: 'crop-2' },
    });
    const cleanup = $effect.root(() => {
      const tabs = createReviewTabsController(ws);
      flushSync();
      // The loader clears the pending item once it is located; the list must not jump queue.
      tabs.pendingCropId = null;
      flushSync();
      expect(tabs.defaultPending).toBe(false);
      expect(tabs.nothingToReview).toBe(false);
      expect(setQueue).not.toHaveBeenCalled();
    });
    cleanup();
  });
});
