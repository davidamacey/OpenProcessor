import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import AssistSuggestions from '$components/assistRun/AssistSuggestions.svelte';
import type { ItemAssist } from '$lib/types_assist';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const ASSIST: ItemAssist = {
  run_id: 'asr_1',
  run_created_at: '2026-10-15T00:00:00Z',
  vlm_endpoint: 'env',
  prompt_pack: 'generic',
  class_name: 'widget_b',
  class_confidence: 'medium',
  attributes: [
    { key: 'finish', value: 'matte', confidence: 'high' },
    { key: 'edge', value: 'rounded', confidence: 'low' },
  ],
  boxes: [
    {
      box_id: 'b2',
      attributes: [{ key: 'tint', value: 'blue', confidence: 'medium' }],
    },
  ],
};

function render(props: {
  assist: ItemAssist | null | undefined;
  className?: string | null;
  boxId?: string | null;
}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(AssistSuggestions, { target, props });
  flushSync();
}

const chips = () => [...target.querySelectorAll('[data-testid="assist-chip"]')];

describe('AssistSuggestions', () => {
  it('renders nothing for an item no assist run answered', () => {
    render({ assist: null });
    expect(target.querySelector('[data-testid="assist-suggestions"]')).toBeNull();
  });

  it('shows each attribute as a chip with its confidence, marked as a suggestion', () => {
    render({ assist: ASSIST, className: 'widget_b' });
    expect(chips().map((c) => c.textContent?.replace(/\s+/g, ' ').trim())).toEqual([
      'finish: matte high',
      'edge: rounded low',
    ]);
    const root = target.querySelector('[data-testid="assist-suggestions"]')!;
    expect(root.textContent).toContain('VLM suggests');
    expect(root.getAttribute('title')).toContain('not labels');
  });

  it('says what the VLM said when it differs from the item class, which a lock keeps', () => {
    render({ assist: ASSIST, className: 'widget_a' });
    expect(target.querySelector('[data-testid="assist-said"]')?.textContent).toContain(
      'widget_b',
    );
  });

  it('does not repeat the class when the VLM agrees with it', () => {
    render({ assist: ASSIST, className: 'widget_b' });
    expect(target.querySelector('[data-testid="assist-said"]')).toBeNull();
  });

  it('shows one box its own attributes, and no class line', () => {
    render({ assist: ASSIST, className: 'widget_a', boxId: 'b2' });
    expect(chips().map((c) => c.textContent?.replace(/\s+/g, ' ').trim())).toEqual([
      'tint: blue medium',
    ]);
    expect(target.querySelector('[data-testid="assist-said"]')).toBeNull();
  });

  it('renders nothing for a box the run gave no attributes', () => {
    render({ assist: ASSIST, boxId: 'zz' });
    expect(target.querySelector('[data-testid="assist-suggestions"]')).toBeNull();
  });
});
