import { afterEach, describe, expect, it } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import CombineOriginRows from './CombineOriginRows.svelte';
import { mapRawCrop } from '$lib/api';
import { makeItem } from '$lib/test/makeItem';

let target: HTMLDListElement;
let instance: Record<string, unknown> | undefined;

function render(over: Parameters<typeof makeItem>[0]) {
  target = document.createElement('dl');
  document.body.appendChild(target);
  instance = mount(CombineOriginRows, {
    target,
    props: { crop: mapRawCrop(makeItem(over)) },
  });
  flushSync();
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const t = (id: string) =>
  target.querySelector(`[data-testid="${id}"]`)?.textContent?.trim();

describe('CombineOriginRows', () => {
  it('renders nothing for an item no combine produced', () => {
    render({
      origin_project: null,
      origin_item_id: null,
      origin_image_id: null,
      origin_split: null,
      combine_conflict: false,
      combine_conflict_origins: [],
      combine_merged_origins: [],
    });
    expect(target.children).toHaveLength(0);
  });

  it('renders the served origin, conflict chip and merged origins verbatim', () => {
    render({
      origin_project: 'widgets_a',
      origin_item_id: 'item-origin-9',
      origin_image_id: 'image-origin-9',
      origin_split: 'train',
      combine_conflict: true,
      combine_conflict_origins: ['widgets_a', 'widgets_b'],
      combine_merged_origins: ['widgets_c'],
    });
    expect(t('combine-origin-project')).toContain('widgets_a');
    expect(t('combine-origin-conflict')).toBe('Conflict between sources');
    expect(t('combine-origin-item')).toBe('item-origin-9');
    expect(t('combine-origin-image')).toBe('image-origin-9');
    expect(t('combine-origin-split')).toBe('train');
    expect(t('combine-origin-conflicts')).toBe('widgets_a, widgets_b');
    expect(t('combine-origin-merged')).toBe('widgets_c');
  });

  it('no conflict chip or conflicting-origins row without combine_conflict; null facts are omitted', () => {
    render({
      origin_project: 'widgets_a',
      origin_item_id: null,
      origin_image_id: null,
      origin_split: null,
      combine_conflict: false,
      combine_conflict_origins: ['widgets_a'],
      combine_merged_origins: [],
    });
    expect(t('combine-origin-project')).toContain('widgets_a');
    expect(t('combine-origin-conflict')).toBeUndefined();
    expect(t('combine-origin-conflicts')).toBeUndefined();
    expect(t('combine-origin-item')).toBeUndefined();
    expect(t('combine-origin-split')).toBeUndefined();
    expect(t('combine-origin-merged')).toBeUndefined();
  });
});
