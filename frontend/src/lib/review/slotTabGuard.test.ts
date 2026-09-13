import { describe, it, expect } from 'vitest';
import { isSlotSuppressedTab } from './slotTabGuard';

describe('isSlotSuppressedTab', () => {
  it('suppresses the plates tab', () => {
    expect(isSlotSuppressedTab('plates')).toBe(true);
  });

  it('does not suppress core review tabs', () => {
    for (const tab of ['all', 'uncertainty', 'model_disagreements', 'coco_blind_spots']) {
      expect(isSlotSuppressedTab(tab)).toBe(false);
    }
  });

  it('does not suppress an unknown tab id', () => {
    expect(isSlotSuppressedTab('something_else')).toBe(false);
  });
});
