import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import {
  activateNearDupProfile,
  getActiveNearDupProfile,
  getNearDupProfile,
  saveNearDupProfile,
} from '$lib/api_nearDup';
import { createNearDupProfileController } from './nearDupProfileController.svelte';
import type { NearDupProfileBody, NearDupProfileDoc } from '$lib/types_nearDup';

vi.mock('$lib/api_nearDup', () => ({
  activateNearDupProfile: vi.fn(),
  getActiveNearDupProfile: vi.fn(),
  getNearDupProfile: vi.fn(),
  saveNearDupProfile: vi.fn(),
}));

const body = (dedup: number): NearDupProfileBody => ({
  version: 1,
  image: { dedup_cosine: dedup, leak_cosine: dedup, k: 64, max_group_size: 1000 },
  item: {
    dedup_cosine: dedup,
    leak_cosine: dedup,
    max_group_size: 1000,
    scope: 'image_leak_groups',
    same_detector_class: true,
  },
});
const doc = (name: string, revision: number, dedup: number): NearDupProfileDoc => ({
  kind: 'near_dup_profile',
  name,
  revision,
  description: '',
  body: body(dedup),
  created_at: '',
  updated_at: '',
  updated_by: '',
});

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getActiveNearDupProfile).mockResolvedValue({
    axis: 'near_dup',
    state: 'active',
    name: 'tuned',
    revision: 3,
  });
  vi.mocked(getNearDupProfile).mockImplementation(async (name, revision) =>
    name === 'default' ? doc('default', 1, 0.98) : doc(name, revision ?? 3, 0.95),
  );
});

describe('createNearDupProfileController', () => {
  it('loads the active revision as the draft and revision 1 of default as the seeded values', async () => {
    const c = createNearDupProfileController(() => {});
    await c.load();
    expect(getNearDupProfile).toHaveBeenCalledWith('tuned', 3, expect.anything());
    expect(c.draft?.image.dedup_cosine).toBe(0.95);
    expect(c.seeded?.image.dedup_cosine).toBe(0.98);
    expect(c.dirty).toBe(false);
    c.dispose();
  });

  it('refuses to edit when no profile is active', async () => {
    vi.mocked(getActiveNearDupProfile).mockResolvedValue({
      axis: 'near_dup',
      state: 'none',
    });
    const c = createNearDupProfileController(() => {});
    await c.load();
    expect(c.draft).toBeNull();
    expect(c.loadError).toContain('No near-duplicate profile is active');
    c.dispose();
  });

  it('shows no defaults when the seeded revision is missing, and still edits', async () => {
    vi.mocked(getNearDupProfile).mockImplementation(async (name, revision) => {
      if (name === 'default') throw new ApiError(404, '/x', { detail: 'nope' });
      return doc(name, revision ?? 3, 0.95);
    });
    const c = createNearDupProfileController(() => {});
    await c.load();
    expect(c.seeded).toBeNull();
    expect(c.draft).not.toBeNull();
    c.dispose();
  });

  it('saves the next revision against the loaded one, then activates it against the old active', async () => {
    vi.mocked(saveNearDupProfile).mockResolvedValue(doc('tuned', 4, 0.9));
    vi.mocked(activateNearDupProfile).mockResolvedValue({
      axis: 'near_dup',
      state: 'active',
      name: 'tuned',
      revision: 4,
    });
    const onSaved = vi.fn();
    const c = createNearDupProfileController(onSaved);
    await c.load();
    c.draft!.image.dedup_cosine = 0.9;
    expect(c.dirty).toBe(true);
    await c.save();
    expect(saveNearDupProfile).toHaveBeenCalledWith(
      'tuned',
      expect.objectContaining({ image: expect.objectContaining({ dedup_cosine: 0.9 }) }),
      3,
      expect.anything(),
    );
    expect(activateNearDupProfile).toHaveBeenCalledWith(
      'tuned',
      4,
      { name: 'tuned', revision: 3 },
      expect.anything(),
    );
    expect(c.revision).toBe(4);
    expect(c.notice).toContain('revision 4');
    expect(c.dirty).toBe(false);
    expect(onSaved).toHaveBeenCalledTimes(1);
    c.dispose();
  });

  it('shows the served refusal reasons and activates nothing when the save is refused', async () => {
    vi.mocked(saveNearDupProfile).mockRejectedValue(
      new ApiError(422, '/x', {
        detail: {
          error: 'validation_failed',
          message: 'invalid',
          reasons: ['image.leak_cosine: must not exceed dedup_cosine'],
        },
      }),
    );
    const c = createNearDupProfileController(() => {});
    await c.load();
    c.draft!.image.leak_cosine = 0.99;
    await c.save();
    expect(c.saveError).toBe('invalid');
    expect(c.reasons).toEqual(['image.leak_cosine: must not exceed dedup_cosine']);
    expect(activateNearDupProfile).not.toHaveBeenCalled();
    expect(c.dirty).toBe(true);
    c.dispose();
  });

  it('puts the seeded values in the draft without saving, and reverts', async () => {
    const c = createNearDupProfileController(() => {});
    await c.load();
    c.useSeeded();
    expect(c.draft?.image.dedup_cosine).toBe(0.98);
    expect(c.dirty).toBe(true);
    expect(saveNearDupProfile).not.toHaveBeenCalled();
    c.reset();
    expect(c.draft?.image.dedup_cosine).toBe(0.95);
    c.dispose();
  });
});
