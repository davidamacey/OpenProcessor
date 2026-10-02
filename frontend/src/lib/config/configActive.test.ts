/**
 * The shared active-ref state: every write sends the last read's
 * `expected_active`, a successful write runs `onchanged` (and only then),
 * activation keeps the served response, deactivate exists only where the
 * resource has the route, and `active_conflict` re-reads.
 */
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api';
import {
  activateResponseFixture,
  profileActiveFixture,
} from '$lib/test/fixtures/regionProfiles';
import type { ProfileActivateResponse } from '$lib/types_profiles';
import { ConfigActive } from './configActive.svelte';

function refusal(status: number, detail: Record<string, unknown>): ApiError {
  return new ApiError(status, '/x', { detail });
}

function setup(withDeactivate = true) {
  const backend = {
    getActive: vi.fn().mockResolvedValue(profileActiveFixture()),
    activate: vi.fn().mockResolvedValue(activateResponseFixture()),
    rollback: vi.fn().mockResolvedValue(profileActiveFixture()),
    ...(withDeactivate
      ? {
          deactivate: vi
            .fn()
            .mockResolvedValue(
              profileActiveFixture({ active: { name: null, revision: null } }),
            ),
        }
      : {}),
  };
  const ctl = new ConfigActive<ProfileActivateResponse>(backend);
  const onchanged = vi.fn();
  ctl.onchanged = onchanged;
  return { ctl, backend, onchanged };
}

describe('ConfigActive', () => {
  it('deactivate sends expected_active from the last read and adopts the served ref', async () => {
    const { ctl, backend, onchanged } = setup();
    await ctl.load();
    expect(ctl.deactivatable).toBe(true);
    expect(await ctl.deactivate()).toBe(true);
    expect(backend.deactivate).toHaveBeenCalledWith({
      expected_active: { name: 'widget_tag', revision: 2 },
    });
    expect(ctl.active?.active.name).toBeNull();
    expect(onchanged).toHaveBeenCalledTimes(1);
  });

  it('a resource without the route cannot deactivate', async () => {
    const { ctl } = setup(false);
    await ctl.load();
    expect(ctl.deactivatable).toBe(false);
    expect(await ctl.deactivate()).toBe(false);
  });

  it('activation keeps the served response (impact) and runs onchanged', async () => {
    const { ctl, backend, onchanged } = setup();
    await ctl.load();
    expect(await ctl.activate('widget_tag', 3, false)).toBe(true);
    expect(backend.activate).toHaveBeenCalledWith('widget_tag', {
      revision: 3,
      expected_active: { name: 'widget_tag', revision: 2 },
      force: false,
    });
    expect(ctl.lastActivation?.impact?.items_total).toBe(1840);
    expect(ctl.active?.active).toEqual({ name: 'widget_tag', revision: 3 });
    expect(onchanged).toHaveBeenCalledTimes(1);
  });

  it('force is sent only when the caller passes it (after seeing force_allowed)', async () => {
    const { ctl, backend } = setup();
    await ctl.load();
    await ctl.activate('widget_tag', 3, true);
    expect(backend.activate).toHaveBeenLastCalledWith('widget_tag', {
      revision: 3,
      expected_active: { name: 'widget_tag', revision: 2 },
      force: true,
    });
  });

  it('a refusal runs no onchanged, keeps the served message and report, and clears the last activation', async () => {
    const { ctl, backend, onchanged } = setup();
    await ctl.load();
    await ctl.activate('widget_tag', 3, false);
    const report = { ok: false, errors: [], warnings: [], force_allowed: true };
    backend.deactivate!.mockRejectedValue(
      refusal(422, { error: 'validation_failed', message: 'Refused.', report }),
    );
    expect(await ctl.deactivate()).toBe(false);
    expect(ctl.actionError).toBe('Refused.');
    expect(ctl.activateReport).toEqual(report);
    expect(ctl.lastActivation).toBeNull();
    expect(onchanged).toHaveBeenCalledTimes(1);
  });

  it('active_conflict on deactivate re-reads the active ref', async () => {
    const { ctl, backend } = setup();
    await ctl.load();
    backend.deactivate!.mockRejectedValue(
      refusal(409, { error: 'active_conflict', message: 'Changed.' }),
    );
    backend.getActive.mockResolvedValue(
      profileActiveFixture({ active: { name: 'env_tags', revision: null } }),
    );
    await ctl.deactivate();
    expect(backend.getActive).toHaveBeenCalledTimes(2);
    expect(ctl.active?.active.name).toBe('env_tags');
  });
});
