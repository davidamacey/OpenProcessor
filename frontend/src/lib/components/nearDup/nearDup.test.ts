import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { ApiError } from '$lib/api_errors';
import * as api from '$lib/api_nearDup';
import { createNearDupDecisionsController } from '$lib/nearDup/nearDupDecisionsController.svelte';
import { createNearDupPageController } from '$lib/nearDup/nearDupPageController.svelte';
import { createNearDupProfileController } from '$lib/nearDup/nearDupProfileController.svelte';
import NearDupDecisionsPanel from './NearDupDecisionsPanel.svelte';
import NearDupGroupList from './NearDupGroupList.svelte';
import NearDupGroupModal from './NearDupGroupModal.svelte';
import NearDupProfilePanel from './NearDupProfilePanel.svelte';
import NearDupRunsTable from './NearDupRunsTable.svelte';
import NearDupStatusCard from './NearDupStatusCard.svelte';
import type {
  NearDupGroupSummary,
  NearDupMemberView,
  NearDupRunView,
  NearDupStatus,
} from '$lib/types_nearDup';

vi.mock('$lib/api_nearDup');

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  vi.resetAllMocks();
  target = document.createElement('div');
  document.body.appendChild(target);
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
});

const q = (id: string) => target.querySelector(`[data-testid="${id}"]`);
const qa = (id: string) => [...target.querySelectorAll(`[data-testid="${id}"]`)];
async function settle(check: () => void): Promise<void> {
  await vi.waitFor(() => {
    flushSync();
    check();
  });
}

const run = (over: Partial<NearDupRunView> = {}): NearDupRunView => ({
  id: 'r1',
  level: 'image',
  mode: 'full',
  status: 'current',
  space: 's',
  algorithm_version: 'nd-1',
  profile_revision: 1,
  params: {},
  members_scored: 100,
  grouped_members: 40,
  dedup_groups: 12,
  leak_groups: 5,
  largest_group: 9,
  oversized_split: 0,
  holdout_absorbed: 2,
  started_at: '2026-01-01T00:00:00Z',
  finished_at: '2026-01-01T00:05:00Z',
  ...over,
});
const group = (
  id: string,
  over: Partial<NearDupGroupSummary> = {},
): NearDupGroupSummary => ({
  group_id: id,
  purpose: 'dedup',
  size: 4,
  representative_id: `img-${id}`,
  min_edge_cosine: 0.991,
  representative_thumbnail_url: `/curation/projects/p/images/img-${id}/thumbnail`,
  min_cosine_to_rep: 0.982,
  has_holdout: false,
  ...over,
});

describe('NearDupStatusCard', () => {
  const status: NearDupStatus = {
    profile: { name: 'default', revision: 2 },
    levels: [
      {
        level: 'image',
        current_run: run(),
        pending_count: 7,
        stale_reasons: ['pending'],
      },
      { level: 'item', current_run: null, pending_count: 0, stale_reasons: ['no_run'] },
    ],
  };

  it('shows the served stale reasons, an empty level, and starts the mode the reasons call for', async () => {
    vi.mocked(api.getNearDupStatus).mockResolvedValue(status);
    vi.mocked(api.listNearDupRuns).mockResolvedValue({ items: [] });
    vi.mocked(api.startNearDupRun).mockRejectedValue(
      new ApiError(409, '/x', {
        detail: { error: 'no_image_run', message: 'Build the image level first.' },
      }),
    );
    const ctl = createNearDupPageController();
    await ctl.loadOverview();
    instance = mount(NearDupStatusCard, { target, props: { ctl } });
    flushSync();
    expect(q('near-dup-profile-chip')?.textContent).toContain('revision 2');
    expect(q('near-dup-stale-image')?.textContent).toContain('not yet grouped (7)');
    expect(q('near-dup-no-run-item')).not.toBeNull();
    expect(q('near-dup-stale-item')?.textContent).toContain('never been built');
    expect(q('near-dup-run-full-item')?.textContent?.trim()).toBe('Build');
    expect(q('near-dup-run-full-image')?.textContent?.trim()).toBe('Full rebuild');

    (q('near-dup-run-incremental-image') as HTMLButtonElement).click();
    await settle(() =>
      expect(q('near-dup-run-error')?.textContent).toContain('Build the image'),
    );
    expect(api.startNearDupRun).toHaveBeenCalledWith(
      { level: 'image', mode: 'incremental' },
      expect.anything(),
    );
    ctl.dispose();
  });
});

describe('NearDupRunsTable', () => {
  it('lists the runs, or says there are none', () => {
    instance = mount(NearDupRunsTable, {
      target,
      props: { runs: [run(), run({ id: 'r2' })] },
    });
    flushSync();
    expect(qa('near-dup-run-row')).toHaveLength(2);
    unmount(instance);
    instance = mount(NearDupRunsTable, { target, props: { runs: [] } });
    flushSync();
    expect(q('near-dup-runs-empty')).not.toBeNull();
  });
});

describe('NearDupDecisionsPanel', () => {
  it('lists the pins with both pictures and says so when there are none', async () => {
    vi.mocked(api.listNearDupDecisions).mockResolvedValue({
      items: [
        {
          id: 'd1',
          level: 'image',
          a_id: 'a',
          b_id: 'b',
          decision: 'different',
          decided_by: 'me',
          decided_at: '2026-01-01T00:00:00Z',
          a_thumbnail_url: '/p/images/a/thumbnail',
          b_thumbnail_url: '/p/images/b/thumbnail',
        },
      ],
      next_cursor: null,
    });
    const ctl = createNearDupDecisionsController();
    instance = mount(NearDupDecisionsPanel, { target, props: { ctl } });
    await ctl.load();
    await settle(() => expect(qa('near-dup-decision')).toHaveLength(1));
    expect(qa('near-dup-thumb')).toHaveLength(2);
    expect(target.textContent).toContain('Not a duplicate');
    vi.mocked(api.listNearDupDecisions).mockResolvedValue({
      items: [],
      next_cursor: null,
    });
    vi.mocked(api.undecideNearDup).mockResolvedValue();
    (q('near-dup-decision-undo') as HTMLButtonElement).click();
    await settle(() => expect(q('near-dup-decisions-empty')).not.toBeNull());
    expect(api.undecideNearDup).toHaveBeenCalledWith('d1', expect.anything());
    ctl.dispose();
  });
});

describe('NearDupGroupList', () => {
  async function render() {
    const ctl = createNearDupPageController();
    const onopen = vi.fn();
    instance = mount(NearDupGroupList, { target, props: { ctl, onopen } });
    await ctl.showGroups('image', 'dedup');
    flushSync();
    return { ctl, onopen };
  }

  it('renders a group with its figures and the holdout chip, and opens it on click', async () => {
    vi.mocked(api.listNearDupGroups).mockResolvedValue({
      items: [group('a', { has_holdout: true })],
      next_cursor: null,
      empty_reason: null,
    });
    const { ctl, onopen } = await render();
    await settle(() => expect(qa('near-dup-group')).toHaveLength(1));
    const card = q('near-dup-group') as HTMLButtonElement;
    expect(card.textContent).toContain('4 members');
    expect(card.textContent).toContain('0.991');
    expect(card.textContent).toContain('In test holdout');
    expect((q('near-dup-thumb') as HTMLImageElement).getAttribute('src')).toBe(
      '/curation/projects/p/images/img-a/thumbnail',
    );
    card.click();
    expect(onopen).toHaveBeenCalledWith(expect.objectContaining({ group_id: 'a' }));
    ctl.dispose();
  });

  it('links each image group to the Workspace filtered to it', async () => {
    vi.mocked(api.listNearDupGroups).mockResolvedValue({
      items: [group('a')],
      next_cursor: null,
      empty_reason: null,
    });
    const { ctl } = await render();
    await settle(() => expect(qa('near-dup-group')).toHaveLength(1));
    const href = q('near-dup-group-workspace')!.getAttribute('href')!;
    expect(href).toContain('/workspace?');
    expect(new URLSearchParams(href.split('?')[1]).get('scope')).toBe('near_dup:a');
    ctl.dispose();
  });

  it('shows the served reason when the level has no groups', async () => {
    vi.mocked(api.listNearDupGroups).mockResolvedValue({
      items: [],
      next_cursor: null,
      empty_reason: '0 dedup groups at cosine 0.98 among 91 images',
    });
    const { ctl } = await render();
    await settle(() => expect(q('near-dup-groups-empty')).not.toBeNull());
    expect(q('near-dup-groups-empty')?.textContent).toContain(
      '0 dedup groups at cosine 0.98 among 91 images',
    );
    ctl.dispose();
  });

  it('shows the served error with a retry', async () => {
    vi.mocked(api.listNearDupGroups).mockRejectedValue(
      new ApiError(500, '/x', { detail: 'store down' }),
    );
    const { ctl } = await render();
    await settle(() =>
      expect(q('near-dup-groups-error')?.textContent).toContain('store down'),
    );
    ctl.dispose();
  });

  it('switches level and purpose through the served query', async () => {
    vi.mocked(api.listNearDupGroups).mockResolvedValue({
      items: [],
      next_cursor: null,
      empty_reason: 'none',
    });
    const { ctl } = await render();
    (q('near-dup-level-tab-item') as HTMLButtonElement).click();
    await settle(() =>
      expect(vi.mocked(api.listNearDupGroups).mock.calls.at(-1)![0]).toMatchObject({
        level: 'item',
        purpose: 'dedup',
      }),
    );
    (q('near-dup-purpose-tab-leak') as HTMLButtonElement).click();
    await settle(() =>
      expect(vi.mocked(api.listNearDupGroups).mock.calls.at(-1)![0]).toMatchObject({
        level: 'item',
        purpose: 'leak',
      }),
    );
    ctl.dispose();
  });
});

describe('NearDupGroupModal', () => {
  const member = (
    id: string,
    rep: boolean,
    decision: NearDupMemberView['decision'] = null,
  ) => ({
    member_id: id,
    is_representative: rep,
    cosine_to_rep: rep ? 1 : 0.99,
    thumbnail_url: `/curation/projects/p/images/${id}/thumbnail`,
    image_id: id,
    crop_id: null,
    class_name: id === 'b' ? 'van' : null,
    in_holdout: id === 'b',
    decision,
  });
  const detail = (pin: NearDupMemberView['decision'] = null) => ({
    group: group('g1'),
    members: [member('a', true), member('b', false, pin)],
  });

  it('pins a member apart from the representative, offers undo, and takes it back', async () => {
    vi.mocked(api.getNearDupGroup).mockResolvedValue(detail());
    vi.mocked(api.decideNearDup).mockResolvedValue({
      id: 'd1',
      level: 'image',
      a_id: 'a',
      b_id: 'b',
      decision: 'different',
      decided_by: 'me',
      decided_at: '',
      a_thumbnail_url: '',
      b_thumbnail_url: '',
    });
    vi.mocked(api.undecideNearDup).mockResolvedValue();
    const ondecided = vi.fn();
    instance = mount(NearDupGroupModal, {
      target,
      props: { level: 'image', group: group('g1'), onclose: () => {}, ondecided },
    });
    await settle(() => expect(qa('near-dup-member')).toHaveLength(2));
    expect(qa('near-dup-different')).toHaveLength(1);
    expect(target.textContent).toContain('van');
    expect(target.textContent).toContain('In test holdout');

    vi.mocked(api.getNearDupGroup).mockResolvedValue(
      detail({ id: 'd1', kind: 'different' }),
    );
    (q('near-dup-different') as HTMLButtonElement).click();
    await settle(() => expect(q('near-dup-undo')).not.toBeNull());
    expect(api.decideNearDup).toHaveBeenCalledWith({
      level: 'image',
      a_id: 'a',
      b_id: 'b',
      decision: 'different',
    });
    expect(q('near-dup-decided-note')?.textContent).toContain('1 decision saved');
    expect(ondecided).toHaveBeenCalledTimes(1);

    vi.mocked(api.getNearDupGroup).mockResolvedValue(detail());
    (q('near-dup-undo') as HTMLButtonElement).click();
    await settle(() => expect(q('near-dup-different')).not.toBeNull());
    expect(api.undecideNearDup).toHaveBeenCalledWith('d1');
    expect(q('near-dup-decided-note')).toBeNull();
    expect(ondecided).toHaveBeenCalledTimes(2);
  });

  it('offers undo for a pin served from an earlier visit', async () => {
    vi.mocked(api.getNearDupGroup).mockResolvedValue(detail({ id: 'old', kind: 'same' }));
    instance = mount(NearDupGroupModal, {
      target,
      props: {
        level: 'image',
        group: group('g1'),
        onclose: () => {},
        ondecided: () => {},
      },
    });
    await settle(() => expect(q('near-dup-undo')).not.toBeNull());
    expect(q('near-dup-undo')?.textContent).toContain('keep together');
  });

  it('shows the served error when the group is gone', async () => {
    vi.mocked(api.getNearDupGroup).mockRejectedValue(
      new ApiError(404, '/x', {
        detail: { error: 'group_not_found', message: 'no group' },
      }),
    );
    instance = mount(NearDupGroupModal, {
      target,
      props: {
        level: 'item',
        group: group('g1'),
        onclose: () => {},
        ondecided: () => {},
      },
    });
    await settle(() =>
      expect(q('near-dup-group-error')?.textContent).toContain('no group'),
    );
  });
});

describe('NearDupProfilePanel', () => {
  const body = (v: number) => ({
    version: 1,
    image: { dedup_cosine: v, leak_cosine: v, k: 64, max_group_size: 1000 },
    item: {
      dedup_cosine: v,
      leak_cosine: v,
      max_group_size: 1000,
      scope: 'image_leak_groups',
      same_detector_class: true,
    },
  });
  const doc = (name: string, revision: number, v: number) => ({
    kind: 'near_dup_profile',
    name,
    revision,
    description: '',
    body: body(v),
    created_at: '',
    updated_at: '',
    updated_by: '',
  });

  it('shows the seeded defaults beside the fields and saves an edit through the controller', async () => {
    vi.mocked(api.getActiveNearDupProfile).mockResolvedValue({
      axis: 'near_dup',
      state: 'active',
      name: 'default',
      revision: 2,
    });
    vi.mocked(api.getNearDupProfile).mockImplementation(async (name, revision) =>
      revision === 1 ? doc('default', 1, 0.98) : doc(name, revision ?? 2, 0.95),
    );
    vi.mocked(api.saveNearDupProfile).mockResolvedValue(doc('default', 3, 0.9));
    vi.mocked(api.activateNearDupProfile).mockResolvedValue({
      axis: 'near_dup',
      state: 'active',
      name: 'default',
      revision: 3,
    });
    const onSaved = vi.fn();
    const ctl = createNearDupProfileController(onSaved);
    instance = mount(NearDupProfilePanel, { target, props: { ctl } });
    await settle(() => expect(q('np-image-dedup')).not.toBeNull());
    expect(target.textContent).toContain('default 0.98');
    const input = q('np-image-dedup') as HTMLInputElement;
    expect(input.value).toBe('0.95');
    expect((q('near-dup-profile-save') as HTMLButtonElement).disabled).toBe(true);

    input.value = '0.9';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    await settle(() =>
      expect((q('near-dup-profile-save') as HTMLButtonElement).disabled).toBe(false),
    );
    (q('near-dup-profile-save') as HTMLButtonElement).click();
    await settle(() => expect(q('near-dup-profile-notice')).not.toBeNull());
    expect(api.saveNearDupProfile).toHaveBeenCalledWith(
      'default',
      expect.objectContaining({ image: expect.objectContaining({ dedup_cosine: 0.9 }) }),
      2,
      expect.anything(),
    );
    expect(q('near-dup-profile-notice')?.textContent).toContain('need a rebuild');
    expect(onSaved).toHaveBeenCalled();
  });

  it('shows the error and no form when no profile is active', async () => {
    vi.mocked(api.getActiveNearDupProfile).mockResolvedValue({
      axis: 'near_dup',
      state: 'none',
    });
    const ctl = createNearDupProfileController(() => {});
    instance = mount(NearDupProfilePanel, { target, props: { ctl } });
    await settle(() => expect(q('near-dup-profile-error')).not.toBeNull());
    expect(q('np-image-dedup')).toBeNull();
  });
});
