/**
 * `<ClusterBadge>` names a cluster with the server's `display_name`: a class name as it is, the
 * detector's guess marked as such, and an amber "Unlabeled" when no name was served. A click opens
 * the cluster in the Workspace via `goto` and never also selects the crop under the badge.
 */
import { describe, expect, it, vi } from 'vitest';
import { mountUi } from '$lib/test/mountUi';

const nav = vi.hoisted(() => ({ goto: vi.fn() }));
vi.mock('$app/navigation', () => ({ goto: nav.goto }));
vi.mock('$app/paths', () => ({ resolve: (p: string) => p }));

import ClusterBadge from './ClusterBadge.svelte';

function badge(props: Record<string, unknown>): HTMLButtonElement {
  return mountUi(ClusterBadge, {
    clusterUid: 'u-1',
    displayNo: 7,
    ...props,
  }).querySelector('button')!;
}

describe('ClusterBadge', () => {
  it('shows the number and the served class name', () => {
    expect(
      badge({ displayName: 'dog', displayNameKind: 'class' }).textContent?.trim(),
    ).toBe('#7 · dog');
  });

  it("marks the detector's name as the detector's", () => {
    const text = badge({ displayName: 'bus', displayNameKind: 'detector' }).textContent;
    expect(text).toContain('bus');
    expect(text).toContain('(detector)');
  });

  it('is an amber "Unlabeled" when no name was served, with the uid in the title', () => {
    const b = badge({ displayName: null, displayNameKind: 'none' });
    expect(b.textContent?.trim()).toBe('Unlabeled #7');
    expect(b.className).toContain('warn');
    expect(b.title).toContain('u-1');
  });

  it('opens the cluster scope with goto and stops the click', () => {
    const b = badge({ displayName: 'dog', displayNameKind: 'class' });
    const ev = new MouseEvent('click', { bubbles: true, cancelable: true });
    const stop = vi.spyOn(ev, 'stopPropagation');
    b.dispatchEvent(ev);
    expect(nav.goto).toHaveBeenCalledOnce();
    expect(String(nav.goto.mock.calls[0]![0])).toContain('u-1');
    expect(stop).toHaveBeenCalled();
  });
});
