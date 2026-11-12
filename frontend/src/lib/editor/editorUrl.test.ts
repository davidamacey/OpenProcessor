import { describe, expect, it, vi } from 'vitest';

vi.mock('$lib/projectPaths', () => ({ projectHref: (p: string) => `/p/default${p}` }));

import { editorHref, editorListQuery, workspaceBackHref } from './editorUrl';

describe('editor links', () => {
  it('opens an image with the Workspace query attached, and goes back to it', () => {
    expect(editorHref('img 1', '?view=gallery&class_name=widget')).toBe(
      '/p/default/images/img%201?view=gallery&class_name=widget',
    );
    expect(workspaceBackHref('?view=gallery')).toBe('/p/default/workspace?view=gallery');
    expect(workspaceBackHref('')).toBe('/p/default/workspace');
  });

  it('walks the list the Workspace showed: filter, queue, scope and gallery sort', () => {
    const q = editorListQuery(
      new URLSearchParams(
        'view=gallery&class_name=widget&queue=to_label&scope=cluster%3Au1&sort_gallery=taken_at',
      ),
    );
    expect(q).toMatchObject({
      class_name: ['widget'],
      review_tab: 'to_label',
      cluster_uid: ['u1'],
      sort: 'taken_at',
    });
  });

  it('sends no sort when the Workspace chose none', () => {
    expect(editorListQuery(new URLSearchParams('view=gallery')).sort).toBeUndefined();
  });
});
