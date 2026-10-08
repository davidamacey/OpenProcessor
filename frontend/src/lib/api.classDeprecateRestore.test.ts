/**
 * `/classes` Deprecate/Restore (OpenProcessor 01324cb, 243f7f2):
 * `POST {API_PREFIX}/classes/{id}/deprecate` and `.../restore` — the
 * `/classes` "Restore" button used to be permanently disabled (no
 * backend support). deprecateClass()/restoreClass() are thin POST
 * wrappers; classStillReferencedDetail() is the one place that parses
 * deprecate's structured 409 (`{error, message, class_id, item_count,
 * confirmed_label_count}`) — restore's 409 is a PLAIN STRING detail and
 * must NOT be mistaken for the structured shape.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ApiError,
  API_PREFIX,
  classMergedDetail,
  classMergedRestoreText,
  classStillReferencedDetail,
  deprecateClass,
  restoreClass,
} from './api';

const ok = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('deprecateClass', () => {
  it('POSTs to /classes/{id}/deprecate', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(ok({ class_id: 8, class_name: 'widget_a' }));
    vi.stubGlobal('fetch', fetchMock);

    await deprecateClass(8);

    const [calledUrl, calledInit] = fetchMock.mock.calls[0]!;
    expect(String(calledUrl)).toBe(`${API_PREFIX}/classes/8/deprecate`);
    expect(calledInit.method).toBe('POST');
  });
});

describe('restoreClass', () => {
  it('POSTs to /classes/{id}/restore', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(ok({ class_id: 8, class_name: 'widget_a' }));
    vi.stubGlobal('fetch', fetchMock);

    await restoreClass(8);

    const [calledUrl, calledInit] = fetchMock.mock.calls[0]!;
    expect(String(calledUrl)).toBe(`${API_PREFIX}/classes/8/restore`);
    expect(calledInit.method).toBe('POST');
  });
});

describe('classStillReferencedDetail', () => {
  const body409 = {
    detail: {
      error: 'class_still_referenced',
      message: 'class "widget_a" is still referenced by 12 item(s)',
      class_id: 8,
      item_count: 12,
      confirmed_label_count: 3,
    },
  };

  it('parses the structured 409 from deprecate', () => {
    const e = new ApiError(409, `${API_PREFIX}/classes/8/deprecate`, body409);
    expect(classStillReferencedDetail(e)).toEqual({
      error: 'class_still_referenced',
      message: 'class "widget_a" is still referenced by 12 item(s)',
      class_id: 8,
      item_count: 12,
      confirmed_label_count: 3,
    });
  });

  it('ignores other statuses', () => {
    expect(classStillReferencedDetail(new ApiError(422, '/x', body409))).toBeNull();
  });

  it("ignores restore's plain-string 409 detail — never confused for the structured shape", () => {
    const plainStringBody = { detail: 'a live class already uses the name "widget_a"' };
    const e = new ApiError(409, `${API_PREFIX}/classes/8/restore`, plainStringBody);
    expect(classStillReferencedDetail(e)).toBeNull();
    // The plain string still reaches ApiError.detail verbatim, which is
    // what the restore call site shows in its toast.
    expect(e.detail).toBe('a live class already uses the name "widget_a"');
  });

  it('ignores a non-ApiError and a non-object detail', () => {
    expect(classStillReferencedDetail(new Error('x'))).toBeNull();
    expect(
      classStillReferencedDetail(new ApiError(409, '/x', { detail: 'plain text' })),
    ).toBeNull();
  });

  it('requires every field to be present with the right type', () => {
    const partial = {
      detail: { error: 'class_still_referenced', message: 'x', class_id: 8 },
    };
    expect(classStillReferencedDetail(new ApiError(409, '/x', partial))).toBeNull();
  });
});

describe('classMergedDetail (F-56, OpenProcessor 4c125ec)', () => {
  const body = {
    detail: {
      error: 'class_merged',
      message: 'class_id 5 was merged into class_id 1 and cannot be restored',
      class_id: 5,
      merged_into: { class_id: 1, class_name: 'widget_a' },
      hint: 'relabel its former crops by hand instead',
    },
  };

  it('parses the structured restore 409 and renders the served name, message and hint', () => {
    const d = classMergedDetail(
      new ApiError(409, `${API_PREFIX}/classes/5/restore`, body),
    );
    expect(d?.merged_into).toEqual({ class_id: 1, class_name: 'widget_a' });
    expect(classMergedRestoreText(d!)).toBe(
      "Merged into widget_a; un-merge isn't supported. " +
        'class_id 5 was merged into class_id 1 and cannot be restored ' +
        'relabel its former crops by hand instead',
    );
  });

  it('ignores a plain-string 409 and a non-409', () => {
    expect(
      classMergedDetail(new ApiError(409, 'x', { detail: 'name taken' })),
    ).toBeNull();
    expect(classMergedDetail(new ApiError(400, 'x', body))).toBeNull();
  });
});
