/**
 * `resolveNewClassProposal` / `undoLabelBatch` (2026-09-24, OpenProcessor
 * 2f5cda2) — bulk resolve for VLM new-class proposals plus the batch
 * undo it's paired with. Verifies the exact wire shapes: method, query
 * string (`dry_run` only sent when true), body (`class_id` XOR
 * `create`), and that the response is passed through / mapped
 * faithfully rather than re-derived client-side.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, API_PREFIX, resolveNewClassProposal, undoLabelBatch } from './api';

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function rawItem(id: string): Record<string, unknown> {
  return { crop_id: id, image_path: `/img/${id}.jpg`, bbox_norm: [0, 0, 1, 1] };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('resolveNewClassProposal', () => {
  it('POSTs to the resolve route with no dry_run query param by default', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        class_id: 12,
        class_name: 'widget_f',
        created: false,
        label: 'widget_f',
        matched: 40,
        matched_ids: [],
        updated: 39,
        updated_ids: ['a', 'b'],
        conflicts: [{ crop_id: 'c', current_source: 'vlm' }],
        skipped: ['d'],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await resolveNewClassProposal({ label: 'widget_f', class_id: 12 });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/review/new_class_proposals/resolve`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ label: 'widget_f', class_id: 12 });
    expect(res.matched).toBe(40);
    expect(res.updated).toBe(39);
    expect(res.updated_ids).toEqual(['a', 'b']);
    expect(res.conflicts).toEqual([{ crop_id: 'c', current_source: 'vlm' }]);
    expect(res.skipped).toEqual(['d']);
  });

  it('adds ?dry_run=true only when dryRun is requested', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        class_id: null,
        class_name: 'widget_f',
        created: false,
        label: 'widget_f',
        matched: 40,
        matched_ids: ['a', 'b'],
        updated: 0,
        updated_ids: [],
        conflicts: [],
        skipped: [],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const body = {
      label: 'widget_f',
      create: { class_name: 'widget_f', group: 'widget_c' },
    };
    const res = await resolveNewClassProposal(body, { dryRun: true });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/review/new_class_proposals/resolve?dry_run=true`);
    expect(JSON.parse(init.body)).toEqual(body);
    expect(res.class_id).toBeNull();
    expect(res.matched).toBe(40);
    expect(res.matched_ids).toEqual(['a', 'b']);
    // dry_run writes/creates nothing.
    expect(res.updated).toBe(0);
    expect(res.updated_ids).toEqual([]);
  });

  it('a request with neither class_id nor create still sends the body verbatim — the 422 is the backend’s to raise', async () => {
    // A fresh Response per call: `fetch`'s body stream can only be read
    // once, and this test reads it twice (once via `.rejects`, once to
    // inspect `.detail`).
    const fetchMock = vi
      .fn()
      .mockImplementation(() =>
        Promise.resolve(
          jsonResponse({ detail: 'exactly one of class_id/create is required' }, 422),
        ),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(resolveNewClassProposal({ label: 'widget_f' })).rejects.toThrow(
      ApiError,
    );
    try {
      await resolveNewClassProposal({ label: 'widget_f' });
      expect.unreachable('should have thrown');
    } catch (e) {
      expect((e as ApiError).status).toBe(422);
      expect((e as ApiError).detail).toBe('exactly one of class_id/create is required');
    }
  });

  it('propagates a 409 (duplicate create.class_name) with the server detail', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ detail: "class 'widget_f' already exists" }, 409),
      );
    vi.stubGlobal('fetch', fetchMock);

    try {
      await resolveNewClassProposal({
        label: 'widget_f',
        create: { class_name: 'widget_f' },
      });
      expect.unreachable('should have thrown');
    } catch (e) {
      expect((e as ApiError).status).toBe(409);
      expect((e as ApiError).detail).toBe("class 'widget_f' already exists");
    }
  });

  it('propagates a 400 (unknown class_id) with the server detail', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ detail: 'unknown class_id 999' }, 400));
    vi.stubGlobal('fetch', fetchMock);

    try {
      await resolveNewClassProposal({ label: 'widget_f', class_id: 999 });
      expect.unreachable('should have thrown');
    } catch (e) {
      expect((e as ApiError).status).toBe(400);
      expect((e as ApiError).detail).toBe('unknown class_id 999');
    }
  });
});

describe('undoLabelBatch', () => {
  it('POSTs {crop_ids} to label/undo_batch and maps returned items', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [rawItem('a'), rawItem('b')],
        undone: 2,
        nothing_to_undo: [],
        conflicts: [],
        not_found: [],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await undoLabelBatch(['a', 'b']);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/label/undo_batch`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ crop_ids: ['a', 'b'] });
    expect(res.undone).toBe(2);
    expect(res.items.map((c) => c.id)).toEqual(['a', 'b']);
  });

  it('defaults missing id lists to empty arrays rather than throwing', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ undone: 0 }));
    vi.stubGlobal('fetch', fetchMock);
    const res = await undoLabelBatch(['a']);
    expect(res.items).toEqual([]);
    expect(res.nothing_to_undo).toEqual([]);
    expect(res.conflicts).toEqual([]);
    expect(res.not_found).toEqual([]);
  });
});
