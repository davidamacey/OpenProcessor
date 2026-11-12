/**
 * `types_images.ts` vs the generated contract: the image page, its cards, counts and boxes carry
 * exactly the served keys.
 */
import { describe, expect, it } from 'vitest';
import spec from '$contracts/openapi/curation.json';
import type * as T from '$lib/types_images';

type Schema = { properties?: Record<string, { enum?: string[] }> };
const schemas = (spec as unknown as { components: { schemas: Record<string, Schema> } })
  .components.schemas;
const keys = <K extends string>(o: Record<K, true>) => Object.keys(o).sort();
const served = (name: string) => Object.keys(schemas[name]!.properties ?? {}).sort();

describe('image contract', () => {
  it('ImagePage has the declared keys', () => {
    expect(served('ImagePage')).toEqual(
      keys<keyof T.ImagePage>({
        items: true,
        next_cursor: true,
        total: true,
        limit: true,
        join_truncated: true,
        sort_applied: true,
        sort_options: true,
        hidden: true,
        empty_reason: true,
        empty_message: true,
      }),
    );
  });

  it('ImageCard has the declared keys', () => {
    expect(served('ImageCard')).toEqual(
      keys<keyof T.ImageCardWire>({
        image_id: true,
        file_name: true,
        thumbnail_url: true,
        width: true,
        height: true,
        source: true,
        indexed_at: true,
        taken_at: true,
        has_location: true,
        counts: true,
        near_dup: true,
        boxes: true,
        boxes_truncated: true,
      }),
    );
  });

  it('ImageCounts, ImageNearDup and BoxView have the declared keys', () => {
    expect(served('ImageCounts')).toEqual(
      keys<keyof T.ImageCounts>({
        items: true,
        labeled: true,
        validated: true,
        unlabeled: true,
        in_holdout: true,
      }),
    );
    expect(served('ImageNearDup')).toEqual(
      keys<keyof T.ImageNearDup>({
        group_id: true,
        purpose: true,
        size: true,
        is_representative: true,
      }),
    );
    expect(served('BoxView')).toEqual(
      keys<keyof T.ImageBox>({
        crop_id: true,
        box_id: true,
        kind: true,
        bbox_norm: true,
        label: true,
        label_kind: true,
        class_name: true,
        tone: true,
        state: true,
        locked: true,
        lock_label: true,
        geometry_revision: true,
      }),
    );
  });

  it('the empty reasons are the served ids', () => {
    const served = schemas.ImagePage!.properties!.empty_reason as unknown as {
      anyOf: { enum?: string[] }[];
    };
    const ids = served.anyOf.flatMap((a) => a.enum ?? []);
    const mirror: T.ImageEmptyReason[] = ['no_images', 'no_match'];
    expect(ids.sort()).toEqual(mirror.sort());
  });
});
