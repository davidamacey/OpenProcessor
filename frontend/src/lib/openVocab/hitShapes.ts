/**
 * Overlay shapes for an open-vocabulary test's hits. Boxes and outlines are
 * already normalised to the image by the server, so they are drawn as
 * served; a dropped hit (`selected: false`) is dimmed.
 */
import type { OverlayShape } from '$lib/configTest/overlayShapes';
import { humanizeId } from '$lib/humanizeId';
import type { OpenVocabTestHit } from '$lib/types_openVocab';

/** What a served `drop_reason` reads as; the raw id goes in a tooltip. */
export function dropReasonText(reason: string | null | undefined): string {
  if (!reason) return '';
  return reason === 'agree_existing'
    ? 'Agrees with an existing item'
    : humanizeId(reason);
}

export function hitShapes(hits: OpenVocabTestHit[]): OverlayShape[] {
  const out: OverlayShape[] = [];
  hits.forEach((h, i) => {
    const label = `hit #${i}`;
    const dropped = h.drop_reason ? ` · dropped: ${dropReasonText(h.drop_reason)}` : '';
    const base = {
      dimmed: !h.selected,
      label,
      title: `${label} · score ${h.score.toFixed(2)}${dropped}`,
    };
    if (h.bbox_norm.length === 4) {
      out.push({
        ...base,
        key: `hit:${i}:box`,
        kind: 'box',
        box: [h.bbox_norm[0]!, h.bbox_norm[1]!, h.bbox_norm[2]!, h.bbox_norm[3]!],
      });
    }
    if (h.mask_polygon && h.mask_polygon.length >= 3) {
      out.push({
        ...base,
        key: `hit:${i}:poly`,
        kind: 'polygon',
        points: h.mask_polygon
          .filter((p) => p.length >= 2)
          .map((p) => [p[0]!, p[1]!] as [number, number]),
      });
    }
  });
  return out;
}
