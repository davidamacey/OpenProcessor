/**
 * Extra shapes a test panel asks `SourceImageOverlay` to draw on top of
 * the source image: profile-test candidates (boxes and mask outlines),
 * dropped ones dimmed. Everything is already normalised (0-1) in the
 * frame the overlay draws in; nothing is projected client-side (§7.7).
 */
import { humanizeId } from '$lib/humanizeId';
import type { RegionTestLeg } from '$lib/types_configTest';

export type OverlayShape = {
  key: string;
  dimmed: boolean;
  label: string;
  title: string;
} & (
  | { kind: 'box'; box: [number, number, number, number] }
  | { kind: 'polygon'; points: [number, number][] }
);

const asBox = (b: number[]): [number, number, number, number] | null =>
  b.length === 4 ? [b[0]!, b[1]!, b[2]!, b[3]!] : null;

const asPoints = (p: number[][]): [number, number][] =>
  p.filter((q) => q.length >= 2).map((q) => [q[0]!, q[1]!]);

/** What a candidate is called on screen: its leg and index. */
export const candidateLabel = (leg: string, index: number): string => `${leg} #${index}`;

/** One box (and one polygon when a mask came back) per candidate. A
 *  dropped candidate (`selected: false`) is dimmed. `frame` picks which
 *  served geometry to read: the source image's own (`bbox_norm`,
 *  `mask_polygon`) or the one the server already projected into the
 *  parent crop (`bbox_in_parent`, `mask_polygon_in_parent`). */
export function candidateShapes(
  legs: RegionTestLeg[],
  frame: 'source' | 'parent' = 'source',
): OverlayShape[] {
  const out: OverlayShape[] = [];
  for (const leg of legs) {
    for (const c of leg.candidates ?? []) {
      const label = candidateLabel(leg.leg, c.candidate_index);
      const base = {
        dimmed: !c.selected,
        label,
        title: `${label}${c.drop_reason ? ` · dropped: ${humanizeId(c.drop_reason)}` : ''}`,
      };
      const key = `${leg.leg}:${c.candidate_index}`;
      const rawBox = frame === 'source' ? c.bbox_norm : c.bbox_in_parent;
      const rawPoly = frame === 'source' ? c.mask_polygon : c.mask_polygon_in_parent;
      const box = rawBox ? asBox(rawBox) : null;
      if (box) out.push({ ...base, key: `${key}:box`, kind: 'box', box });
      if (rawPoly && rawPoly.length >= 3) {
        out.push({
          ...base,
          key: `${key}:poly`,
          kind: 'polygon',
          points: asPoints(rawPoly),
        });
      }
    }
  }
  return out;
}
