/**
 * The core / non-core cut line on `/clusters/[id]`. With `order=core_first`
 * the server sorts nearest-to-centroid first and serves each item's
 * `cluster_is_core` (bool or null); the cut is the first item served as
 * `false`. No client heuristic about null share or order consistency: in any
 * other order the line is not drawn at all.
 */

export interface CutLineCropLike {
  cluster_is_core?: boolean | null;
}

export interface CutLineResult {
  /** Index of the first item served as non-core; 0 when there is none. */
  index: number;
  /** Whether a line is drawn at `index`. */
  visible: boolean;
}

export function computeCutLine(
  crops: readonly CutLineCropLike[],
  orderIsCoreFirst: boolean,
): CutLineResult {
  if (!orderIsCoreFirst) return { index: 0, visible: false };
  const index = crops.findIndex((c) => c.cluster_is_core === false);
  // Nothing precedes index 0 to call "the core section".
  if (index <= 0) return { index: 0, visible: false };
  return { index, visible: true };
}
