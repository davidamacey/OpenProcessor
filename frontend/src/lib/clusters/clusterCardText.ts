/**
 * Display text for a cluster card. The server serves the figures; this names
 * them: `label_agreement` (mapped onto `Cluster.dominant_pct`) is the dominant
 * class's share of the LABELLED members.
 */
import type { Cluster, ClusterNameKind } from '$lib/types';
import type { ClusterFlag, ValidationState } from '$lib/types_clusterList';

/** "#7", or "" when the serving partition gave the cluster no number. */
export function displayNoText(c: { display_no?: number | null }): string {
  return c.display_no == null ? '' : `#${c.display_no}`;
}

/** What a cluster's name is, in plain words, when it is not a class name; null when it is. */
export function displayNameNote(kind: ClusterNameKind): string | null {
  return kind === 'detector' ? 'from the detector' : null;
}

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

/** "100% of labeled", or null when the server sent no label share. */
export function dominantShareText(c: Pick<Cluster, 'dominant_pct'>): string | null {
  return c.dominant_pct == null ? null : `${pct(c.dominant_pct)} of labeled`;
}

/** Tooltip for the dominant-share figure: the served counts behind it. */
export function dominantShareTitle(
  c: Pick<Cluster, 'dominant_count' | 'labelled_count' | 'size' | 'dominant_class_name'>,
): string {
  const name = c.dominant_class_name ?? 'dominant class';
  if (c.dominant_count != null && c.labelled_count != null) {
    return `${c.dominant_count} of ${c.labelled_count} labeled members are ${name} (cluster size ${c.size})`;
  }
  return `Share of labeled members that are ${name}`;
}

/** The words for the three served validation states. */
export const VALIDATION_TEXT: Record<ValidationState, string> = {
  none: 'not validated',
  partial: 'partly validated',
  complete: 'validated',
};

/** "all: 7 · outliers: 3": the served per-tab review counts. */
export function reviewCountsTitle(counts: Record<string, number> | undefined): string {
  const parts = Object.entries(counts ?? {}).map(([tab, n]) => `${tab}: ${n}`);
  return parts.length > 0 ? `Members in each review tab. ${parts.join(' · ')}` : '';
}

export interface CardStatus {
  /** The words on the chip: a served flag label, or the served review count. */
  label: string;
  tone: 'warn' | 'info';
  /** The chip's tooltip: the served flag description or the review counts. */
  title: string;
}

/**
 * The one status a cluster card shows: the highest of needs review, mixed detector classes and
 * unvalidated, in a warn or info tone (never danger). Labels and descriptions are the served
 * flag vocabulary; nothing else about a cluster is a status.
 */
export function cardStatus(
  c: Pick<Cluster, 'needs_review_count' | 'review_counts' | 'flags' | 'validation_state'>,
  vocabulary: readonly ClusterFlag[],
): CardStatus | null {
  const flag = (id: string) => vocabulary.find((f) => f.id === id);
  if (c.needs_review_count)
    return {
      label: `${c.needs_review_count.toLocaleString()} to review`,
      tone: 'warn',
      title: reviewCountsTitle(c.review_counts),
    };
  const mixed = c.flags?.includes('mixed_detector_classes')
    ? flag('mixed_detector_classes')
    : null;
  if (mixed) return { label: mixed.label, tone: 'warn', title: mixed.description };
  if (c.validation_state === 'none') {
    const un = flag('unvalidated');
    return {
      label: un?.label ?? VALIDATION_TEXT.none,
      tone: 'info',
      title: un?.description ?? '',
    };
  }
  return null;
}

/** "100% labelled · detector: cup 50%", from the served shares; empty when neither is served. */
export function cardMeta(
  c: Pick<Cluster, 'labelled_share' | 'detector_top_class' | 'detector_top_share'>,
): string {
  const parts: string[] = [];
  if (c.labelled_share != null) parts.push(`${pct(c.labelled_share)} labelled`);
  if (c.detector_top_class) {
    const share = c.detector_top_share == null ? '' : ` ${pct(c.detector_top_share)}`;
    parts.push(`detector: ${c.detector_top_class}${share}`);
  }
  return parts.join(' \u00b7 ');
}

/** The card's hover text: what the page keeps out of the card (agreement, confidence, counts). */
export function cardHoverText(
  c: Pick<
    Cluster,
    | 'dominant_pct'
    | 'detector_conf_mean'
    | 'validated_count'
    | 'size'
    | 'near_dup_group_count'
    | 'display_name'
  >,
): string {
  const parts: string[] = [];
  if (c.dominant_pct != null) parts.push(`agreement ${pct(c.dominant_pct)}`);
  if (c.detector_conf_mean != null)
    parts.push(`detector confidence ${c.detector_conf_mean.toFixed(2)}`);
  parts.push(`${c.validated_count} of ${c.size} validated`);
  if (c.near_dup_group_count > 0)
    parts.push(
      `${c.near_dup_group_count} duplicate group${c.near_dup_group_count === 1 ? '' : 's'}`,
    );
  return parts.join(' \u00b7 ');
}
