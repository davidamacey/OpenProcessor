/**
 * Rows for `/classes`' New-class-proposals list (visual audit 2026-09-24,
 * L2). The served summary splits terms into `top_terms` (create-able) and
 * `flagged_terms` (existing class / generic parent / not an object). The
 * page used to render the flagged ones in a collapsed section with no
 * actions at all, although they held most of the pending crops
 * (motorcycle 156, sports_car 46, car 29, ...), while about 80 one- and
 * two-crop terms each got the full Create/Map controls above them.
 *
 * Now every term is one list, biggest first, and every term can be
 * resolved. Only "Create class" stays limited to un-flagged terms, and the
 * backend's own flag still decides which terms those are. An
 * `existing_class` term keeps its one-click map to the served `class_id`.
 */
import type { NewClassProposalTerm, NewClassProposalsSummary } from '$lib/api';

export interface ProposalRow {
  term: NewClassProposalTerm;
  /** Only an un-flagged term may become a new class (DQ-M11). */
  canCreate: boolean;
  /** An `existing_class` term maps straight onto the served class id. */
  fixedMapClassId: number | null;
  /** Every other term can be mapped onto an operator-picked class. */
  canPickMapTarget: boolean;
}

export function proposalRows(summary: NewClassProposalsSummary): ProposalRow[] {
  const rows: ProposalRow[] = [
    ...summary.top_terms.map((term) => ({
      term,
      canCreate: term.flag == null,
      fixedMapClassId: null,
      canPickMapTarget: true,
    })),
    ...summary.flagged_terms.map((term) => {
      const fixed =
        term.flag === 'existing_class' && term.class_id != null ? term.class_id : null;
      return {
        term,
        canCreate: false,
        fixedMapClassId: fixed,
        canPickMapTarget: fixed == null,
      };
    }),
  ];
  return rows.sort(
    (a, b) => b.term.count - a.term.count || a.term.label.localeCompare(b.term.label),
  );
}
