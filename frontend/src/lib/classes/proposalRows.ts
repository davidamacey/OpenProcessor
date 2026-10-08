/**
 * Rows for `/classes`' New-class-proposals list (visual audit 2026-09-24,
 * L2). The served summary splits terms into `top_terms` (create-able) and
 * `flagged_terms` (existing class / generic parent / not an object). The
 * page used to render the flagged ones in a collapsed section with no
 * actions at all, although they held most of the pending crops
 * (the top few terms accounted for the bulk of them), while about 80 one- and
 * two-crop terms each got the full Create/Map controls above them.
 *
 * Now every term is one list, biggest first, and every term can be
 * resolved. Only "Create class" stays limited to un-flagged terms, and the
 * backend's own flag still decides which terms those are. An
 * `existing_class` term keeps its one-click map to the served `class_id`.
 */
import type {
  NewClassProposalTerm,
  NewClassProposalsSummary,
  NewClassTermRules,
} from '$lib/api';

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

/**
 * The help line explaining which terms get auto-flagged, built from the
 * served `term_rules`. F-52: an empty served list used to render as
 * "(, plus …)" / "()"; each clause now appears only when it has content.
 * Returns null when nothing is configured.
 */
export function termRulesText(rules: NewClassTermRules): string | null {
  const generic = rules.generic_terms.filter((t) => t.trim().length > 0);
  const nonObject = rules.non_object_terms.filter((t) => t.trim().length > 0);
  const genericParts: string[] = [];
  if (generic.length > 0) genericParts.push(generic.join(', '));
  if (rules.registry_groups_are_generic)
    genericParts.push('any class-registry group name');
  const clauses: string[] = [];
  if (genericParts.length > 0) {
    clauses.push(`match a generic-parent term (${genericParts.join(', plus ')})`);
  }
  if (nonObject.length > 0) {
    clauses.push(`match a non-object term (${nonObject.join(', ')})`);
  }
  if (rules.existing_classes_flagged) clauses.push('already name a registered class');
  if (clauses.length === 0) return null;
  const last = clauses.pop()!;
  const joined = clauses.length > 0 ? `${clauses.join(', ')}, or ${last}` : last;
  return `Terms are auto-flagged, not offered a one-click create, when they ${joined}.`;
}
