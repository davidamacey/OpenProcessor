/**
 * The /classes new-class proposals section: the served summary of
 * VLM-proposed terms and the per-term resolve actions (create a class, or
 * map onto an existing one), each dry-run first for a served confirm count.
 */
import { apiErrorText } from '$lib/api_errors';
import {
  getNewClassProposalsSummary,
  resolveNewClassProposal,
  type NewClassProposalsSummary,
  type NewClassProposalTerm,
} from '$lib/api_review';
import type { ResolveNewClassRequest, ResolveNewClassResponse } from '$lib/types_review';
import { classesStore } from '$stores/classes.svelte';
import { toastStore } from '$stores/toast.svelte';
import { undoStore } from '$stores/undo.svelte';

export function createNewClassProposalsController() {
  // -- New-class proposals -------------------------------------------------
  //
  // Aggregate view of the same cohort the `/review` "New Class Proposals"
  // tab pages through one crop at a time — top VLM-proposed-but-unmatched
  // terms with counts and a handful of sample crop ids each (GET
  // {API_PREFIX}/review/new_class_proposals/summary, rendered as thumbnails
  // only). The two actions per term — "Create class & assign" / "Map to
  // existing" — now go through `POST {API_PREFIX}/review/new_class_proposals/
  // resolve`, which the backend resolves against **every** pending item
  // proposing the term, not just the summary's capped `sample_crop_ids`
  // preview. Each action dry-runs first (`?dry_run=true`) to show the
  // operator the real served `matched` count in a confirm dialog before
  // writing anything.
  let proposalsSummary = $state<NewClassProposalsSummary | null>(null);
  let proposalsError = $state<string | null>(null);
  let proposalsLoading = $state<boolean>(false);
  // Per-term inline form state, keyed by term label.
  let newClassNameByTerm = $state<Record<string, string>>({});
  let mapTargetByTerm = $state<Record<string, number | null>>({});
  let proposalBusyTerm = $state<string | null>(null);

  async function loadProposals(): Promise<void> {
    proposalsLoading = true;
    proposalsError = null;
    try {
      proposalsSummary = await getNewClassProposalsSummary();
    } catch (e) {
      // Observed live: this aggregate can 500 on an opensearch outage even
      // while the rest of /classes works fine — degrade to an inline error
      // rather than breaking the page.
      proposalsError = apiErrorText(e);
    } finally {
      proposalsLoading = false;
    }
  }

  function dismissProposalTerm(label: string): void {
    if (!proposalsSummary) return;
    proposalsSummary = {
      ...proposalsSummary,
      top_terms: proposalsSummary.top_terms.filter((t) => t.label !== label),
      flagged_terms: proposalsSummary.flagged_terms.filter((t) => t.label !== label),
    };
  }

  /**
   * Record the resolve's `updated_ids` for undo (same ring buffer + `Z`
   * behavior `bulkLabel`/`moveCropsToCluster` already use elsewhere in
   * the app — ONE entry for the whole resolve, so a single `Z` reverts
   * every crop it touched, batched through `undo_batch` when there's
   * more than one id), and toast the served counts. Never a
   * client-computed count: `matched` only ever comes from the dry-run/
   * real response, `updated`/`conflicts`/`skipped` only from the real
   * resolve's response.
   */
  function reportResolve(res: ResolveNewClassResponse, verb: string): void {
    if (res.updated_ids.length > 0) undoStore.recordWrites(res.updated_ids);
    const parts = [`${res.updated} updated`];
    if (res.conflicts.length > 0) parts.push(`${res.conflicts.length} conflict(s)`);
    if (res.skipped.length > 0) parts.push(`${res.skipped.length} skipped`);
    toastStore.success(`${verb} — ${parts.join(', ')}.`);
  }

  async function createClassAndAssign(term: { label: string }): Promise<void> {
    const name = (newClassNameByTerm[term.label] ?? term.label).trim();
    if (!name) {
      toastStore.error('Class name is required.');
      return;
    }
    proposalBusyTerm = term.label;
    try {
      // No client-side slug check — the resolve call 422s on a bad
      // `create.class_name` with the pattern in its detail; that message
      // is what the toast shows on failure.
      const body: ResolveNewClassRequest = {
        label: term.label,
        create: { class_name: name, group: '' },
      };
      const preview = await resolveNewClassProposal(body, { dryRun: true });
      const ok = window.confirm(
        `Create class "${name}" and assign ${preview.matched} crop(s) proposing "${term.label}"?`,
      );
      if (!ok) return;
      const res = await resolveNewClassProposal(body);
      reportResolve(res, `Created "${res.class_name}"`);
      dismissProposalTerm(term.label);
      await Promise.all([loadProposals(), classesStore.clearAndRefetch()]);
    } catch (e) {
      toastStore.error(`Create & assign failed: ${apiErrorText(e)}`);
    } finally {
      proposalBusyTerm = null;
    }
  }

  async function mapToExisting(term: { label: string }): Promise<void> {
    const targetId = mapTargetByTerm[term.label];
    if (targetId == null) {
      toastStore.error('Pick an existing class first.');
      return;
    }
    proposalBusyTerm = term.label;
    try {
      const body: ResolveNewClassRequest = { label: term.label, class_id: targetId };
      const preview = await resolveNewClassProposal(body, { dryRun: true });
      const cls = classesStore.byId(targetId);
      const ok = window.confirm(
        `Assign ${preview.matched} crop(s) proposing "${term.label}" to "${cls?.name ?? targetId}"?`,
      );
      if (!ok) return;
      const res = await resolveNewClassProposal(body);
      reportResolve(res, `Assigned to "${cls?.name ?? res.class_name}"`);
      dismissProposalTerm(term.label);
      await loadProposals();
    } catch (e) {
      toastStore.error(`Assign failed: ${apiErrorText(e)}`);
    } finally {
      proposalBusyTerm = null;
    }
  }

  // DQ-M11 (dq-queues cutover, 2026-09-24): flagged_terms — super-category
  // ('generic_parent'), junk ('non_object') and already-registered
  // ('existing_class') proposed terms. No "create class" action is
  // offered for any of these (that was the original DQ-M11 bug — a
  // one-click create over 89 crops proposing a generic-parent term would
  // have made a super-class). `existing_class` still gets a one-click map action,
  // using the server's own `class_id`, not an operator-picked select.
  function flagReason(term: NewClassProposalTerm): string {
    if (term.flag === 'generic_parent') return 'generic parent';
    if (term.flag === 'non_object') return 'not an object';
    if (term.flag === 'existing_class') {
      const cls = term.class_id != null ? classesStore.byId(term.class_id) : null;
      return `existing class → map to ${cls?.name ?? term.class_id}`;
    }
    return '';
  }

  async function mapFlaggedTermToClass(term: NewClassProposalTerm): Promise<void> {
    if (term.class_id == null) return;
    proposalBusyTerm = term.label;
    try {
      const body: ResolveNewClassRequest = { label: term.label, class_id: term.class_id };
      const preview = await resolveNewClassProposal(body, { dryRun: true });
      const cls = classesStore.byId(term.class_id);
      const ok = window.confirm(
        `Assign ${preview.matched} crop(s) proposing "${term.label}" to "${cls?.name ?? term.class_id}"?`,
      );
      if (!ok) return;
      const res = await resolveNewClassProposal(body);
      reportResolve(res, `Assigned to "${cls?.name ?? res.class_name}"`);
      await loadProposals();
    } catch (e) {
      toastStore.error(`Assign failed: ${apiErrorText(e)}`);
    } finally {
      proposalBusyTerm = null;
    }
  }

  return {
    get proposalsSummary() {
      return proposalsSummary;
    },
    get proposalsError() {
      return proposalsError;
    },
    get proposalsLoading() {
      return proposalsLoading;
    },
    get newClassNameByTerm() {
      return newClassNameByTerm;
    },
    set newClassNameByTerm(v: Record<string, string>) {
      newClassNameByTerm = v;
    },
    get mapTargetByTerm() {
      return mapTargetByTerm;
    },
    set mapTargetByTerm(v: Record<string, number | null>) {
      mapTargetByTerm = v;
    },
    get proposalBusyTerm() {
      return proposalBusyTerm;
    },
    loadProposals,
    dismissProposalTerm,
    createClassAndAssign,
    mapToExisting,
    flagReason,
    mapFlaggedTermToClass,
  };
}

export type NewClassProposalsController = ReturnType<
  typeof createNewClassProposalsController
>;
