<script lang="ts">
  import { getThumbUrl } from '$lib/api_cropRead';
  import { proposalRows, termRulesText } from '$lib/classes/proposalRows';
  import type { NewClassProposalsController } from '$lib/classes/newClassProposalsController.svelte';
  import type { RegistryClass } from '$lib/types_classes';
  import { classesStore } from '$stores/classes.svelte';

  let {
    p,
    allClasses,
  }: {
    p: NewClassProposalsController;
    allClasses: RegistryClass[];
  } = $props();
</script>

<!-- New-class proposals — aggregate view of
     the same cohort /review's "New Class Proposals" tab pages one crop
     at a time. Absent (not shown as an error banner) while nothing has
     loaded yet or the pool is empty; shown as an inline error when the
     backend genuinely failed (e.g. the opensearch aggregation 500 seen
     live), never a page-breaking crash. -->
{#if p.proposalsLoading}
  <div class="surface mt-4 p-4 text-xs text-zinc-500">Loading proposals…</div>
{:else if p.proposalsError}
  <div class="surface mt-4 flex items-center gap-3 p-4 text-xs text-red-300">
    <span>Proposals unavailable: {p.proposalsError}</span>
    <button type="button" class="btn-sm" onclick={() => void p.loadProposals()}>
      retry
    </button>
  </div>
{:else if p.proposalsSummary && (p.proposalsSummary.top_terms.length > 0 || p.proposalsSummary.flagged_terms.length > 0)}
  <details class="surface mt-4 p-4" data-testid="proposals-section">
    <summary class="cursor-pointer text-sm font-semibold text-zinc-200">
      New class proposals
      <span class="ml-1 font-normal text-zinc-500"
        >({p.proposalsSummary.total_pending} pending{#if p.proposalsSummary.without_term > 0},
          {p.proposalsSummary.without_term} with no proposed term{/if})</span
      >
    </summary>
    <p class="mb-3 mt-2 text-xs text-zinc-500">
      Crops the VLM flagged as needing a class the registry doesn't have yet. Each row is
      a proposed term with a few sample thumbnails — create a class or map onto an
      existing one to resolve <strong>every</strong> pending crop proposing that term (not
      just the samples shown), after a confirm step showing the real count. Crops can
      still be triaged one at a time on <code>/review</code>'s "New Class Proposals" tab
      instead.
      {#if p.proposalsSummary.without_term > 0}
        {p.proposalsSummary.without_term} pending item(s) have no proposed term (a human "needs
        new class" flag with no name) and aren't listed below — triage those on
        <code>/review</code> instead.
      {/if}
    </p>
    {#if p.proposalsSummary.term_rules}
      {@const rules = termRulesText(p.proposalsSummary.term_rules)}
      {#if rules}
        <p class="mb-3 text-[11px] text-zinc-600" data-testid="proposal-term-rules">
          {rules}
        </p>
      {/if}
    {/if}
    <!-- L2 (visual audit 2026-09-24): one list, biggest term first.
         Flagged terms (generic parent / not an object / existing class)
         used to sit in a collapsed section with no action at all while
         holding most pending crops; they now get "map to existing"
         too. Only "Create class" stays limited to un-flagged terms. -->
    <ul class="flex flex-col gap-3" data-testid="proposal-rows">
      {#each proposalRows(p.proposalsSummary) as row (row.term.label)}
        {@const term = row.term}
        <li
          class="flex flex-wrap items-center gap-3 rounded border border-zinc-800 p-2"
          data-testid="proposal-row"
        >
          <div class="flex shrink-0 items-center gap-1">
            {#each term.sample_crop_ids.slice(0, 4) as cropId (cropId)}
              <img
                src={getThumbUrl(cropId, 64)}
                alt=""
                loading="lazy"
                class="h-10 w-10 rounded object-cover"
              />
            {/each}
          </div>
          <div class="min-w-0 shrink-0">
            <div class="text-sm text-zinc-100">{term.label}</div>
            <div class="text-[11px] text-zinc-500">
              {term.count} crop(s)
              {#if term.flag}
                · <span class="text-amber-300">{p.flagReason(term)}</span>
              {/if}
            </div>
          </div>
          <span class="grow"></span>
          {#if row.canCreate}
            <div class="flex shrink-0 items-center gap-1.5">
              <input
                type="text"
                placeholder={term.label}
                value={p.newClassNameByTerm[term.label] ?? ''}
                oninput={(e) => {
                  p.newClassNameByTerm = {
                    ...p.newClassNameByTerm,
                    [term.label]: (e.currentTarget as HTMLInputElement).value,
                  };
                }}
                class="input-sm w-32"
                disabled={p.proposalBusyTerm === term.label}
              />
              <button
                type="button"
                class="btn-sm btn-primary"
                disabled={p.proposalBusyTerm === term.label}
                onclick={() => void p.createClassAndAssign(term)}
              >
                Create class & assign
              </button>
            </div>
          {/if}
          {#if row.fixedMapClassId != null}
            <button
              type="button"
              class="btn-sm"
              disabled={p.proposalBusyTerm === term.label}
              onclick={() => void p.mapFlaggedTermToClass(term)}
            >
              Map to {classesStore.byId(row.fixedMapClassId)?.name ?? row.fixedMapClassId}
            </button>
          {/if}
          {#if row.canPickMapTarget}
            <div class="flex shrink-0 items-center gap-1.5">
              <select
                class="select-sm"
                aria-label="Map {term.label} to an existing class"
                value={p.mapTargetByTerm[term.label] ?? ''}
                onchange={(e) => {
                  const v = (e.currentTarget as HTMLSelectElement).value;
                  p.mapTargetByTerm = {
                    ...p.mapTargetByTerm,
                    [term.label]: v === '' ? null : Number(v),
                  };
                }}
                disabled={p.proposalBusyTerm === term.label}
              >
                <option value="">map to existing…</option>
                {#each allClasses.filter((c) => !c.deprecated) as cls (cls.id)}
                  <option value={cls.id}>{cls.name}</option>
                {/each}
              </select>
              <button
                type="button"
                class="btn-sm"
                disabled={p.proposalBusyTerm === term.label ||
                  p.mapTargetByTerm[term.label] == null}
                onclick={() => void p.mapToExisting(term)}
              >
                Assign
              </button>
            </div>
          {/if}
          <button
            type="button"
            class="btn-sm btn-icon"
            title="Hide this term from the list until the page reloads. Not saved: the crops stay pending."
            aria-label="Hide {term.label} until reload (not saved)"
            onclick={() => p.dismissProposalTerm(term.label)}
          >
            Hide
          </button>
        </li>
      {/each}
    </ul>
  </details>
{/if}
