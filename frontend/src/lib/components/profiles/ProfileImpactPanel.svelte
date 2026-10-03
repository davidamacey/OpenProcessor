<script lang="ts" module>
  /** One-line meanings of the six documented `ActivationImpact` counts
   *  (§4.6). No labels are served for them yet (W4-Q7). */
  export const IMPACT_ROWS = [
    { key: 'items_total', label: 'Items in the project' },
    { key: 'validated_items', label: 'Human-validated (never re-run)' },
    { key: 'pending_items', label: 'Waiting for detection' },
    { key: 'pending_not_matching', label: 'Waiting, but outside the item classes' },
    { key: 'unseeded_items', label: 'Never queued for a region' },
  ] as const;
</script>

<script lang="ts">
  /**
   * The served impact of the active region profile (`ActivationImpact`,
   * §4.6, §7.3): items by the profile@revision that produced them and the
   * cohorts a profile change never touches. Nothing already processed is
   * rewritten by an activation; a re-run is explicit. When the server
   * suggests one (`suggested_reprocess`, W10) and serves Reprocess, "Re-run…"
   * sends that request exactly as served to `POST /reprocess`: a dry run
   * first, then the apply behind a confirm.
   */
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import ReprocessCounts from '$components/datasets/ReprocessCounts.svelte';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import { reprocessVocabularyStore } from '$lib/stores/reprocessVocabulary.svelte';
  import { ReprocessFlow } from '$lib/datasets/reprocessController.svelte';
  import type { ActivationImpact } from '$lib/types_profiles';

  interface Props {
    impact: ActivationImpact;
  }

  let { impact }: Props = $props();

  $effect(() => {
    if (impact.suggested_reprocess) {
      void datasetsAvailability.init();
      void reprocessVocabularyStore.init();
    }
  });

  const available = $derived(datasetsAvailability.available === true);
  const scopeLabel = (id: string): string => reprocessVocabularyStore.label('scopes', id);

  let flow = $state<ReprocessFlow | null>(null);
  let confirming = $state(false);

  async function startRerun(): Promise<void> {
    const req = impact.suggested_reprocess;
    if (!req) return;
    flow?.destroy();
    const f = new ReprocessFlow({ kind: 'request', request: req });
    flow = f;
    await f.preview();
  }

  async function apply(): Promise<void> {
    const f = flow;
    if (!f) return;
    await f.apply();
    if (f.result) confirming = false;
  }

  $effect(() => () => flow?.destroy());

  function refText(name: string | null, revision: number | null): string {
    if (name == null) return '—';
    return revision == null ? name : `${name} r${revision}`;
  }
</script>

<section
  class="flex flex-col gap-3 rounded border border-zinc-800 bg-zinc-900/40 p-3 text-sm"
  data-testid="profile-impact"
  aria-label="Activation impact"
>
  <p class="text-xs text-zinc-400">
    Items already processed keep their results. Only items still waiting are detected with
    the active profile.
  </p>
  <dl class="grid grid-cols-[1fr_auto] gap-x-4 gap-y-0.5 text-xs sm:max-w-md">
    {#each IMPACT_ROWS as r (r.key)}
      <dt class="text-zinc-400">{r.label}</dt>
      <dd class="text-right font-mono" data-testid="impact-{r.key}">
        {impact[r.key].toLocaleString()}
      </dd>
    {/each}
  </dl>
  {#if impact.by_profile.length > 0}
    <table class="text-xs sm:max-w-md" data-testid="impact-by-profile">
      <thead class="text-left text-zinc-500">
        <tr>
          <th class="py-0.5 pr-3 font-normal">Produced by</th>
          <th class="py-0.5 text-right font-normal">Items</th>
        </tr>
      </thead>
      <tbody>
        {#each impact.by_profile as p, i (i)}
          <tr class="border-t border-zinc-800">
            <td class="py-0.5 pr-3 font-mono">{refText(p.name, p.revision)}</td>
            <td class="py-0.5 text-right font-mono">{p.count.toLocaleString()}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}

  {#if impact.suggested_reprocess && available}
    <div class="flex flex-col gap-2 border-t border-zinc-800 pt-2">
      {#if !flow}
        <button
          type="button"
          class="btn btn-sm self-start"
          data-testid="rerun-open"
          onclick={() => void startRerun()}>Re-run items from other profiles…</button
        >
      {:else}
        {@const f = flow}
        {#if f.busy && !f.dryRun && !f.result}
          <p class="text-xs text-zinc-500">Checking what would re-run…</p>
        {/if}
        {#if f.dryRun && !f.result}
          <div class="space-y-1" data-testid="rerun-dry-run">
            <ReprocessCounts res={f.dryRun} {scopeLabel} />
          </div>
          <div class="flex gap-2">
            <button
              type="button"
              class="btn btn-sm btn-primary"
              disabled={!f.canApply}
              data-testid="rerun-apply"
              onclick={() => (confirming = true)}>Re-run</button
            >
            <button
              type="button"
              class="btn btn-sm"
              onclick={() => {
                f.destroy();
                flow = null;
              }}>Cancel</button
            >
          </div>
        {/if}
        {#if f.result}
          <div class="space-y-1" data-testid="rerun-result">
            <ReprocessCounts res={f.result} {scopeLabel} />
            {#if f.job}
              <p class="text-xs text-zinc-300" data-testid="rerun-job">
                Job <code class="font-mono">{f.job.job_id}</code>:
                {datasetsAvailability.statusLabel(f.job.status)}
                {#if f.job.error}<span class="text-red-300"> — {f.job.error}</span>{/if}
              </p>
            {/if}
          </div>
        {/if}
        {#if f.error && !confirming}
          <p class="text-xs text-red-300" data-testid="rerun-error">{f.error}</p>
        {/if}
      {/if}
    </div>
  {/if}
</section>

{#if confirming && flow?.dryRun}
  <ConfirmDialog
    title="Re-run items"
    confirmLabel="Re-run"
    busy={flow.busy}
    onconfirm={() => void apply()}
    oncancel={() => (confirming = false)}
  >
    <p>Sends the server's suggested re-run with the counts below.</p>
    <ReprocessCounts res={flow.dryRun} {scopeLabel} />
    {#if flow.error}<p class="text-red-300">{flow.error}</p>{/if}
  </ConfirmDialog>
{/if}
