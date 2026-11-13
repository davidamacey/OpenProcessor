<script lang="ts">
  /**
   * "Accept confident labels…": the dry run first (total, by class with the audited precision
   * where an accuracy audit exists, and what is held back), then one recorded operation that
   * validates them as a person's decision, with Undo. The numbers are all served.
   */
  import { onDestroy, onMount } from 'svelte';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import { opStatusLine } from '$lib/labelOps/opStatusLine';
  import { createAcceptConfident } from '$lib/review/acceptConfidentController.svelte';
  import type { ItemFilter } from '$lib/types_itemFilter';

  interface Props {
    filter: () => ItemFilter;
    onclose: () => void;
    onapplied: () => void;
  }
  let { filter, onclose, onapplied }: Props = $props();

  const ctl = createAcceptConfident(() => filter());
  onMount(() => void ctl.open());
  onDestroy(() => ctl.dispose());

  const HELD: [keyof NonNullable<typeof ctl.dryRun>['held_back'], string][] = [
    ['holdout', 'in the frozen test holdout'],
    ['audit_sample', 'in an open accuracy audit'],
    ['excluded', 'excluded or dismissed'],
    ['locked', 'already validated'],
    ['deprecated_class', 'carrying a deprecated class'],
  ];
  const held = $derived(
    ctl.dryRun ? HELD.filter(([k]) => ctl.dryRun!.held_back[k] > 0) : [],
  );

  async function apply(): Promise<void> {
    if (await ctl.apply()) {
      onapplied();
      onclose();
    }
  }

  const pct = (v: number): string => `${Math.round(v * 100)}%`;
</script>

<ConfirmDialog
  title="Accept confident labels"
  confirmLabel="Accept {ctl.dryRun?.items.toLocaleString() ?? ''} labels"
  busy={ctl.applying}
  busyLabel={opStatusLine(ctl.job, 'Accepting') ?? 'Working…'}
  confirmDisabled={!ctl.canApply}
  onconfirm={apply}
  oncancel={onclose}
>
  <p>
    Validates the machine labels at or above the bar as your decision. Nothing validated
    by a person or an import is touched, and one Undo reverts the whole operation.
  </p>
  <div class="flex flex-wrap items-center gap-4 text-meta">
    <label class="flex items-center gap-2">
      <span class="text-fg-2">Confidence at least</span>
      <input
        type="number"
        min="0.5"
        max="1"
        step="0.01"
        class="input-sm w-20"
        data-testid="accept-threshold"
        value={ctl.threshold}
        onchange={(e) => ctl.setThreshold(Number(e.currentTarget.value))}
      />
    </label>
    {#each [['detector', 'Detector'], ['vlm', 'VLM']] as [id, label] (id)}
      <label class="flex items-center gap-1.5">
        <input
          type="checkbox"
          checked={ctl.sources.includes(id as 'detector' | 'vlm')}
          data-testid="accept-source-{id}"
          onchange={() => ctl.toggleSource(id as 'detector' | 'vlm')}
        />
        {label}
      </label>
    {/each}
  </div>

  {#if ctl.error}
    <p class="text-danger-text" role="alert" data-testid="accept-error">{ctl.error}</p>
  {:else if !ctl.valid}
    <p class="text-warn-text">Pick a bar from 0.50 to 1.00 and at least one source.</p>
  {:else if ctl.loading && !ctl.dryRun}
    <p class="text-fg-3">Counting…</p>
  {:else if ctl.dryRun}
    <p class="text-fg-1" data-testid="accept-total">
      This accepts <strong class="tabular-nums"
        >{ctl.dryRun.items.toLocaleString()}</strong
      >
      label{ctl.dryRun.items === 1 ? '' : 's'}.
      {#if ctl.dryRun.remaining > 0}
        <span class="text-fg-2"
          >{ctl.dryRun.remaining.toLocaleString()} more are beyond one operation: accept again
          afterwards.</span
        >
      {/if}
    </p>
    {#if ctl.dryRun.by_class.length > 0}
      <table class="w-full text-meta" data-testid="accept-by-class">
        <thead>
          <tr class="text-left text-fg-3">
            <th class="py-1 font-medium">Class</th>
            <th class="py-1 text-right font-medium">Labels</th>
            <th class="py-1 text-right font-medium">Audited precision</th>
          </tr>
        </thead>
        <tbody>
          {#each ctl.dryRun.by_class as row (row.class_name)}
            {@const p = ctl.precision[row.class_name]}
            <tr class="border-t border-line">
              <td class="py-1 text-fg-1">{row.class_name}</td>
              <td class="py-1 text-right tabular-nums">{row.count.toLocaleString()}</td>
              <td class="py-1 text-right tabular-nums text-fg-2">
                {#if p && p.precision != null}
                  {pct(p.precision)}
                  <span class="text-fg-3">({pct(p.ci_low)} to {pct(p.ci_high)})</span>
                {:else}
                  not audited
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
    {#if held.length > 0}
      <p class="text-meta text-fg-2" data-testid="accept-held-back">
        Held back:
        {#each held as [key, label], i (key)}
          {i > 0 ? ', ' : ''}<span class="tabular-nums"
            >{ctl.dryRun.held_back[key].toLocaleString()}</span
          >
          {label}
        {/each}.
      </p>
    {/if}
  {/if}
</ConfirmDialog>
