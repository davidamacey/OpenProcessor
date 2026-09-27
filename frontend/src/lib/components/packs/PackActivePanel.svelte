<script lang="ts">
  /**
   * The project's active prompt pack as served by `GET /prompt_packs/active`
   * (§7.2, §7.6 item 4): what is active, since when, what was active
   * before, whether the server is serving a stale snapshot, and which
   * worker processes have applied it. Rollback (confirm-gated) is offered
   * only when the server names a `previous`.
   */
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import { formatTimestamp } from '$lib/formatDate';
  import type { PackActive } from '$lib/packs/packActive.svelte';
  import type { ActiveRef } from '$lib/types_packs';

  interface Props {
    ctl: PackActive;
    /** Runs the rollback (the list re-reads after it; the editor doesn't). */
    onrollback: () => Promise<boolean>;
  }

  let { ctl, onrollback }: Props = $props();

  let confirming = $state(false);

  function refText(ref: ActiveRef | null | undefined): string {
    if (!ref || ref.name == null) return 'none';
    return ref.revision == null ? ref.name : `${ref.name} r${ref.revision}`;
  }

  function openRollback(): void {
    ctl.clearAction();
    confirming = true;
  }

  async function doRollback(): Promise<void> {
    if (await onrollback()) confirming = false;
  }
</script>

<section
  class="surface flex flex-col gap-2 p-4 text-sm"
  data-testid="pack-active-panel"
  aria-label="Active prompt pack"
>
  {#if ctl.loadError && !ctl.active}
    <p class="text-red-300">Could not read the active pack: {ctl.loadError}</p>
  {:else if !ctl.active}
    <p class="text-zinc-500">Loading the active pack…</p>
  {:else}
    {@const a = ctl.active}
    <div class="flex flex-wrap items-center gap-2">
      <span class="text-zinc-400">Active pack</span>
      {#if a.active.name == null}
        <span class="font-medium text-zinc-200" data-testid="active-ref"
          >none: the deployment default applies</span
        >
      {:else}
        <span class="font-mono font-medium text-emerald-300" data-testid="active-ref"
          >{refText(a.active)}</span
        >
      {/if}
      {#if a.source}<span class="text-xs text-zinc-500">({a.source})</span>{/if}
      {#if a.activated_at}
        <span class="text-xs text-zinc-500" title={a.activated_at}
          >since {formatTimestamp(a.activated_at)}</span
        >
      {/if}
      <span class="grow"></span>
      {#if a.previous}
        <button
          type="button"
          class="btn btn-sm"
          disabled={ctl.busy}
          onclick={openRollback}>Roll back to {refText(a.previous)}</button
        >
      {/if}
    </div>
    {#if a.stale}
      <p
        class="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-xs text-amber-200"
        data-testid="active-stale"
      >
        The server could not refresh its config and is showing its last known ctl.
      </p>
    {/if}
    {#if a.applied && a.applied.length > 0}
      <table class="text-xs" data-testid="active-applied">
        <thead class="text-left text-zinc-500">
          <tr>
            <th class="py-0.5 pr-3 font-normal">Process</th>
            <th class="py-0.5 pr-3 font-normal">Host</th>
            <th class="py-0.5 pr-3 font-normal">Pack</th>
            <th class="py-0.5 pr-3 font-normal">Config revision</th>
            <th class="py-0.5 font-normal"></th>
          </tr>
        </thead>
        <tbody>
          {#each a.applied as r (r.process + r.host)}
            <tr class="border-t border-zinc-800">
              <td class="py-0.5 pr-3">{r.process}</td>
              <td class="py-0.5 pr-3 font-mono">{r.host}</td>
              <td class="py-0.5 pr-3 font-mono">{refText(r.pack)}</td>
              <td class="py-0.5 pr-3 font-mono">{r.applied_config_revision}</td>
              <td class="py-0.5">
                {#if r.lagging}
                  <span
                    class="rounded border border-amber-500/40 bg-amber-500/10 px-1.5 text-amber-200"
                    data-testid="applied-lagging">lagging</span
                  >
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
    {#if ctl.actionError && !confirming}
      <p class="text-xs text-red-300" data-testid="active-action-error">
        {ctl.actionError}
      </p>
    {/if}
  {/if}
</section>

{#if confirming && ctl.active?.previous}
  <ConfirmDialog
    title="Roll back the active pack"
    confirmLabel="Roll back"
    busy={ctl.busy}
    onconfirm={() => void doRollback()}
    oncancel={() => (confirming = false)}
  >
    <p>
      <span class="font-mono">{refText(ctl.active.active)}</span>
      →
      <strong class="font-mono">{refText(ctl.active.previous)}</strong>
    </p>
    <p class="text-xs text-zinc-400">
      Every VLM step that doesn't pick its own pack uses the active pack. Workers switch
      at their next quiet point.
    </p>
    {#if ctl.actionError}<p class="text-red-300" data-testid="rollback-error">
        {ctl.actionError}
      </p>{/if}
  </ConfirmDialog>
{/if}
