<script lang="ts" module>
  /** The words a resource puts on the panel. Structure is shared. */
  export interface ActivePanelCopy {
    /** "Active pack" / "Active region profile". */
    title: string;
    /** Shown for `active.name: null` (an explicit `off`, or any source
     *  without a more specific text below). */
    noneText: string;
    /** Shown instead for `active.name: null` with `source: 'env'` (never
     *  activated, and no env default to name). */
    noneEnvText?: string;
    /** Header of the applied-runtime column naming the ref. */
    appliedColumn: string;
    /** Which applied-runtime ref this axis names. */
    appliedRef: 'pack' | 'profile' | 'vlm';
    /** An applied ref the worker reported with `name: null`. */
    appliedNoneText: string;
    rollbackTitle: string;
    rollbackBlurb: string;
    /** Present for a resource with a deactivate route. */
    deactivate?: { label: string; title: string; blurb: string };
  }
</script>

<script lang="ts">
  /**
   * What is active on one config axis, as served by `GET /{resource}/active`
   * (§7.2, §7.3, §7.6 item 4): the active ref, since when, what was active
   * before, whether the server is serving a stale snapshot, and which
   * worker processes have applied it. Rollback (confirm-gated) is offered
   * only when the server names a `previous`; Turn off (confirm-gated) only
   * for a resource with a deactivate route and something active.
   */
  import type { Snippet } from 'svelte';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import type { ConfigActive } from '$lib/config/configActive.svelte';
  import { formatTimestamp } from '$lib/formatDate';
  import type { ActiveConfigResponse, ActiveRef } from '$lib/types_config';

  interface Props {
    ctl: ConfigActive<ActiveConfigResponse>;
    copy: ActivePanelCopy;
    /** Runs the rollback (the list re-reads after it; the editor doesn't). */
    onrollback: () => Promise<boolean>;
    /** Runs the deactivate; required when `copy.deactivate` is set. */
    ondeactivate?: () => Promise<boolean>;
    /** Extra actions in the header row. */
    actions?: Snippet;
    /** Served labels for the `source` ids (a VLM endpoint list's
     *  `labels.source`); an id with no label prints raw. */
    sourceLabels?: Record<string, string> | null;
  }

  let {
    ctl,
    copy,
    onrollback,
    ondeactivate,
    actions,
    sourceLabels = null,
  }: Props = $props();

  let confirming = $state<'rollback' | 'deactivate' | null>(null);

  function noneText(source: string | null | undefined): string {
    return source === 'env' && copy.noneEnvText ? copy.noneEnvText : copy.noneText;
  }

  function refText(ref: ActiveRef | null | undefined): string {
    if (!ref || ref.name == null) return 'none';
    return ref.revision == null ? ref.name : `${ref.name} r${ref.revision}`;
  }

  /** An applied ref: `null` is a worker that never reported the axis
   *  (served only for `vlm`); a null name is a reported "nothing here". */
  function appliedRefText(ref: ActiveRef | null | undefined): string {
    if (ref == null) return 'not reported';
    if (ref.name == null) return copy.appliedNoneText;
    return ref.revision == null ? ref.name : `${ref.name} r${ref.revision}`;
  }

  function open(kind: 'rollback' | 'deactivate'): void {
    ctl.clearAction();
    confirming = kind;
  }

  async function run(): Promise<void> {
    const ok = confirming === 'deactivate' ? await ondeactivate?.() : await onrollback();
    if (ok) confirming = null;
  }
</script>

<section
  class="surface flex flex-col gap-2 p-4 text-sm"
  data-testid="config-active-panel"
  aria-label={copy.title}
>
  {#if ctl.loadError && !ctl.active}
    <p class="text-red-300">Could not read what is active: {ctl.loadError}</p>
  {:else if !ctl.active}
    <p class="text-zinc-500">Loading what is active…</p>
  {:else}
    {@const a = ctl.active}
    <div class="flex flex-wrap items-center gap-2">
      <span class="text-zinc-400">{copy.title}</span>
      {#if a.active.name == null}
        <span class="font-medium text-zinc-200" data-testid="active-ref"
          >{noneText(a.source)}</span
        >
      {:else}
        <span class="font-mono font-medium text-emerald-300" data-testid="active-ref"
          >{refText(a.active)}</span
        >
      {/if}
      {#if a.source}<span class="text-xs text-zinc-500" data-testid="active-source"
          >({sourceLabels?.[a.source] ?? a.source})</span
        >{/if}
      {#if a.activated_at}
        <span class="text-xs text-zinc-500" title={a.activated_at}
          >since {formatTimestamp(a.activated_at)}</span
        >
      {/if}
      <span class="grow"></span>
      {@render actions?.()}
      {#if a.previous}
        <button
          type="button"
          class="btn btn-sm"
          disabled={ctl.busy}
          onclick={() => open('rollback')}>Roll back to {refText(a.previous)}</button
        >
      {/if}
      {#if copy.deactivate && ondeactivate && a.active.name != null}
        <button
          type="button"
          class="btn btn-sm"
          disabled={ctl.busy}
          data-testid="active-deactivate"
          onclick={() => open('deactivate')}>{copy.deactivate.label}</button
        >
      {/if}
    </div>
    {#if a.stale}
      <p
        class="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-xs text-amber-200"
        data-testid="active-stale"
      >
        The server could not refresh its config and is showing its last known state.
      </p>
    {/if}
    {#if a.applied && a.applied.length > 0}
      <div class="overflow-x-auto">
        <table class="text-xs" data-testid="active-applied">
          <thead class="text-left text-zinc-500">
            <tr>
              <th class="py-0.5 pr-3 font-normal">Process</th>
              <th class="py-0.5 pr-3 font-normal">Host</th>
              <th class="py-0.5 pr-3 font-normal">{copy.appliedColumn}</th>
              <th class="py-0.5 pr-3 font-normal">Config revision</th>
              <th class="py-0.5 pr-3 font-normal">Applied at</th>
              <th class="py-0.5 font-normal"></th>
            </tr>
          </thead>
          <tbody>
            {#each a.applied as r (r.process + r.host)}
              <tr class="border-t border-zinc-800">
                <td class="py-0.5 pr-3">{r.process}</td>
                <td class="py-0.5 pr-3 font-mono">{r.host}</td>
                <td class="py-0.5 pr-3 font-mono">{appliedRefText(r[copy.appliedRef])}</td
                >
                <td class="py-0.5 pr-3 font-mono">{r.applied_config_revision}</td>
                <td class="py-0.5 pr-3 font-mono" data-testid="applied-at"
                  >{formatTimestamp(r.applied_at)}</td
                >
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
      </div>
    {/if}
    {#if ctl.actionError && !confirming}
      <p class="text-xs text-red-300" data-testid="active-action-error">
        {ctl.actionError}
      </p>
    {/if}
  {/if}
</section>

{#if confirming === 'rollback' && ctl.active?.previous}
  <ConfirmDialog
    title={copy.rollbackTitle}
    confirmLabel="Roll back"
    busy={ctl.busy}
    onconfirm={() => void run()}
    oncancel={() => (confirming = null)}
  >
    <p>
      <span class="font-mono">{refText(ctl.active.active)}</span>
      →
      <strong class="font-mono">{refText(ctl.active.previous)}</strong>
    </p>
    <p class="text-xs text-zinc-400">{copy.rollbackBlurb}</p>
    {#if ctl.actionError}<p class="text-red-300" data-testid="rollback-error">
        {ctl.actionError}
      </p>{/if}
  </ConfirmDialog>
{/if}

{#if confirming === 'deactivate' && copy.deactivate && ctl.active}
  <ConfirmDialog
    title={copy.deactivate.title}
    confirmLabel={copy.deactivate.label}
    danger
    busy={ctl.busy}
    onconfirm={() => void run()}
    oncancel={() => (confirming = null)}
  >
    <p>
      <span class="font-mono">{refText(ctl.active.active)}</span>
      →
      <strong>{copy.noneText}</strong>
    </p>
    <p class="text-xs text-zinc-400">{copy.deactivate.blurb}</p>
    {#if ctl.actionError}<p class="text-red-300" data-testid="deactivate-error">
        {ctl.actionError}
      </p>{/if}
  </ConfirmDialog>
{/if}
