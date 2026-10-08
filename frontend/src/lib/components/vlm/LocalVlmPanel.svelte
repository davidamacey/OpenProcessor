<script lang="ts">
  /**
   * The local-model catalog (`GET /vlm/catalog`) and what the server is
   * serving (`local`). When no local model is configured, only the served
   * `reason` shows. Otherwise: a table of the served entries, a Switch
   * button on every entry that is not being served (confirm; a served
   * `vlm_catalog_does_not_fit` refusal offers "Select anyway", which
   * resends with `force`), and, while a restart is required, a banner
   * with the served reason, the copyable command and "Cancel the request".
   *
   * Nothing here says the model switched: a Switch only records the wish,
   * and an entry reads "serving" when the server says so.
   */
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import type { VlmCatalogEntry, VlmCatalogResponse } from '$lib/types_vlm';
  import { externalHref } from '$lib/mlflowLink';

  interface Props {
    catalog: VlmCatalogResponse;
    busy: boolean;
    /** The last refused select/cancel, as served, and its code. */
    error: string | null;
    errorCode: string | null;
    onselect: (catalogId: string, force: boolean) => Promise<boolean>;
    onclear: () => Promise<boolean>;
  }

  let { catalog, busy, error, errorCode, onselect, onclear }: Props = $props();

  const local = $derived(catalog.local);
  let switching = $state<VlmCatalogEntry | null>(null);
  let force = $state(false);
  let cancelling = $state(false);
  let copied = $state(false);

  const yn = (v: boolean | null): string => (v == null ? '—' : v ? 'yes' : 'no');
  const num = (v: number | null): string => (v == null ? '—' : String(v));

  function openSwitch(e: VlmCatalogEntry): void {
    force = false;
    switching = e;
  }

  async function confirmSwitch(): Promise<void> {
    const e = switching;
    if (!e) return;
    if (await onselect(e.id, force)) switching = null;
  }

  async function confirmCancel(): Promise<void> {
    if (await onclear()) cancelling = false;
  }

  async function copyCommand(command: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(command);
      copied = true;
      setTimeout(() => (copied = false), 2000);
    } catch {
      copied = false;
    }
  }
</script>

<div class="flex flex-col gap-3" data-testid="local-vlm">
  {#if !local.configured}
    <p class="text-sm text-zinc-400" data-testid="local-vlm-reason">{local.reason}</p>
  {:else}
    <p class="text-xs text-zinc-400" data-testid="local-vlm-served">
      Serving:
      <span class="font-mono text-zinc-200">{local.served?.model ?? '—'}</span>
      {#if local.served?.catalog_id}
        (catalog <span class="font-mono">{local.served.catalog_id}</span>)
      {/if}
      {#if local.gpu_total_gb != null}
        · GPU memory {local.gpu_total_gb} GB
      {/if}
    </p>

    {#if local.restart_required}
      <div
        class="flex flex-col gap-2 rounded border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-100"
        data-testid="local-vlm-restart"
      >
        <p>{local.reason}</p>
        {#if local.desired}
          <p class="text-xs">
            Requested: <span class="font-mono">{local.desired.catalog_id}</span>
          </p>
          <div class="flex flex-wrap items-center gap-2">
            <code
              class="break-all rounded bg-zinc-950 px-2 py-1 font-mono text-xs text-zinc-200"
              data-testid="local-vlm-command">{local.desired.command}</code
            >
            <button
              type="button"
              class="btn btn-sm"
              data-testid="local-vlm-copy"
              onclick={() => void copyCommand(local.desired!.command)}
              >{copied ? 'Copied' : 'Copy'}</button
            >
          </div>
        {/if}
        <div>
          <button
            type="button"
            class="btn btn-sm"
            disabled={busy}
            data-testid="local-vlm-cancel"
            onclick={() => (cancelling = true)}>Cancel the request</button
          >
        </div>
      </div>
    {/if}

    {#if error && !switching && !cancelling}
      <p class="text-sm text-red-300" data-testid="local-vlm-error">{error}</p>
    {/if}

    <div class="overflow-x-auto">
      <table class="w-full text-left text-xs" data-testid="local-vlm-table">
        <thead class="text-zinc-500">
          <tr>
            <th class="py-1 pr-3 font-normal">Model</th>
            <th class="py-1 pr-3 font-normal">Repo</th>
            <th class="py-1 pr-3 font-normal">License</th>
            <th class="py-1 pr-3 font-normal">Size</th>
            <th class="py-1 pr-3 font-normal">Memory (GB)</th>
            <th class="py-1 pr-3 font-normal">Context</th>
            <th class="py-1 pr-3 font-normal">Max images</th>
            <th class="py-1 pr-3 font-normal">Status</th>
            <th class="py-1 pr-3 font-normal">Fits</th>
            <th class="py-1 pr-3 font-normal">Multi-box</th>
            <th class="py-1 pr-3 font-normal">Text reading</th>
            <th class="py-1 font-normal"></th>
          </tr>
        </thead>
        <tbody>
          {#each catalog.entries as e (e.id)}
            <tr
              class="border-t border-zinc-800 align-top"
              data-testid="local-vlm-row"
              data-id={e.id}
            >
              <td class="py-1.5 pr-3">
                <span class="text-zinc-100">{e.choice.label}</span>
                {#if e.serving}<span
                    class="ml-1 rounded border border-emerald-500/40 bg-emerald-500/10 px-1 text-[10px] text-emerald-300"
                    data-testid="local-vlm-serving">serving</span
                  >{/if}
                {#if e.desired}<span
                    class="ml-1 rounded border border-amber-500/40 bg-amber-500/10 px-1 text-[10px] text-amber-200"
                    data-testid="local-vlm-desired">requested</span
                  >{/if}
              </td>
              <td class="py-1.5 pr-3 font-mono break-all">{e.hf_repo}</td>
              <td class="py-1.5 pr-3">
                {#if externalHref(e.license_url)}
                  <!-- eslint-disable svelte/no-navigation-without-resolve -- external license page, not a SvelteKit route -->
                  <a
                    class="text-blue-300 hover:underline"
                    href={externalHref(e.license_url)}
                    target="_blank"
                    rel="noreferrer noopener">{e.license}</a
                  >
                  <!-- eslint-enable svelte/no-navigation-without-resolve -->
                {:else}
                  {e.license}
                {/if}{#if e.gated}<span class="ml-1 text-amber-300">gated</span>{/if}
              </td>
              <td class="py-1.5 pr-3 font-mono"
                >{e.params_b == null ? '—' : `${e.params_b}B`}{e.quantization
                  ? ` ${e.quantization}`
                  : ''}</td
              >
              <td class="py-1.5 pr-3 font-mono">{e.vram_gb} / {num(e.disk_gb)} disk</td>
              <td class="py-1.5 pr-3 font-mono">{e.context_max} / {e.max_model_len}</td>
              <td class="py-1.5 pr-3 font-mono">{e.max_images}</td>
              <td class="py-1.5 pr-3">{catalog.labels.status[e.status] ?? e.status}</td>
              <td class="py-1.5 pr-3 font-mono">{yn(e.fits)}</td>
              <td class="py-1.5 pr-3 font-mono">{yn(e.multi_box_verified)}</td>
              <td class="py-1.5 pr-3 font-mono">{yn(e.text_reading_verified)}</td>
              <td class="py-1.5">
                {#if !e.serving}
                  <button
                    type="button"
                    class="btn btn-sm"
                    disabled={busy}
                    data-testid="local-vlm-switch"
                    onclick={() => openSwitch(e)}>Switch</button
                  >
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

{#if switching}
  <ConfirmDialog
    title="Switch the local model to {switching.choice.label}"
    confirmLabel={force ? 'Select anyway' : 'Switch'}
    danger={force}
    {busy}
    onconfirm={() => void confirmSwitch()}
    oncancel={() => (switching = null)}
  >
    <p class="text-xs text-zinc-400">
      This records a request. The server keeps serving the current model until it is
      restarted.
    </p>
    {#if error}<p class="text-red-300" data-testid="local-vlm-switch-error">
        {error}
      </p>{/if}
    {#if errorCode === 'vlm_catalog_does_not_fit'}
      <label class="flex items-center gap-2 text-xs text-amber-200">
        <input type="checkbox" bind:checked={force} data-testid="local-vlm-force" />
        Select anyway (the server says this does not fit the GPU)
      </label>
    {/if}
  </ConfirmDialog>
{/if}

{#if cancelling}
  <ConfirmDialog
    title="Cancel the local-model request"
    confirmLabel="Cancel the request"
    danger
    {busy}
    onconfirm={() => void confirmCancel()}
    oncancel={() => (cancelling = false)}
  >
    <p class="text-xs text-zinc-400">
      The server keeps serving its current model; the pending request is dropped.
    </p>
    {#if error}<p class="text-red-300" data-testid="local-vlm-cancel-error">
        {error}
      </p>{/if}
  </ConfirmDialog>
{/if}
