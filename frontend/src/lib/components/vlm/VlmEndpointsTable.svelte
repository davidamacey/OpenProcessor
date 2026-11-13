<script lang="ts">
  /**
   * The registry's endpoints, as served (`GET /vlm/endpoints`), in served
   * order: name (link to the editor), source / status / locality through
   * the served labels, model, URL, catalog id, key reference (and whether
   * the host has it; never a key), a red chip with the served warning for
   * an endpoint that sends crops outside the deployment, last probe,
   * revision, the projects running it (this one marked), and the row
   * actions. Delete is offered for stored rows only; an env row is
   * read-only. The registry is shared by every project; activation is per
   * project.
   */
  import { resolve } from '$app/paths';
  import { formatTimestamp } from '$lib/format/date';
  import { projectHref } from '$lib/projectPaths';
  import type {
    VlmEndpointList,
    VlmEndpointSummary,
    VlmProbeResult,
  } from '$lib/types_vlm';
  import VlmProbePanel from './VlmProbePanel.svelte';

  interface Props {
    list: VlmEndpointList;
    /** This project's slug, marked in `active_in`. */
    currentSlug: string | null;
    probes: Record<string, VlmProbeResult>;
    probeErrors: Record<string, string>;
    /** The endpoint a probe is running for. */
    probing: string | null;
    busy: boolean;
    onactivate: (row: VlmEndpointSummary) => void;
    onprobe: (row: VlmEndpointSummary) => void;
    onclone: (row: VlmEndpointSummary) => void;
    ondelete: (row: VlmEndpointSummary) => void;
  }

  let {
    list,
    currentSlug,
    probes,
    probeErrors,
    probing,
    busy,
    onactivate,
    onprobe,
    onclone,
    ondelete,
  }: Props = $props();

  const label = (map: Record<string, string>, id: string | null): string =>
    id == null ? '—' : (map[id] ?? id);
</script>

<div class="flex flex-col gap-2" data-testid="vlm-endpoints">
  <p class="text-meta text-fg-2">
    The registry is shared by every project; activation is per project. External policy:
    <span class="text-fg-1" data-testid="vlm-external-policy">{list.external_policy}</span
    >.
  </p>
  <div class="overflow-x-auto">
    <table class="w-full text-left text-body" data-testid="vlm-endpoints-table">
      <thead class="text-meta text-fg-3">
        <tr>
          <th class="py-1 pr-3 font-normal">Name</th>
          <th class="py-1 pr-3 font-normal">Status</th>
          <th class="py-1 pr-3 font-normal">Where</th>
          <th class="py-1 pr-3 font-normal">Model</th>
          <th class="py-1 pr-3 font-normal">Connection</th>
          <th class="py-1 pr-3 font-normal">Last probe</th>
          <th class="py-1 pr-3 font-normal">Used by</th>
          <th class="py-1 font-normal"><span class="sr-only">Actions</span></th>
        </tr>
      </thead>
      <tbody>
        {#each list.endpoints as e (e.name)}
          <tr
            class="border-t border-line align-top"
            data-testid="vlm-endpoint-row"
            data-name={e.name}
          >
            <td class="py-1.5 pr-3">
              <a
                class="text-accent-text hover:underline"
                href={resolve(
                  projectHref(`/settings/models/vlm/${encodeURIComponent(e.name)}`),
                )}><code>{e.name}</code></a
              >
              <span class="block text-meta text-fg-3"
                >{label(list.labels.source, e.source)}{e.revision != null
                  ? ` · r${e.revision}`
                  : ''}</span
              >
              {#if e.read_only}
                <span
                  class="rounded border border-line-strong px-1 text-caption text-fg-2"
                  data-testid="vlm-read-only">read-only</span
                >
              {/if}
            </td>
            <td class="py-1.5 pr-3 text-meta" data-testid="vlm-endpoint-status"
              >{label(list.labels.status, e.status)}</td
            >
            <td class="py-1.5 pr-3 text-meta">
              {label(list.labels.locality, e.locality)}
              {#if e.sends_images_externally}
                <span
                  class="mt-1 block rounded border border-danger/40 bg-danger/10 px-1.5 py-0.5 text-caption text-danger-text"
                  data-testid="vlm-external-warning"
                  >{e.warning ?? 'Sends crops outside this deployment.'}</span
                >
              {/if}
            </td>
            <td class="py-1.5 pr-3 text-meta"><code>{e.model}</code></td>
            <td class="py-1.5 pr-3 text-meta">
              <code class="block break-all text-fg-2">{e.base_url}</code>
              {#if e.catalog_id}<span class="block text-fg-3"
                  >catalog <code>{e.catalog_id}</code></span
                >{/if}
              {#if e.api_key_ref}
                <span class="block text-fg-3" data-testid="vlm-key-ref"
                  >key <code>{e.api_key_ref}</code>
                  {e.api_key_present
                    ? '(present on the host)'
                    : '(not found on the host)'}</span
                >
              {/if}
            </td>
            <td class="py-1.5 pr-3 text-meta text-fg-2" title={e.last_probe_at ?? ''}
              >{formatTimestamp(e.last_probe_at)}</td
            >
            <td class="py-1.5 pr-3 text-meta" data-testid="vlm-active-in">
              {#each e.active_in ?? [] as slug (slug)}
                <code
                  class="mr-1 inline-block rounded border px-1 {slug === currentSlug
                    ? 'border-success/40 bg-success/10 text-success-text'
                    : 'border-line-strong text-fg-2'}"
                  >{slug}{slug === currentSlug ? ' (this project)' : ''}</code
                >
              {:else}
                <span class="text-fg-3">none</span>
              {/each}
            </td>
            <td class="py-1.5">
              <div class="flex flex-wrap justify-end gap-1">
                <button
                  type="button"
                  class="btn btn-sm"
                  disabled={busy}
                  data-testid="vlm-activate"
                  onclick={() => onactivate(e)}>Activate here</button
                >
                <button
                  type="button"
                  class="btn btn-sm"
                  disabled={busy || probing != null}
                  data-testid="vlm-probe-btn"
                  onclick={() => onprobe(e)}
                  >{probing === e.name ? 'Probing…' : 'Probe'}</button
                >
                <button
                  type="button"
                  class="btn btn-sm"
                  disabled={busy}
                  data-testid="vlm-clone"
                  onclick={() => onclone(e)}>Clone</button
                >
                {#if e.source === 'stored'}
                  <button
                    type="button"
                    class="btn btn-sm"
                    disabled={busy}
                    data-testid="vlm-delete"
                    onclick={() => ondelete(e)}>Delete</button
                  >
                {/if}
              </div>
            </td>
          </tr>
          {#if probes[e.name] || probeErrors[e.name]}
            <tr data-testid="vlm-probe-row" data-name={e.name}>
              <td colspan="8" class="pb-2">
                <details open class="rounded border border-line p-2">
                  <summary class="cursor-pointer text-meta text-fg-2"
                    >Probe result for <code>{e.name}</code></summary
                  >
                  {#if probeErrors[e.name]}
                    <p
                      class="mt-1 text-meta text-danger-text"
                      data-testid="vlm-probe-error"
                    >
                      {probeErrors[e.name]}
                    </p>
                  {:else if probes[e.name]}
                    <div class="mt-1"><VlmProbePanel probe={probes[e.name]!} /></div>
                  {/if}
                </details>
              </td>
            </tr>
          {/if}
        {/each}
      </tbody>
    </table>
  </div>
</div>
