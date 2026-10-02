<script lang="ts">
  /**
   * A served Reprocess response's per-scope counts and its served summary
   * `message` (W10.13). Shared by the Reprocess dialog and the region-
   * profile impact panel's Re-run; nothing here words an outcome.
   */
  import type { ReprocessResponse } from '$lib/types_import';

  interface Props {
    res: ReprocessResponse;
    /** The served label of a scope id (raw id when none). */
    scopeLabel: (id: string) => string;
  }

  let { res, scopeLabel }: Props = $props();
</script>

<table class="w-full text-left text-xs" data-testid="reprocess-counts">
  <thead class="text-zinc-500">
    <tr>
      <th class="py-0.5 pr-3">Scope</th>
      <th class="py-0.5 pr-3 text-right">Selected</th>
      <th class="py-0.5 pr-3 text-right">Locked, skipped</th>
      <th class="py-0.5 text-right">Queued</th>
    </tr>
  </thead>
  <tbody class="font-mono">
    {#each res.scopes as s (s.scope)}
      <tr class="border-t border-zinc-800">
        <td class="py-0.5 pr-3 font-sans">{scopeLabel(s.scope)}</td>
        <td class="py-0.5 pr-3 text-right">{s.selected.toLocaleString()}</td>
        <td class="py-0.5 pr-3 text-right">{s.locked_skipped.toLocaleString()}</td>
        <td class="py-0.5 text-right">{s.queued.toLocaleString()}</td>
      </tr>
    {/each}
  </tbody>
</table>
{#if res.message}
  <p class="text-xs text-zinc-300" data-testid="reprocess-message">{res.message}</p>
{/if}
