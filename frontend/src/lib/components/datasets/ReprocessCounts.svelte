<script lang="ts">
  /**
   * A served Reprocess response's per-scope counts (W10.13). Shared by the
   * Reprocess dialog and the region-profile impact panel's Re-run; nothing
   * here words an outcome.
   */
  import { humanizeId } from '$lib/humanizeId';
  import type { ReprocessResponse } from '$lib/types_import';

  interface Props {
    res: Pick<ReprocessResponse, 'scopes'>;
    /** The served label of a scope id (raw id when none). */
    scopeLabel: (id: string) => string;
  }

  let { res, scopeLabel }: Props = $props();

  /** A count the server omitted reads "—", never a false 0. */
  const count = (n: number | undefined): string => (n == null ? '—' : n.toLocaleString());

  /** A served detail value verbatim; booleans as yes/no. */
  const detailValue = (v: number | boolean | string): string =>
    typeof v === 'boolean' ? (v ? 'yes' : 'no') : String(v);
</script>

<table class="w-full text-left text-xs" data-testid="reprocess-counts">
  <thead class="text-zinc-500">
    <tr>
      <th class="py-0.5 pr-3">Scope</th>
      <th class="py-0.5 pr-3 text-right">Selected</th>
      <th class="py-0.5 pr-3 text-right">Locked, skipped</th>
      <th class="py-0.5 pr-3 text-right">Queued</th>
      <th class="py-0.5 pr-3 text-right">Failed</th>
      <th class="py-0.5 text-right">Not found</th>
    </tr>
  </thead>
  <tbody class="font-mono">
    {#each res.scopes as s (s.scope)}
      <tr class="border-t border-zinc-800">
        <td class="py-0.5 pr-3 font-sans">{scopeLabel(s.scope)}</td>
        <td class="py-0.5 pr-3 text-right">{count(s.selected)}</td>
        <td class="py-0.5 pr-3 text-right">{count(s.locked_skipped)}</td>
        <td class="py-0.5 pr-3 text-right">{count(s.queued)}</td>
        <td class="py-0.5 pr-3 text-right">{count(s.failed)}</td>
        <td class="py-0.5 text-right">{count(s.not_found)}</td>
      </tr>
      {#if s.detail && Object.keys(s.detail).length > 0}
        <tr data-testid="reprocess-detail">
          <td class="py-0.5 pr-3 pl-3 font-sans" colspan="6">
            <dl class="grid grid-cols-[auto_1fr] gap-x-3 text-zinc-400">
              {#each Object.entries(s.detail) as [k, v] (k)}
                <dt>{humanizeId(k)}</dt>
                <dd class="font-mono text-zinc-200">{detailValue(v)}</dd>
              {/each}
            </dl>
          </td>
        </tr>
      {/if}
      {#each s.breakdown ?? [] as b (b.detector + ':' + b.reason)}
        <tr class="text-zinc-500" data-testid="reprocess-breakdown">
          <td class="py-0.5 pr-3 pl-3 font-sans" colspan="5">
            {b.detector} · {b.reason}
          </td>
          <td class="py-0.5 text-right">{b.count.toLocaleString()}</td>
        </tr>
      {/each}
    {/each}
  </tbody>
</table>
