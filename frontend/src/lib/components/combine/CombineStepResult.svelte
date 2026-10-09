<script lang="ts">
  /**
   * The served response of the last confirmed next step. Rendered
   * generically: a `status` string becomes the chip, every other scalar is
   * a key/value row, nested or long values sit in a collapsed disclosure.
   */
  import { combineLabel, reportCell } from '$lib/combine/combineText';

  interface Props {
    action: string;
    result: unknown;
  }
  let { action, result }: Props = $props();

  const LONG = 80;
  const obj = $derived(
    result && typeof result === 'object' && !Array.isArray(result)
      ? (result as Record<string, unknown>)
      : null,
  );
  const status = $derived(typeof obj?.status === 'string' ? obj.status : null);
  const entries = $derived(
    Object.entries(obj ?? {}).filter(([k]) => !(k === 'status' && status !== null)),
  );
  const isCollapsed = (v: unknown): boolean =>
    (v !== null && typeof v === 'object') || (typeof v === 'string' && v.length > LONG);
  const inline = $derived(entries.filter(([, v]) => !isCollapsed(v)));
  const collapsed = $derived(entries.filter(([, v]) => isCollapsed(v)));
</script>

<section
  class="space-y-1 rounded border border-zinc-800 p-3"
  data-testid="combine-step-result"
>
  <div class="flex flex-wrap items-center gap-2">
    <h2 class="text-sm font-semibold text-zinc-200">Last next step</h2>
    <span class="text-xs text-zinc-300" data-testid="combine-step-result-action"
      >{combineLabel(action)}</span
    >
    <span
      class="rounded bg-zinc-800 px-1.5 py-0.5 text-xs text-zinc-200"
      data-testid="combine-step-result-status"
      >{status ? combineLabel(status) : 'Done'}</span
    >
  </div>
  {#if inline.length > 0}
    <dl class="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-0.5 text-xs">
      {#each inline as [k, v] (k)}
        <dt class="text-zinc-500">{combineLabel(k)}</dt>
        <dd class="font-mono text-zinc-200" data-result-key={k}>{reportCell(v)}</dd>
      {/each}
    </dl>
  {/if}
  {#each collapsed as [k, v] (k)}
    <details class="text-xs" data-testid="combine-step-result-detail">
      <summary class="cursor-pointer text-zinc-400">{combineLabel(k)}</summary>
      <pre class="mt-1 overflow-x-auto font-mono text-zinc-300">{typeof v === 'string'
          ? v
          : JSON.stringify(v, null, 2)}</pre>
    </details>
  {/each}
  {#if obj === null}
    <p class="text-xs text-zinc-400">{reportCell(result)}</p>
  {/if}
</section>
