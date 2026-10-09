<!--
  /export: how much of the dataset a human (or an audited auto-validation)
  has confirmed. Only validated crops are exported, so a pool that is mostly
  machine labels exports a small dataset. Both counts are served
  (`GET /stats/dataset`); the warning tone is the plain fact that
  validated < total, not a threshold: the backend serves no total floor.
-->
<script lang="ts">
  import { formatCount } from '$lib/formatCount';

  interface Props {
    validated: number;
    total: number;
  }
  let { validated, total }: Props = $props();

  const none = $derived(validated === 0);
  const partial = $derived(validated < total);
</script>

<section
  class="rounded-md border px-4 py-3 text-sm {partial
    ? 'border-amber-500/40 bg-amber-500/10 text-amber-100'
    : 'border-green-500/40 bg-green-500/10 text-green-100'}"
  data-testid="validated-ratio"
  data-level={none ? 'none' : partial ? 'partial' : 'full'}
>
  <p>
    <span class="font-mono font-semibold" data-testid="validated-ratio-counts"
      >{formatCount(validated)} of {formatCount(total)}</span
    >
    crops are validated.
    {#if none}
      Nothing is exportable yet: VLM and detector labels are suggestions until a human
      validates them.
    {:else if partial}
      Only validated crops are exported; the rest are machine labels (VLM or detector
      suggestions) that no human has confirmed.
    {:else}
      Every crop is validated.
    {/if}
  </p>
</section>
