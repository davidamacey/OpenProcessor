<!--
  /export: how much of the dataset a human (or an audited auto-validation)
  has confirmed. Only validated crops are exported, so a pool that is mostly
  machine labels exports a small dataset. Both counts are served
  (`GET /stats/dataset`); the warning tone is the plain fact that
  validated < total, not a threshold: the backend serves no total floor.
-->
<script lang="ts">
  import { formatCount } from '$lib/format/count';

  interface Props {
    validated: number;
    total: number;
  }
  let { validated, total }: Props = $props();

  const none = $derived(validated === 0);
  const partial = $derived(validated < total);
</script>

<section
  class="rounded-md border px-4 py-3 text-body {partial
    ? 'border-warn/40 bg-warn/10 text-warn-text'
    : 'border-success/40 bg-success/10 text-success-text'}"
  data-testid="validated-ratio"
  data-level={none ? 'none' : partial ? 'partial' : 'full'}
>
  <p>
    <span class="font-semibold tabular-nums" data-testid="validated-ratio-counts"
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
