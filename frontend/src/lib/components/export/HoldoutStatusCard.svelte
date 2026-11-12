<script lang="ts">
  /** /export's test-holdout status card: frozen totals and the
   *  server-flagged deficient classes, or the Freeze action. */
  import type { ExportRow } from '$lib/export/exportDatasetRows';
  import type { TestHoldoutStats } from '$lib/types_export';

  interface Props {
    holdout: TestHoldoutStats | null;
    rows: ExportRow[];
    onFreeze: () => void;
  }

  let { holdout, rows, onFreeze }: Props = $props();

  const totalTestCrops = $derived(holdout?.total ?? 0);
  const testFrozen = $derived(totalTestCrops > 0);
  // Server-flagged deficient classes (`{API_PREFIX}/test_holdout/stats`'s
  // per-bucket `deficient`, falling back to the served `min_test_per_class`
  // when a bucket omits the flag) — never a hardcoded "< 5".
  const deficientClassCount = $derived(rows.filter((r) => r.testDeficient).length);
</script>

<!-- Test holdout status card -->
<section class="surface p-4">
  <header class="mb-2 flex flex-wrap items-center gap-3">
    <h2 class="text-sm font-semibold text-fg-2">Test holdout</h2>
    <span class="text-xs text-fg-3">
      red badge: a class with frozen test crops that the server flags deficient (below {holdout?.min_test_per_class ??
        '…'}). Classes with no validated crops have no test crops and are not flagged.
    </span>
    <span class="grow"></span>
    {#if !testFrozen}
      <button class="btn btn-primary" type="button" onclick={onFreeze}>
        Freeze test set
      </button>
    {:else}
      <span
        class="rounded-md border border-success/40 bg-success/10 px-2 py-1 text-xs text-success"
      >
        frozen — {totalTestCrops.toLocaleString()} crops
      </span>
    {/if}
  </header>
  {#if testFrozen}
    <div class="flex flex-wrap items-center gap-3 text-xs">
      <span class="text-fg-2">
        {totalTestCrops.toLocaleString()} test crops across
        {(holdout?.by_class?.length ?? 0).toString()} classes.
      </span>
      {#if deficientClassCount > 0}
        <span
          class="rounded-md border border-danger/40 bg-danger/10 px-2 py-0.5 text-danger"
        >
          {deficientClassCount} class{deficientClassCount === 1 ? '' : 'es'} below {holdout?.min_test_per_class ??
            '…'} test crops
        </span>
      {/if}
    </div>
  {:else}
    <p class="text-xs text-fg-3">
      Test set is not yet frozen. Freezing is one-shot per dataset version.
    </p>
  {/if}
</section>
