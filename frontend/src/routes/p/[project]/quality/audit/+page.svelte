<script lang="ts">
  /**
   * Accuracy audit (#119): how often is the detector, and the VLM, right?
   * Draw a stratified sample of machine-labelled crops, label them (a human
   * label is ground truth), then read the served per-class precision, the
   * confusion matrix and the outcome counts. Every figure is served; the
   * `insufficient_sample` flag is the server's verdict on a class.
   */
  import { onMount } from 'svelte';
  import { AuditController } from '$lib/labelConfirmation/auditController.svelte';
  import AuditClassTable from '$lib/components/audit/AuditClassTable.svelte';
  import EmptyState from '$components/ui/EmptyState.svelte';
  import AuditOutcomes from '$lib/components/audit/AuditOutcomes.svelte';
  import AuditPendingStrip from '$lib/components/audit/AuditPendingStrip.svelte';
  import AuditStartForm from '$lib/components/audit/AuditStartForm.svelte';
  import ConfusionMatrix from '$lib/components/audit/ConfusionMatrix.svelte';

  const audit = new AuditController();
  // Nothing audited and nothing waiting: one empty state carries the way in.
  const blank = $derived(
    !audit.loadError &&
      audit.report != null &&
      audit.report.audited === 0 &&
      audit.report.pending === 0,
  );

  onMount(() => {
    const ctl = new AbortController();
    void audit.load(ctl.signal);
    return () => ctl.abort();
  });
</script>

<div class="flex flex-col gap-4">
  <header class="flex flex-wrap items-center gap-3">
    <h2 class="text-title font-semibold text-fg-1">Accuracy audit</h2>
    <span class="grow"></span>
    <button
      type="button"
      class="btn"
      disabled={audit.loading}
      onclick={() => void audit.load()}>Refresh</button
    >
  </header>

  {#if !blank}
    <AuditStartForm {audit} />
  {/if}

  {#if audit.loading && !audit.report}
    <section class="surface p-6 text-body text-fg-3">Loading…</section>
  {:else if audit.loadError}
    <section class="surface flex flex-col gap-3 p-6 text-body">
      <p class="text-danger-text" data-testid="audit-load-error">{audit.loadError}</p>
      <button type="button" class="btn w-fit" onclick={() => void audit.load()}
        >Retry</button
      >
    </section>
  {:else if blank}
    <div data-testid="audit-empty">
      <EmptyState
        title="No accuracy audit yet"
        description="Draw a sample of machine-labelled crops, then label them: a human label is the ground truth the detector and the VLM are measured against."
      >
        {#snippet action()}
          <AuditStartForm {audit} />
        {/snippet}
      </EmptyState>
    </div>
  {:else if audit.report}
    {@const r = audit.report}
    <section
      class="surface flex flex-wrap items-center gap-x-6 gap-y-2 p-4 text-body"
      data-testid="audit-summary"
    >
      <span
        ><span class="text-title tabular-nums" data-testid="audit-audited"
          >{r.audited}</span
        >
        <span class="text-fg-2">labeled by a human</span></span
      >
      <span
        ><span class="text-title tabular-nums" data-testid="audit-pending"
          >{r.pending}</span
        >
        <span class="text-fg-2">still waiting</span></span
      >
      <AuditOutcomes outcomes={r.outcomes} />
    </section>

    <div class="grid grid-cols-1 gap-4 xl:grid-cols-2">
      <AuditClassTable
        title="Detector precision"
        blurb="How often the human agreed with the detector's class."
        stats={r.detector}
        minPerClass={r.min_per_class}
        testId="audit-detector"
      />
      <AuditClassTable
        title="VLM precision"
        blurb="How often the human agreed with the VLM, over crops whose label came from the VLM."
        stats={r.vlm}
        minPerClass={r.min_per_class}
        testId="audit-vlm"
      />
    </div>

    <ConfusionMatrix confusion={r.confusion} />

    {#if r.pending > 0}
      <AuditPendingStrip audited={r.audited} pending={r.pending} />
    {:else}
      <section class="surface p-4 text-body">
        <span class="text-fg-3" data-testid="audit-nothing-waiting"
          >Nothing is waiting. Draw a sample to start an audit.</span
        >
      </section>
    {/if}
  {/if}
</div>
