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
  import { humanizeId } from '$lib/humanizeId';
  import AuditClassTable from '$lib/components/audit/AuditClassTable.svelte';
  import AuditQueueList from '$lib/components/audit/AuditQueueList.svelte';
  import AuditStartForm from '$lib/components/audit/AuditStartForm.svelte';
  import ConfusionMatrix from '$lib/components/audit/ConfusionMatrix.svelte';

  const audit = new AuditController();

  onMount(() => {
    const ctl = new AbortController();
    void audit.load(ctl.signal);
    return () => ctl.abort();
  });
</script>

<div class="mx-auto flex h-full max-w-7xl flex-col gap-4 p-6">
  <header class="flex flex-wrap items-center gap-3">
    <h1 class="text-2xl font-semibold tracking-tight">Accuracy audit</h1>
    <span class="grow"></span>
    <button
      type="button"
      class="btn"
      disabled={audit.loading}
      onclick={() => void audit.load()}>Refresh</button
    >
  </header>

  <AuditStartForm {audit} />

  {#if audit.loading && !audit.report}
    <section class="surface p-6 text-sm text-zinc-500">Loading…</section>
  {:else if audit.loadError}
    <section class="surface flex flex-col gap-3 p-6 text-sm">
      <p class="text-red-300" data-testid="audit-load-error">{audit.loadError}</p>
      <button type="button" class="btn w-fit" onclick={() => void audit.load()}
        >Retry</button
      >
    </section>
  {:else if audit.report}
    {@const r = audit.report}
    <section
      class="surface flex flex-wrap items-center gap-x-6 gap-y-2 p-4 text-sm"
      data-testid="audit-summary"
    >
      <span
        ><span class="font-mono text-lg" data-testid="audit-audited">{r.audited}</span>
        <span class="text-zinc-400">labeled by a human</span></span
      >
      <span
        ><span class="font-mono text-lg" data-testid="audit-pending">{r.pending}</span>
        <span class="text-zinc-400">still waiting</span></span
      >
      <ul class="flex flex-wrap gap-2 text-xs" data-testid="audit-outcomes">
        {#each Object.entries(r.outcomes) as [outcome, n] (outcome)}
          <li class="rounded border border-zinc-700 px-1.5 py-0.5 text-zinc-300">
            {humanizeId(outcome)}: <span class="font-mono">{n}</span>
          </li>
        {/each}
      </ul>
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

    {#if audit.queue}
      <AuditQueueList queue={audit.queue} ongoto={(p) => void audit.goToPage(p)} />
    {/if}
  {/if}
</div>
