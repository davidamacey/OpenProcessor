<script lang="ts">
  /**
   * `/projects/combine` — the combine-projects wizard (OpenProcessor P4;
   * projects_plan.md §8). Global, like `/projects`: the target project does
   * not exist yet. Gated by `combineAvailability` (a one-shot probe): when
   * the backend has no combine router the page says so and fires nothing
   * else. State lives in `combineWizardController`; every count, error and
   * verdict on screen is the served preview's.
   */
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import CombineIssueList from '$components/combine/CombineIssueList.svelte';
  import CombineMappingTable from '$components/combine/CombineMappingTable.svelte';
  import CombineOptions from '$components/combine/CombineOptions.svelte';
  import CombinePreviewSummary from '$components/combine/CombinePreviewSummary.svelte';
  import CombineSourcesStep from '$components/combine/CombineSourcesStep.svelte';
  import { combineAvailability } from '$lib/combine/combineAvailability.svelte';
  import { createCombineWizard } from '$lib/combine/combineWizardController.svelte';
  import { formatBytes } from '$lib/combine/combineText';
  import { formatCount } from '$lib/formatCount';
  import { projectsStore } from '$stores/projects.svelte';
  import { toastStore } from '$stores/toast.svelte';

  const appName =
    (import.meta.env?.PUBLIC_APP_NAME as string | undefined) || 'Cropwright';

  const wizard = createCombineWizard({
    projectOf: (slug) => projectsStore.list.find((p) => p.slug === slug) ?? null,
  });

  let confirmOpen = $state(false);

  onMount(() => {
    void combineAvailability.init();
    return () => wizard.destroy();
  });

  const candidates = $derived(
    projectsStore.list.filter((p) => p.selectable && p.status === 'active'),
  );
  const preview = $derived(wizard.preview);

  async function confirmStart(): Promise<void> {
    const started = await wizard.start();
    confirmOpen = false;
    if (!started) return;
    await projectsStore.refresh();
    toastStore.info(`Combining into "${started.target}".`);
    await goto(
      resolve('/projects/combine/[job_id]', {
        job_id: encodeURIComponent(started.job_id),
      }),
    );
  }
</script>

<div class="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
  <header class="flex h-12 shrink-0 items-center gap-4 border-b border-zinc-800 px-4">
    <span class="text-sm font-semibold tracking-tight">{appName}</span>
    <span class="text-zinc-600">/</span>
    <a href={resolve('/projects')} class="text-sm text-zinc-300 hover:text-white"
      >Projects</a
    >
    <span class="text-zinc-600">/</span>
    <h1 class="text-sm text-zinc-200">Combine projects</h1>
  </header>

  <main class="mx-auto w-full max-w-5xl flex-1 space-y-6 p-4">
    {#if combineAvailability.available === true}
      <p class="text-xs text-zinc-400">
        Merge several projects into a new one. The sources are left untouched; the target
        is created and filled by a background job.
      </p>

      <CombineSourcesStep
        {wizard}
        {candidates}
        slugPattern={projectsStore.limits?.slug_pattern ?? null}
      />

      {#if wizard.previewError}
        <p class="text-sm text-red-300" data-testid="combine-preview-error">
          {wizard.previewError}
        </p>
      {/if}

      {#if preview}
        <section class="space-y-3" data-testid="combine-step-mapping">
          <div class="flex items-center gap-3">
            <h2 class="text-sm font-semibold text-zinc-200">2. Class mapping</h2>
            <span class="grow"></span>
            <button
              type="button"
              class="btn btn-sm"
              data-testid="combine-reset-suggestions"
              onclick={() => wizard.resetToSuggestions()}>Reset to suggestions</button
            >
          </div>
          <p class="text-xs text-zinc-500">
            Untouched rows follow the server's suggestions. A class is its name: a
            <em>create</em> row defines a target class and a <em>map</em> row points at one.
          </p>
          {#each preview.sources as s (s.project)}
            <CombineMappingTable {wizard} source={s} />
          {/each}
        </section>
      {/if}

      <CombineOptions {wizard} />

      {#if wizard.previewing && !preview}
        <p class="text-xs text-zinc-500" data-testid="combine-previewing">Previewing…</p>
      {/if}
      {#if preview}
        <CombinePreviewSummary {preview} stale={wizard.stale || wizard.previewing} />
      {/if}

      <section class="space-y-2" data-testid="combine-step-start">
        {#if wizard.refusal}
          <div
            class="space-y-1 rounded border border-red-900 bg-red-950/30 p-2 text-xs text-red-200"
            data-testid="combine-start-refusal"
            data-code={wizard.refusal.code ?? ''}
          >
            <p>{wizard.refusal.message}</p>
            {#if wizard.refusal.report}
              <CombineIssueList
                testid="combine-refusal-report"
                issues={[
                  ...wizard.refusal.report.errors,
                  ...wizard.refusal.report.warnings,
                ].map((i) => ({
                  code: i.code,
                  severity: i.severity === 'warning' ? 'warning' : 'error',
                  project: i.field,
                  message: i.message,
                  detail: i.detail,
                }))}
              />
            {/if}
            {#if wizard.refusal.jobs.length > 0}
              <ul class="list-inside list-disc">
                {#each wizard.refusal.jobs as j (j.id)}
                  <li>{j.kind_label}: {j.label}</li>
                {/each}
              </ul>
            {/if}
          </div>
        {/if}
        <button
          type="button"
          class="btn btn-primary"
          data-testid="combine-start"
          disabled={!wizard.canStart}
          onclick={() => (confirmOpen = true)}>Start combine…</button
        >
      </section>
    {:else if combineAvailability.available === false}
      <p class="text-sm text-zinc-400" data-testid="combine-unavailable">
        Combining projects isn't available on this backend.
      </p>
    {:else if combineAvailability.error}
      <div class="space-y-2 text-sm">
        <p class="text-red-300">
          Could not check combine support: {combineAvailability.error}
        </p>
        <button
          type="button"
          class="btn btn-sm"
          onclick={() => void combineAvailability.retry()}>Retry</button
        >
      </div>
    {:else}
      <p class="text-sm text-zinc-500">Loading…</p>
    {/if}
  </main>
</div>

{#if confirmOpen && preview}
  <ConfirmDialog
    title="Start combine?"
    confirmLabel="Start combine"
    busy={wizard.starting}
    onconfirm={() => void confirmStart()}
    oncancel={() => (confirmOpen = false)}
  >
    <p data-testid="combine-confirm-body">
      Creates project <span class="font-mono text-zinc-100">{preview.target.slug}</span>
      from {wizard.sources.length} source{wizard.sources.length === 1 ? '' : 's'}:
      {formatCount(preview.target.images)} images, {formatCount(preview.target.items)} items,
      {formatCount(preview.target.holdout_images)} test images,
      {(preview.target.classes ?? []).length} classes.
    </p>
    <p class="text-xs text-zinc-400">
      Images to link: {formatBytes(preview.bytes?.to_link)} · to copy: {formatBytes(
        preview.bytes?.to_copy,
      )}. The sources are not changed. You can undo it later by deleting the target.
    </p>
  </ConfirmDialog>
{/if}
