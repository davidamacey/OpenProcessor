<script lang="ts">
  /**
   * `/settings/ingest-policy` — what an ingest keeps and embeds
   * (OpenProcessor v0.4.0 `GET/PUT /ingest/policy`, cost preview
   * `POST /ingest/policy/preview`). The editor controller sends the draft
   * as typed and shows every served refusal; the policy changes only future
   * ingests.
   */
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import ActiveIngestProfileCard from '$lib/components/detector/ActiveIngestProfileCard.svelte';
  import IngestProfilePicker from '$lib/components/detector/IngestProfilePicker.svelte';
  import IngestPolicyForm from '$lib/components/detector/IngestPolicyForm.svelte';
  import IngestPolicyPreviewPanel from '$lib/components/detector/IngestPolicyPreviewPanel.svelte';
  import { IngestProfilePicker as ProfilePickerCtl } from '$lib/detector/ingestProfilePickerController.svelte';
  import { IngestPolicyEditor } from '$lib/detector/ingestPolicyController.svelte';
  import { projectHref } from '$lib/projectPaths';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const editor = new IngestPolicyEditor();
  const picker = ProfilePickerCtl.withDefaults(() => editor.reloadProfile());
  let confirming = $state(false);

  onMount(() => {
    const ctl = new AbortController();
    void editor.load(ctl.signal);
    void picker.load(ctl.signal);
    return () => {
      ctl.abort();
      editor.dispose();
    };
  });

  // One preview per quiet period after any edit (and once after the load).
  $effect(() => {
    if (editor.revision == null) return;
    void JSON.stringify(editor.draft);
    editor.schedulePreview();
  });

  async function doSave(): Promise<void> {
    const ok = await editor.save();
    confirming = false;
    if (ok) toastStore.success('Ingest policy saved');
  }
</script>

<div class="flex flex-col gap-4">
  <header class="flex flex-wrap items-baseline gap-3">
    <h1 class="text-h1 font-semibold text-fg-1">Ingest policy</h1>
    <span class="grow"></span>
    <a
      class="text-meta text-accent-text hover:underline"
      href={resolve(projectHref('/settings'))}>Back to settings</a
    >
  </header>
  <p class="text-body text-fg-2">
    Which detections an ingest keeps, and which of those get a vector. The policy changes
    only future ingests; embed detections already stored from the dashboard.
  </p>

  {#if editor.loadError}
    <p class="text-body text-danger-text" data-testid="policy-load-error">
      {editor.loadError}
    </p>
  {:else if editor.loading}
    <p class="text-body text-fg-3">Loading...</p>
  {:else}
    <ActiveIngestProfileCard
      profile={editor.activeProfile}
      error={editor.profileError}
      detector={editor.detector}
    />
    <IngestProfilePicker ctl={picker} />
    <IngestPolicyForm {editor} />
    <IngestPolicyPreviewPanel {editor} />

    {#if editor.conflict}
      <div
        class="space-y-2 rounded border border-warn-line bg-warn-soft p-3 text-body text-warn-text"
        data-testid="policy-conflict"
      >
        <p>The policy changed since you opened it.</p>
        {#each editor.saveLines as line, i (i)}
          <p class="text-meta">{line}</p>
        {/each}
        <div class="flex gap-2">
          <button
            type="button"
            class="btn btn-sm"
            data-testid="policy-reload"
            onclick={() => void editor.reload()}>Reload</button
          >
          <button
            type="button"
            class="btn btn-sm"
            data-testid="policy-keep"
            onclick={() => void editor.keepMyEdits()}>Keep my edits</button
          >
        </div>
      </div>
    {:else if editor.saveLines.length > 0}
      <div class="space-y-1 text-body text-danger-text" data-testid="policy-save-error">
        {#each editor.saveLines as line, i (i)}
          <p>{line}</p>
        {/each}
      </div>
    {/if}

    {#if editor.unknownNames.length > 0}
      <div
        class="rounded border border-warn-line bg-warn-soft p-3 text-meta text-warn-text"
        data-testid="policy-unknown-names"
      >
        <p>These names are not in the detector's labels or the registry; saved anyway:</p>
        <ul class="mt-1 list-disc pl-5">
          {#each editor.unknownNames as n, i (i)}
            <li class="font-mono">{n}</li>
          {/each}
        </ul>
      </div>
    {/if}

    <div class="flex items-center gap-3">
      <button
        type="button"
        class="btn btn-primary"
        data-testid="policy-save"
        disabled={!editor.dirty || editor.saving}
        onclick={() => (confirming = true)}>Save policy</button
      >
      <span class="text-meta text-fg-3">Revision {editor.revision}</span>
    </div>
  {/if}
</div>

{#if confirming}
  <ConfirmDialog
    title="Save the ingest policy?"
    confirmLabel="Save policy"
    busy={editor.saving}
    onconfirm={() => void doSave()}
    oncancel={() => (confirming = false)}
  >
    <p class="text-body text-fg-2">
      Future ingests will keep and embed detections under this policy. Detections already
      stored are not changed.
    </p>
  </ConfirmDialog>
{/if}
