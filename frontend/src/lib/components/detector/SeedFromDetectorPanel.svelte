<!--
  "Create classes from the detector" on `/classes`: choose detector labels
  (none = all), Preview (a dry run), read the served created / skipped /
  conflicts lists, then Create behind a confirm. Absent when the deployment
  reports no detector.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import { humanizeId } from '$lib/humanizeId';
  import { SeedFromDetector } from '$lib/detector/seedController.svelte';
  import { toastStore } from '$stores/toast.svelte';

  let { controller = new SeedFromDetector() }: { controller?: SeedFromDetector } =
    $props();

  let confirming = $state(false);

  onMount(() => {
    const ctl = new AbortController();
    void controller.load(ctl.signal);
    return () => ctl.abort();
  });

  function toggle(name: string, on: boolean): void {
    controller.setChosen(
      on ? [...controller.chosen, name] : controller.chosen.filter((n) => n !== name),
    );
  }

  async function doCreate(): Promise<void> {
    const ok = await controller.create();
    confirming = false;
    if (ok && controller.created) {
      toastStore.success(
        `Created ${controller.created.created.length} classes from ${controller.created.detector_model}`,
      );
    }
  }
</script>

{#if controller.available}
  <details class="surface mt-4 p-4" data-testid="seed-panel">
    <summary class="cursor-pointer text-sm font-semibold text-zinc-200">
      Create classes from the detector
    </summary>
    <div class="mt-3 space-y-3 text-xs">
      <p class="text-zinc-400">
        Choose detector labels to turn into classes. With none chosen, every label is
        used.
      </p>
      <div class="flex max-h-40 flex-wrap gap-x-3 gap-y-1 overflow-y-auto">
        {#each controller.labelNames as name, i (i)}
          <label class="flex items-center gap-1">
            <input
              type="checkbox"
              checked={controller.chosen.includes(name)}
              onchange={(e) => toggle(name, e.currentTarget.checked)}
            />
            <span class="text-zinc-200">{name}</span>
          </label>
        {/each}
      </div>
      <div class="flex items-center gap-2">
        <button
          type="button"
          class="btn btn-sm"
          data-testid="seed-preview"
          disabled={controller.busy}
          onclick={() => void controller.preview()}>Preview</button
        >
        {#if controller.result && controller.result.created.length > 0}
          <button
            type="button"
            class="btn btn-sm btn-primary"
            data-testid="seed-create"
            disabled={controller.busy}
            onclick={() => (confirming = true)}
            >Create {controller.result.created.length} classes</button
          >
        {/if}
      </div>

      {#if controller.errorLines.length > 0}
        <div class="space-y-1 text-red-300" data-testid="seed-error">
          {#each controller.errorLines as line, i (i)}
            <p>{line}</p>
          {/each}
        </div>
      {/if}

      {#if controller.result}
        {@const r = controller.result}
        <div class="space-y-2" data-testid="seed-result">
          <p class="text-zinc-400">Detector {r.detector_model}</p>
          {#if r.created.length > 0}
            <ul class="list-disc pl-5" data-testid="seed-created">
              {#each r.created as c (c.detector_label)}
                <li class="text-zinc-200">
                  would create <span class="font-mono">{c.name}</span> (from {c.detector_label})
                </li>
              {/each}
            </ul>
          {:else}
            <p class="text-zinc-500">Nothing to create.</p>
          {/if}
          {#if r.skipped.length > 0}
            <ul class="list-disc pl-5" data-testid="seed-skipped">
              {#each r.skipped as s (s.detector_label)}
                <li class="text-zinc-400">
                  skipped <span class="font-mono">{s.name}</span>: {humanizeId(s.reason)}
                </li>
              {/each}
            </ul>
          {/if}
          {#if r.conflicts.length > 0}
            <ul class="list-disc pl-5" data-testid="seed-conflicts">
              {#each r.conflicts as c (c.class_id_in_detector)}
                <li class="text-amber-200">
                  conflict on <span class="font-mono">{c.detector_label}</span> (detector
                  class
                  {c.class_id_in_detector}): {humanizeId(c.reason)}
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      {/if}
    </div>
  </details>
{/if}

{#if confirming && controller.result}
  <ConfirmDialog
    title="Create classes from the detector?"
    confirmLabel="Create {controller.result.created.length} classes"
    busy={controller.busy}
    onconfirm={() => void doCreate()}
    oncancel={() => (confirming = false)}
  >
    <p class="text-sm text-zinc-300">
      {controller.result.created.length} new classes will be added to the registry, one per
      detector label listed in the preview.
    </p>
  </ConfirmDialog>
{/if}
