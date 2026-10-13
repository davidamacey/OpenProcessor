<script lang="ts">
  /**
   * Create or edit an ingest source: an S3 bucket/prefix or a server folder, its
   * options and its triggers (watch, schedule, webhook). The location of an
   * existing source is fixed. Validation (allow-list, cron, minimum interval) is the
   * backend's; its refusal is shown as served.
   */
  import type { IngestSources } from '$lib/ingest/sourcesController.svelte';

  interface Props {
    controller: IngestSources;
  }
  let { controller }: Props = $props();

  const list = $derived(controller.list);
  const creating = $derived(controller.editing === 'new');
</script>

<form
  class="space-y-3 rounded border border-zinc-700 p-3"
  data-testid="source-form"
  onsubmit={(e) => {
    e.preventDefault();
    void controller.save();
  }}
>
  <div class="flex flex-wrap items-end gap-3">
    <label class="text-xs text-zinc-400">
      Name
      <input
        class="input input-sm block"
        data-testid="source-name"
        bind:value={controller.form.name}
        placeholder="Incoming photos"
      />
    </label>
    {#if creating}
      <fieldset class="flex gap-3 text-xs text-zinc-300">
        <label class="flex items-center gap-1">
          <input
            type="radio"
            name="source-backend"
            value="s3"
            bind:group={controller.form.backend}
            disabled={!list?.s3_configured}
            data-testid="source-backend-s3"
          /> S3 bucket
        </label>
        <label class="flex items-center gap-1">
          <input
            type="radio"
            name="source-backend"
            value="folder"
            bind:group={controller.form.backend}
            data-testid="source-backend-folder"
          /> Server folder
        </label>
      </fieldset>
    {/if}
  </div>

  {#if creating && list && !list.s3_configured}
    <p class="text-xs text-amber-300" data-testid="s3-off">
      S3 is off on this deployment: the API's OP_S3_ALLOWED lists no bucket.
    </p>
  {/if}

  {#if creating}
    {#if controller.form.backend === 's3'}
      <div class="flex flex-wrap items-end gap-3">
        <label class="text-xs text-zinc-400">
          Bucket
          <input
            class="input input-sm block font-mono"
            data-testid="source-bucket"
            bind:value={controller.form.bucket}
          />
        </label>
        <label class="text-xs text-zinc-400">
          Prefix
          <input
            class="input input-sm block font-mono"
            data-testid="source-prefix"
            placeholder="incoming/"
            bind:value={controller.form.prefix}
          />
        </label>
      </div>
      {#if list && list.s3_allowed.length > 0}
        <p class="text-xs text-zinc-500">
          Allowed by this deployment:
          <span class="font-mono">{list.s3_allowed.join(', ')}</span>
        </p>
      {/if}
    {:else}
      <label class="block text-xs text-zinc-400">
        Folder (a path inside the API container, under a source root)
        <input
          class="input block w-full font-mono text-xs"
          data-testid="source-folder"
          bind:value={controller.form.folder}
        />
      </label>
    {/if}
  {/if}

  <fieldset class="space-y-2 text-xs text-zinc-300">
    <legend class="text-zinc-400">Triggers</legend>
    <div class="flex flex-wrap items-center gap-2">
      <label class="flex items-center gap-1">
        <input
          type="checkbox"
          bind:checked={controller.form.watchEnabled}
          data-testid="trigger-watch"
        />
        Watch every
      </label>
      <input
        class="input input-sm w-20"
        type="number"
        min={list?.min_watch_interval_s ?? 10}
        placeholder={String(list?.default_watch_interval_s ?? 60)}
        bind:value={controller.form.watchInterval}
        disabled={!controller.form.watchEnabled}
        data-testid="trigger-watch-interval"
      />
      <span class="text-zinc-500">seconds</span>
    </div>
    <div class="flex flex-wrap items-center gap-2">
      <label class="flex items-center gap-1">
        <input
          type="checkbox"
          bind:checked={controller.form.scheduleEnabled}
          data-testid="trigger-schedule"
        />
        On a schedule (cron, UTC)
      </label>
      <input
        class="input input-sm w-44 font-mono"
        placeholder="0 2 * * *"
        bind:value={controller.form.cron}
        disabled={!controller.form.scheduleEnabled}
        data-testid="trigger-cron"
      />
    </div>
    {#if controller.form.backend === 's3'}
      <label class="flex items-center gap-1">
        <input
          type="checkbox"
          bind:checked={controller.form.webhook}
          data-testid="trigger-webhook"
        />
        Accept S3 event notifications (webhook)
      </label>
    {/if}
  </fieldset>

  <details class="text-xs text-zinc-400">
    <summary class="cursor-pointer">Options</summary>
    <div class="mt-2 flex flex-wrap items-end gap-3">
      <label>
        Source tag
        <input class="input input-sm block" bind:value={controller.form.source} />
      </label>
      <label>
        File extensions
        <input
          class="input input-sm block"
          placeholder=".jpg,.jpeg,.png"
          bind:value={controller.form.extensions}
        />
      </label>
      <label>
        Images per batch
        <input
          class="input input-sm block w-24"
          type="number"
          min="1"
          placeholder={String(list?.default_batch_size ?? 16)}
          bind:value={controller.form.batchSize}
        />
      </label>
      <label>
        Batches in flight
        <input
          class="input input-sm block w-24"
          type="number"
          min="1"
          placeholder={String(list?.default_concurrency ?? 16)}
          bind:value={controller.form.concurrency}
        />
      </label>
    </div>
  </details>

  {#if controller.actionError}
    <p class="text-xs text-red-300" data-testid="source-form-error">
      {controller.actionError}
    </p>
  {/if}
  <div class="flex gap-2">
    <button
      class="btn btn-primary btn-sm"
      type="submit"
      data-testid="source-save"
      disabled={!controller.canSave}
    >
      {creating ? 'Add source' : 'Save changes'}
    </button>
    <button class="btn btn-sm" type="button" onclick={() => controller.closeForm()}>
      Cancel
    </button>
  </div>
</form>
