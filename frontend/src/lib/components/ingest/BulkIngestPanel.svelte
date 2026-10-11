<script lang="ts">
  /**
   * Bulk ingest of a server folder or manifest as a background run: start it,
   * watch live progress and rates, cancel it, resume it. The run lives in the
   * backend (the same runner the command line drives); this panel renders the
   * served run and sends start/cancel. All state is in `bulkIngestController`.
   */
  import { onDestroy, onMount } from 'svelte';
  import {
    BulkIngest,
    FAILURES_PAGE_SIZE,
  } from '$lib/ingest/bulkIngestController.svelte';
  import {
    formatDuration,
    headlineRate,
    progressPercent,
    stageRows,
  } from '$lib/ingest/bulkIngestFormat';
  import type { BulkIngestStatus } from '$lib/types_bulkIngest';

  interface Props {
    /** Injected in tests. */
    controller?: BulkIngest;
  }
  let { controller = new BulkIngest() }: Props = $props();

  onMount(() => void controller.init());
  onDestroy(() => controller.stop());

  const STATUS_TONE: Record<BulkIngestStatus, string> = {
    queued: 'text-zinc-300',
    running: 'text-blue-300',
    completed: 'text-emerald-300',
    completed_with_errors: 'text-amber-300',
    failed: 'text-red-300',
    cancelled: 'text-amber-300',
    interrupted: 'text-amber-300',
  };

  const run = $derived(controller.run);
  const progress = $derived(run?.progress ?? null);
  const config = $derived(controller.config);
  const pathLabel = $derived(
    controller.form.kind === 'folder'
      ? 'Folder (a path inside the API container)'
      : 'Manifest file (one image path per line)',
  );
</script>

<div class="space-y-4" data-testid="bulk-ingest">
  {#if controller.configError}
    <p class="text-xs text-red-300">
      Could not load bulk ingest: {controller.configError}
    </p>
  {:else if !config}
    <p class="text-xs text-zinc-500">Loading…</p>
  {:else}
    <p class="text-xs text-zinc-400">
      Ingest a whole folder or a list of images that already lie under a source root on
      the server. The run continues if you leave this page.
    </p>
    <ul class="text-xs text-zinc-300" data-testid="bulk-roots">
      {#each config.source_roots as root (root)}
        <li class="font-mono break-all">{root}</li>
      {/each}
    </ul>

    <form
      class="space-y-3"
      onsubmit={(e) => {
        e.preventDefault();
        void controller.start();
      }}
    >
      <div class="flex flex-wrap gap-4 text-xs text-zinc-300">
        <label class="flex items-center gap-1">
          <input
            type="radio"
            name="bulk-kind"
            value="folder"
            bind:group={controller.form.kind}
            disabled={controller.live}
          /> Folder
        </label>
        <label class="flex items-center gap-1">
          <input
            type="radio"
            name="bulk-kind"
            value="manifest"
            bind:group={controller.form.kind}
            disabled={controller.live}
          /> Manifest
        </label>
      </div>
      <label class="block text-xs text-zinc-400">
        {pathLabel}
        <input
          class="input block w-full font-mono text-xs"
          data-testid="bulk-path"
          placeholder={config.source_roots[0] ?? '/data/source/…'}
          bind:value={controller.form.path}
          disabled={controller.live}
        />
      </label>
      <div class="flex flex-wrap items-end gap-3">
        <label class="text-xs text-zinc-400">
          Source tag
          <input
            class="input input-sm block"
            data-testid="bulk-source"
            bind:value={controller.form.source}
            disabled={controller.live}
          />
        </label>
        <button
          class="btn btn-primary btn-sm"
          type="submit"
          data-testid="bulk-start"
          disabled={!controller.canStart}
        >
          {controller.busy && !controller.live ? 'Starting…' : 'Start bulk ingest'}
        </button>
      </div>
      <details class="text-xs text-zinc-400">
        <summary class="cursor-pointer">Advanced</summary>
        <div class="mt-2 flex flex-wrap items-end gap-3">
          <label>
            File extensions
            <input
              class="input input-sm block"
              placeholder={config.default_extensions.join(',')}
              bind:value={controller.form.extensions}
              disabled={controller.live || controller.form.kind === 'manifest'}
            />
          </label>
          <label>
            Images per batch
            <input
              class="input input-sm block w-24"
              type="number"
              min="1"
              max={config.max_batch_size}
              placeholder={String(config.default_batch_size)}
              bind:value={controller.form.batchSize}
              disabled={controller.live}
            />
          </label>
          <label>
            Batches in flight
            <input
              class="input input-sm block w-24"
              type="number"
              min="1"
              max={config.max_concurrency}
              placeholder={String(config.default_concurrency)}
              bind:value={controller.form.concurrency}
              disabled={controller.live}
            />
          </label>
        </div>
      </details>
    </form>

    {#if controller.actionError}
      <p class="text-xs text-red-300" data-testid="bulk-action-error">
        {controller.actionError}
      </p>
    {/if}
    {#if controller.loadError}
      <p class="text-xs text-amber-300" data-testid="bulk-load-error">
        Could not refresh: {controller.loadError}
      </p>
    {/if}

    {#if run && progress}
      <section
        class="space-y-3 rounded border border-zinc-800 p-3"
        data-testid="bulk-run"
        data-status={run.status}
      >
        <div class="flex flex-wrap items-baseline justify-between gap-2">
          <p class="min-w-0 text-xs text-zinc-400">
            <span class="font-mono break-all">{run.source.path}</span>
          </p>
          <span
            class="text-sm font-semibold {STATUS_TONE[run.status]}"
            data-testid="bulk-status"
          >
            {run.cancel_requested ? 'cancelling…' : run.status.replaceAll('_', ' ')}
          </span>
        </div>

        <div
          class="h-2 w-full overflow-hidden rounded bg-zinc-800"
          role="progressbar"
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow={progressPercent(progress)}
        >
          <div
            class="h-full bg-blue-600 transition-[width]"
            style="width: {progressPercent(progress)}%"
          ></div>
        </div>
        <p class="text-xs text-zinc-300" data-testid="bulk-counts">
          {progress.processed} of {progress.discovered}{progress.discovery_complete
            ? ''
            : run.can_cancel
              ? '+ (still scanning)'
              : '+ (found before it stopped)'} images
          {#if progress.skipped > 0}· {progress.skipped} already stored by this run{/if}
        </p>
        <div class="flex flex-wrap gap-2 text-xs">
          <span class="chip" data-testid="bulk-ingested"
            >ingested {progress.ingested}</span
          >
          <span class="chip">duplicate {progress.duplicates}</span>
          <span class="chip {progress.failed > 0 ? 'border-red-700 text-red-200' : ''}"
            >failed {progress.failed}</span
          >
          <span class="chip">crops {progress.crops_indexed}</span>
          <span class="chip"
            >embedded {progress.n_embedded} · not embedded {progress.n_not_embedded}{#if progress.n_embed_failed > 0}
              · encoder failed {progress.n_embed_failed}{/if}</span
          >
        </div>
        <dl class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
          <div>
            <dt class="text-zinc-500">Speed</dt>
            <dd class="text-zinc-200" data-testid="bulk-rate">
              {headlineRate(progress, run.can_cancel)?.toFixed(1) ?? '--'} img/s
            </dd>
          </div>
          <div>
            <dt class="text-zinc-500">Run average</dt>
            <dd class="text-zinc-200">
              {progress.images_per_s?.toFixed(1) ?? '--'} img/s
            </dd>
          </div>
          <div>
            <dt class="text-zinc-500">Time left</dt>
            <dd class="text-zinc-200" data-testid="bulk-eta">
              {formatDuration(progress.eta_s)}
            </dd>
          </div>
          <div>
            <dt class="text-zinc-500">Elapsed</dt>
            <dd class="text-zinc-200">{formatDuration(progress.elapsed_s)}</dd>
          </div>
          <div>
            <dt class="text-zinc-500">In flight</dt>
            <dd class="text-zinc-200">{progress.in_flight}</dd>
          </div>
          <div>
            <dt class="text-zinc-500">Batches</dt>
            <dd class="text-zinc-200">
              {run.options.batch_size} × {run.options.concurrency}
            </dd>
          </div>
        </dl>

        {#if run.error}
          <p class="text-xs text-red-300" data-testid="bulk-run-error">{run.error}</p>
        {/if}

        <div class="flex flex-wrap gap-2">
          {#if run.can_cancel}
            <button
              class="btn btn-danger btn-sm"
              type="button"
              data-testid="bulk-cancel"
              disabled={controller.busy || run.cancel_requested}
              onclick={() => void controller.cancel()}
            >
              Cancel
            </button>
          {/if}
          {#if run.can_resume}
            <button
              class="btn btn-primary btn-sm"
              type="button"
              data-testid="bulk-resume"
              disabled={controller.busy}
              onclick={() => void controller.resume(run)}
            >
              Resume
            </button>
            <span class="self-center text-xs text-zinc-500">
              Images already stored are skipped.
            </span>
          {/if}
        </div>

        {#if stageRows(run).length > 0}
          <details class="text-xs text-zinc-400">
            <summary class="cursor-pointer">Where the time went</summary>
            <table class="mt-2 w-full max-w-md" data-testid="bulk-stages">
              <tbody>
                {#each stageRows(run) as [stage, seconds] (stage)}
                  <tr>
                    <td class="pr-4 font-mono">{stage}</td>
                    <td class="text-right">{seconds.toFixed(1)} s</td>
                  </tr>
                {/each}
              </tbody>
            </table>
            <p class="mt-1 text-zinc-500">
              Seconds summed over concurrent work, so a stage can exceed the elapsed time.
            </p>
          </details>
        {/if}

        {#if controller.failures && controller.failures.total > 0}
          <div class="space-y-1" data-testid="bulk-failures">
            <p class="text-xs font-semibold text-red-300">
              {controller.failures.total} failed image{controller.failures.total === 1
                ? ''
                : 's'}
            </p>
            <ul class="max-h-48 overflow-y-auto text-xs">
              {#each controller.failures.items as f (f.path)}
                <li class="border-b border-zinc-900 py-1 break-all">
                  <span class="font-mono">{f.path}</span>
                  <span class="text-zinc-500"> [{f.error_kind}]</span>
                  <span class="text-red-300"> — {f.error}</span>
                </li>
              {/each}
            </ul>
            {#if controller.failures.total > FAILURES_PAGE_SIZE}
              <div class="flex gap-2 text-xs">
                <button
                  class="btn btn-sm"
                  type="button"
                  disabled={controller.failuresOffset === 0}
                  onclick={() =>
                    void controller.loadFailures(
                      Math.max(0, controller.failuresOffset - FAILURES_PAGE_SIZE),
                    )}>Previous</button
                >
                <button
                  class="btn btn-sm"
                  type="button"
                  disabled={controller.failuresOffset + FAILURES_PAGE_SIZE >=
                    controller.failures.total}
                  onclick={() =>
                    void controller.loadFailures(
                      controller.failuresOffset + FAILURES_PAGE_SIZE,
                    )}>Next</button
                >
              </div>
            {/if}
          </div>
        {/if}
      </section>
    {/if}

    {#if controller.recent.length > 1}
      <details class="text-xs text-zinc-400" data-testid="bulk-recent">
        <summary class="cursor-pointer">Recent runs</summary>
        <ul class="mt-2 space-y-1">
          {#each controller.recent as r (r.run_id)}
            <li class="flex flex-wrap items-baseline gap-2">
              <span class="font-mono break-all">{r.source.path}</span>
              <span class={STATUS_TONE[r.status]}>{r.status.replaceAll('_', ' ')}</span>
              <span>{r.progress.ingested} ingested · {r.progress.failed} failed</span>
              <button
                class="text-blue-300 hover:underline"
                type="button"
                onclick={() => controller.select(r)}>View</button
              >
            </li>
          {/each}
        </ul>
      </details>
    {/if}
  {/if}
</div>
