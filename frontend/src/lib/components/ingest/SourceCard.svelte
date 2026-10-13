<script lang="ts">
  /**
   * One ingest source as the backend served it: location, triggers with their last
   * and next times, queue counts, the scan in progress, the last run, the objects
   * that failed, and the actions the backend allows now.
   */
  import { formatWhen, scanText, STATUS_TONE } from '$lib/ingest/sourcesFormat';
  import type { IngestSources } from '$lib/ingest/sourcesController.svelte';
  import type { IngestSourceStatus } from '$lib/types_ingestSources';

  interface Props {
    source: IngestSourceStatus;
    controller: IngestSources;
  }
  let { source, controller }: Props = $props();

  const triggers = $derived(source.triggers);
  const noTriggers = $derived(
    !triggers.watch?.enabled && !triggers.schedule?.enabled && !triggers.webhook_enabled,
  );
</script>

<article
  class="space-y-2 rounded border border-zinc-800 p-3"
  data-testid="source-card"
  data-source-id={source.source_id}
>
  <header class="flex flex-wrap items-baseline justify-between gap-2">
    <div class="min-w-0">
      <h3 class="text-sm font-semibold text-zinc-100">{source.name}</h3>
      <p class="font-mono text-xs break-all text-zinc-400">{source.location}</p>
    </div>
    <span class="chip">{source.backend === 's3' ? 'S3' : 'folder'}</span>
  </header>

  <div class="flex flex-wrap gap-2 text-xs" data-testid="source-counts">
    <span class="chip">pending {source.counts.pending}</span>
    <span class="chip">ingested {source.counts.done}</span>
    <span class="chip {source.counts.failed > 0 ? 'border-amber-700 text-amber-200' : ''}"
      >retrying {source.counts.failed}</span
    >
    <span class="chip {source.counts.dead > 0 ? 'border-red-700 text-red-200' : ''}"
      >gave up {source.counts.dead}</span
    >
  </div>

  {#if source.scan.active}
    <p class="text-xs text-blue-300" data-testid="source-scan">{scanText(source.scan)}</p>
  {/if}

  <dl
    class="grid grid-cols-1 gap-x-4 gap-y-1 text-xs sm:grid-cols-2"
    data-testid="source-triggers"
  >
    {#if triggers.watch?.enabled}
      <div>
        <dt class="text-zinc-500">Watch every {triggers.watch.interval_s}s</dt>
        <dd class="text-zinc-200">
          last {formatWhen(source.watch.last_poll_at)} · next {formatWhen(
            source.watch.next_poll_at,
          )}
        </dd>
        {#if source.watch.error}
          <dd class="text-red-300" data-testid="source-watch-error">
            {source.watch.error}
            {#if source.watch.backoff_until}(retry after {formatWhen(
                source.watch.backoff_until,
              )}){/if}
          </dd>
        {/if}
      </div>
    {/if}
    {#if triggers.schedule?.enabled}
      <div>
        <dt class="text-zinc-500">
          Schedule <span class="font-mono">{triggers.schedule.cron}</span> UTC
        </dt>
        <dd class="text-zinc-200">
          last {formatWhen(source.schedule.last_fire_at)} · next {formatWhen(
            source.schedule.next_fire_at,
          )}
        </dd>
      </div>
    {/if}
    {#if triggers.webhook_enabled}
      <div class="sm:col-span-2">
        <dt class="text-zinc-500">Webhook</dt>
        <dd class="font-mono break-all text-zinc-200" data-testid="source-webhook-path">
          {source.webhook.path}
        </dd>
        <dd class="text-zinc-400">
          {source.webhook.received} received · {source.webhook.queued} queued · {source
            .webhook.duplicates} repeated · {source.webhook.ignored} ignored · last {formatWhen(
            source.webhook.last_event_at,
          )}
        </dd>
      </div>
    {/if}
    {#if noTriggers}
      <div class="text-zinc-500">No triggers: runs only when you start it.</div>
    {/if}
  </dl>

  {#if source.last_run}
    <p class="text-xs text-zinc-300" data-testid="source-last-run">
      Last run
      <span class={STATUS_TONE[source.last_run.status] ?? ''}
        >{source.last_run.status.replaceAll('_', ' ')}</span
      >
      · {formatWhen(source.last_run.started_at)} · ingested {source.last_run.ingested} · duplicates
      {source.last_run.duplicates} · failed {source.last_run.failed}
      {#if source.last_run.error}<span class="text-red-300">
          — {source.last_run.error}</span
        >{/if}
    </p>
  {:else if source.scan.last}
    <p class="text-xs text-zinc-400">
      Last scan {formatWhen(source.scan.last.finished_at)}: {source.scan.last.listed} listed,
      {source.scan.last.queued} new.
    </p>
  {/if}

  {#if source.errors && source.errors.length > 0}
    <details class="text-xs" data-testid="source-errors">
      <summary class="cursor-pointer text-red-300"
        >{source.errors.length} object{source.errors.length === 1 ? '' : 's'} with errors</summary
      >
      <ul class="mt-1 max-h-40 overflow-y-auto">
        {#each source.errors as e (e.key + e.etag)}
          <li class="border-b border-zinc-900 py-1 break-all">
            <span class="font-mono">{e.key}</span>
            <span class="text-zinc-500"> [{e.state}, {e.attempts} tries]</span>
            <span class="text-red-300"> — {e.error}</span>
          </li>
        {/each}
      </ul>
    </details>
  {/if}

  <div class="flex flex-wrap items-center gap-2">
    <button
      class="btn btn-primary btn-sm"
      type="button"
      data-testid="source-run"
      disabled={!source.can_run || controller.busy}
      onclick={() => void controller.run(source.source_id)}
    >
      Run now
    </button>
    {#if source.can_retry_dead}
      <button
        class="btn btn-sm"
        type="button"
        data-testid="source-retry"
        disabled={controller.busy}
        onclick={() => void controller.retryDead(source.source_id)}
      >
        Retry the {source.counts.dead} that gave up
      </button>
    {/if}
    <button class="btn btn-sm" type="button" onclick={() => controller.openEdit(source)}>
      Edit
    </button>
    <button
      class="btn btn-sm"
      type="button"
      disabled={controller.busy}
      onclick={() => void controller.remove(source.source_id)}
    >
      Remove
    </button>
    {#if !source.can_run && source.blocked_message}
      <span class="text-xs text-amber-300" data-testid="source-blocked"
        >{source.blocked_message}</span
      >
    {/if}
  </div>
</article>
