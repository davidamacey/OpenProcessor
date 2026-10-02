<script lang="ts">
  /**
   * Test-on-crop for one pack (`POST /prompt_packs/test`, W5; §5.1, §7.5,
   * §7.6 item 3). Absent unless the served schema marks a call
   * `testable`. Shows exactly what the server returns: the prompt it
   * built, the raw reply, the parsed models and the preview item.
   */
  import { getThumbUrl } from '$lib/api';
  import { createPackTest } from '$lib/packs/packTestController.svelte';
  import type { PackSchemaCall, PromptPackBody } from '$lib/types_packs';
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import PackTestPreview from './PackTestPreview.svelte';

  interface Props {
    calls: PackSchemaCall[];
    name: string;
    /** The saved revision on screen (null for builtin/file packs). */
    revision: number | null;
    /** True when there is no draft to test (read-only pack, or a
     *  revision being viewed): only the saved source is offered. */
    savedOnly: boolean;
    draft: PromptPackBody;
  }

  let { calls, name, revision, savedOnly, draft }: Props = $props();

  const t = createPackTest();
  const testable = $derived(calls.filter((c) => c.testable));

  $effect(() => {
    if (t.call === '' && testable.length > 0) t.call = testable[0]!.id;
  });
  $effect(() => {
    t.setSavedOnly(savedOnly);
  });
  $effect(() => () => t.stop());

  function run(): void {
    void t.run({ name, revision, draft });
  }

  const parsedBlocks = (r: {
    parsed_combined: unknown;
    parsed_class: unknown;
    parsed_region: unknown;
    parsed_visible: unknown;
  }): Array<[string, unknown]> =>
    (
      [
        ['parsed_combined', r.parsed_combined],
        ['parsed_class', r.parsed_class],
        ['parsed_region', r.parsed_region],
        ['parsed_visible', r.parsed_visible],
      ] as Array<[string, unknown]>
    ).filter(([, v]) => v != null);
</script>

{#if testable.length > 0}
  <section
    class="surface flex flex-col gap-3 p-4 text-sm"
    data-testid="pack-test-panel"
    aria-label="Test on a crop"
  >
    <h2 class="text-base font-semibold">Test on a crop</h2>
    <p class="text-xs text-zinc-400">
      Runs one call on real crops and shows what the VLM answered. Nothing is written.
    </p>
    <div class="grid gap-2 sm:grid-cols-2">
      <label class="flex flex-col gap-1 text-xs text-zinc-400">
        Call
        <select class="select select-sm" bind:value={t.call} data-testid="test-call">
          {#each testable as c (c.id)}
            <option value={c.id}>{c.label}</option>
          {/each}
        </select>
      </label>
      <label class="flex flex-col gap-1 text-xs text-zinc-400">
        Pack version
        <select class="select select-sm" bind:value={t.source} data-testid="test-source">
          {#if !savedOnly}<option value="draft">Unsaved draft</option>{/if}
          <option value="saved"
            >{revision == null ? 'Saved pack' : `Saved revision ${revision}`}</option
          >
        </select>
      </label>
      <label class="flex flex-col gap-1 text-xs text-zinc-400 sm:col-span-2">
        Crop ids (from review or browse, separated by commas or spaces)
        <input
          class="input input-sm font-mono"
          bind:value={t.cropIdsText}
          placeholder="c_123, c_456"
          data-testid="test-crop-ids"
        />
      </label>
      <label class="flex flex-col gap-1 text-xs text-zinc-400">
        Boxes on the crop
        <select class="select select-sm" bind:value={t.useRegionBox}>
          <option value="">Server default</option>
          <option value="current">current: every stored box, numbered</option>
          <option value="none">none: no boxes</option>
        </select>
      </label>
    </div>
    <div>
      <button
        type="button"
        class="btn btn-primary btn-sm"
        disabled={!t.canRun}
        onclick={run}
        data-testid="test-run">{t.running ? 'Testing…' : 'Run test'}</button
      >
    </div>

    {#if t.error}
      <div class="space-y-1" data-testid="test-error">
        <p class="text-red-300">{t.error}</p>
        {#if t.errorReport}
          <ConfigIssueList
            issues={[...t.errorReport.errors, ...t.errorReport.warnings]}
            showField
          />
        {/if}
      </div>
    {/if}

    {#if t.result}
      {@const r = t.result}
      <div
        class="flex flex-col gap-3 border-t border-zinc-800 pt-3"
        data-testid="test-result"
      >
        <dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
          <dt class="text-zinc-500">Pack</dt>
          <dd class="font-mono">
            {r.pack.draft
              ? 'unsaved draft'
              : `${r.pack.name ?? ''}${r.pack.revision == null ? '' : ` r${r.pack.revision}`}`}
          </dd>
          {#if r.vlm}
            <dt class="text-zinc-500">VLM</dt>
            <dd class="font-mono">
              {r.vlm.name ?? '—'} · {r.vlm.resolved_model ?? r.vlm.model ?? '—'}
              {#if r.vlm.sends_images_externally}
                <span class="ml-1 text-amber-300"
                  >sends images outside this deployment</span
                >
              {/if}
            </dd>
          {/if}
          <dt class="text-zinc-500">Latency</dt>
          <dd class="font-mono">{r.latency_ms} ms</dd>
        </dl>
        {#if r.validation}
          <ConfigIssueList
            issues={[...r.validation.errors, ...r.validation.warnings]}
            showField
          />
        {/if}
        <details>
          <summary class="cursor-pointer text-xs text-zinc-300">Prompt sent</summary>
          <p class="mt-1 text-[11px] text-zinc-500">System</p>
          <pre
            class="max-h-60 overflow-auto rounded bg-zinc-900 p-2 text-[11px] whitespace-pre-wrap"
            data-testid="test-prompt-system">{r.prompt.system}</pre>
          <p class="mt-1 text-[11px] text-zinc-500">User</p>
          <pre
            class="max-h-60 overflow-auto rounded bg-zinc-900 p-2 text-[11px] whitespace-pre-wrap"
            data-testid="test-prompt-user">{r.prompt.user_text}</pre>
        </details>
        <div>
          <p class="text-xs text-zinc-500">Raw reply</p>
          <pre
            class="max-h-60 overflow-auto rounded bg-zinc-900 p-2 text-[11px] whitespace-pre-wrap"
            data-testid="test-raw-reply">{r.raw_reply}</pre>
        </div>
        {#if r.reasoning}
          <div>
            <p class="text-xs text-zinc-500">Reasoning</p>
            <pre
              class="max-h-40 overflow-auto rounded bg-zinc-900 p-2 text-[11px] whitespace-pre-wrap">{r.reasoning}</pre>
          </div>
        {/if}
        {#each r.results as res, idx (res.crop_id + ':' + (res.box_id ?? '') + ':' + idx)}
          <article
            class="flex flex-col gap-2 rounded border border-zinc-800 p-3"
            data-testid="test-result-item"
            data-crop-id={res.crop_id}
          >
            <div class="flex flex-wrap items-center gap-2 text-xs">
              <img
                src={getThumbUrl(res.crop_id, 96)}
                alt="crop {res.crop_id}"
                class="h-12 w-12 rounded object-cover"
              />
              <code class="font-mono">{res.crop_id}</code>
              {#if res.box_id}<code class="font-mono text-zinc-400">{res.box_id}</code
                >{/if}
              {#if res.parse_ok}
                <span
                  class="rounded border border-emerald-500/40 px-1.5 text-emerald-300"
                  data-testid="test-parse-ok">parsed</span
                >
              {:else}
                <span
                  class="rounded border border-red-500/40 px-1.5 text-red-300"
                  data-testid="test-parse-failed">not parsed</span
                >
                {#if res.parse_error}<span class="text-red-300">{res.parse_error}</span
                  >{/if}
              {/if}
            </div>
            {#each parsedBlocks(res) as [key, value] (key)}
              <div>
                <p class="font-mono text-[11px] text-zinc-500">{key}</p>
                <pre
                  class="max-h-48 overflow-auto rounded bg-zinc-900 p-2 text-[11px]"
                  data-testid="test-parsed">{JSON.stringify(value, null, 2)}</pre>
              </div>
            {/each}
            {#if res.preview}
              <p class="text-[11px] text-zinc-500">
                The item as this reply would leave it
              </p>
              <PackTestPreview preview={res.preview} />
              <details>
                <summary class="cursor-pointer text-[11px] text-zinc-400"
                  >preview_item</summary
                >
                <pre
                  class="max-h-60 overflow-auto rounded bg-zinc-900 p-2 text-[11px]">{JSON.stringify(
                    res.preview_item,
                    null,
                    2,
                  )}</pre>
              </details>
            {/if}
          </article>
        {/each}
      </div>
    {/if}
  </section>
{/if}
