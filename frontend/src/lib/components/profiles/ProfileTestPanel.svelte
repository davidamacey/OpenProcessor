<script lang="ts">
  /**
   * Test-on-crop for one region profile (`POST /region_profiles/test`, W5;
   * §5.2, §7.6, §7.7). Runs the profile's legs over one stored crop and
   * shows what the server returned: the legs and their candidates (kept
   * and dropped), the candidates drawn over the source image and in the
   * crop's own frame, the preview item, and, when asked, the VLM verify.
   * Nothing is written. Shown whenever the editor loads (the schema
   * serves no `testable` flag, question W5-1).
   */
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import TestPreviewItem from '$components/config/TestPreviewItem.svelte';
  import CropFrameShapes from '$components/configTest/CropFrameShapes.svelte';
  import TestRefs from '$components/configTest/TestRefs.svelte';
  import TestVlmPicker from '$components/configTest/TestVlmPicker.svelte';
  import ProfileTestLegs from '$components/profiles/ProfileTestLegs.svelte';
  import { candidateShapes } from '$lib/configTest/overlayShapes';
  import { refText } from '$lib/configTest/refText';
  import { createProfileTest } from '$lib/profiles/profileTestController.svelte';
  import type { RegionProfileBody } from '$lib/types_profiles';

  interface Props {
    name: string;
    /** The saved revision on screen (null for an env/registered profile). */
    revision: number | null;
    /** True when there is no draft to test (read-only profile, or a
     *  revision being viewed): only the saved source is offered. */
    savedOnly: boolean;
    draft: RegionProfileBody;
  }

  let { name, revision, savedOnly, draft }: Props = $props();

  const t = createProfileTest();

  $effect(() => {
    t.setSavedOnly(savedOnly);
  });
  $effect(() => () => t.stop());

  function run(): void {
    void t.run({ name, revision, draft });
  }

  /** The served heading for what the preview item reflects. */
  const PREVIEW_HEADINGS = {
    selection_accepted: 'Selection (not verified)',
    vlm_verdicts: 'VLM verdicts',
  } as const;

  const sourceShapes = $derived(t.result ? candidateShapes(t.result.legs, 'source') : []);
  const parentShapes = $derived(t.result ? candidateShapes(t.result.legs, 'parent') : []);
</script>

<section
  class="surface flex flex-col gap-3 p-4 text-sm"
  data-testid="profile-test-panel"
  aria-label="Test on a crop"
>
  <h2 class="text-base font-semibold">Test on a crop</h2>
  <p class="text-xs text-zinc-400">
    Runs this profile's detection on one stored crop and shows every candidate it found
    and what the selection kept. Nothing is written.
  </p>
  <div class="grid gap-2 sm:grid-cols-2">
    <label class="flex flex-col gap-1 text-xs text-zinc-400 sm:col-span-2">
      Crop id (from review or browse)
      <input
        class="input input-sm font-mono {t.missingCropIds.length > 0
          ? 'border-red-500'
          : ''}"
        bind:value={t.cropId}
        placeholder="c_123"
        aria-invalid={t.missingCropIds.length > 0}
        data-testid="profile-test-crop-id"
      />
      {#if t.missingCropIds.length > 0}
        <span class="text-red-300" data-testid="test-missing-ids">
          Not found:
          {#each t.missingCropIds as id, i (id)}<code class="font-mono">{id}</code>{i <
            t.missingCropIds.length - 1
              ? ', '
              : ''}{/each}
        </span>
      {/if}
    </label>
    <label class="flex flex-col gap-1 text-xs text-zinc-400">
      Profile version
      <select class="select select-sm" bind:value={t.source} data-testid="test-source">
        {#if !savedOnly}<option value="draft">Unsaved draft</option>{/if}
        <option value="saved"
          >{revision == null ? 'Saved profile' : `Saved revision ${revision}`}</option
        >
      </select>
    </label>
    <label class="flex flex-col gap-1 text-xs text-zinc-400">
      Segmenter prompt (optional, replaces the profile's)
      <input
        class="input input-sm"
        bind:value={t.segmenterPrompt}
        data-testid="test-segmenter-prompt"
      />
    </label>
    <label class="flex items-center gap-2 text-xs text-zinc-300 sm:col-span-2">
      <input type="checkbox" bind:checked={t.verify} data-testid="test-verify" />
      Verify with the VLM (uses the active prompt pack)
    </label>
    {#if t.verify}
      <div class="sm:col-span-2">
        <TestVlmPicker
          selection={t.vlmSelection}
          disabled={t.running}
          onchange={(next) => {
            t.vlmSelection = next;
          }}
        />
      </div>
    {/if}
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
      <p class="text-xs">
        <span class="text-zinc-500">Profile</span>
        <span class="font-mono" data-testid="test-profile-ref">{refText(r.profile)}</span>
      </p>
      {#if !r.item_eligible}
        <p class="text-amber-300" data-testid="test-not-eligible">
          This item is not eligible for this profile.
        </p>
      {/if}
      {#if r.validation}
        <ConfigIssueList
          issues={[...r.validation.errors, ...r.validation.warnings]}
          showField
        />
      {/if}

      <ProfileTestLegs legs={r.legs} />

      <div>
        <h3 class="mb-1 text-xs font-semibold text-zinc-300">
          {PREVIEW_HEADINGS[r.preview_basis]}
        </h3>
        <div class="grid gap-3 lg:grid-cols-[minmax(0,36rem)_auto] lg:justify-start">
          <TestPreviewItem preview={r.preview} extraShapes={sourceShapes} />
          {#if parentShapes.length > 0}
            <div>
              <p class="mb-1 text-[11px] text-zinc-500">In the crop's frame</p>
              <CropFrameShapes cropId={r.crop_id} shapes={parentShapes} />
            </div>
          {/if}
        </div>
        <details class="mt-1">
          <summary class="cursor-pointer text-[11px] text-zinc-400">preview_item</summary>
          <pre
            class="max-h-60 overflow-auto rounded bg-zinc-900 p-2 text-[11px]">{JSON.stringify(
              r.preview_item,
              null,
              2,
            )}</pre>
        </details>
      </div>

      {#if r.verify}
        {@const v = r.verify}
        <div
          class="flex flex-col gap-2 rounded border border-zinc-800 p-3"
          data-testid="test-verify-block"
        >
          <h3 class="text-xs font-semibold text-zinc-300">VLM verification</h3>
          <TestRefs
            pack={v.pack}
            packLabel="Prompt pack"
            vlm={v.vlm}
            latencyMs={v.latency_ms}
          />
          <p class="text-xs" data-testid="test-parse-status">
            {#if v.parse_ok}
              <span class="rounded border border-emerald-500/40 px-1.5 text-emerald-300"
                >parsed</span
              >
            {:else}
              <span class="rounded border border-red-500/40 px-1.5 text-red-300"
                >not parsed</span
              >
              {#if v.parse_error}<span class="ml-1 text-red-300">{v.parse_error}</span
                >{/if}
            {/if}
          </p>
          <details>
            <summary class="cursor-pointer text-xs text-zinc-300">Prompt sent</summary>
            <p class="mt-1 text-[11px] text-zinc-500">System</p>
            <pre
              class="max-h-60 overflow-auto rounded bg-zinc-900 p-2 text-[11px] whitespace-pre-wrap"
              data-testid="test-prompt-system">{v.prompt.system ?? ''}</pre>
            <p class="mt-1 text-[11px] text-zinc-500">User</p>
            <pre
              class="max-h-60 overflow-auto rounded bg-zinc-900 p-2 text-[11px] whitespace-pre-wrap"
              data-testid="test-prompt-user">{v.prompt.user_text ?? ''}</pre>
          </details>
          <div>
            <p class="text-xs text-zinc-500">Raw reply</p>
            <pre
              class="max-h-60 overflow-auto rounded bg-zinc-900 p-2 text-[11px] whitespace-pre-wrap"
              data-testid="test-raw-reply">{v.raw_reply ?? ''}</pre>
          </div>
          {#if v.reasoning}
            <div>
              <p class="text-xs text-zinc-500">Reasoning</p>
              <pre
                class="max-h-40 overflow-auto rounded bg-zinc-900 p-2 text-[11px] whitespace-pre-wrap">{v.reasoning}</pre>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
</section>
