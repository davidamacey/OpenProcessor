<script lang="ts">
  /**
   * Test-on-crop for one region profile (`POST /region_profiles/test`).
   * Runs the profile's legs over one stored crop and
   * shows what the server returned: the legs and their candidates (kept
   * and dropped), the candidates drawn over the source image and in the
   * crop's own frame, the preview item, and, when asked, the VLM verify.
   * Nothing is written. Shown whenever the editor loads (the schema
   * serves no `testable` flag).
   */
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import TestPreviewItem from '$components/config/TestPreviewItem.svelte';
  import CropFrameShapes from '$components/configTest/CropFrameShapes.svelte';
  import TestRefs from '$components/configTest/TestRefs.svelte';
  import TestRunOutcome from '$components/configTest/TestRunOutcome.svelte';
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
  class="surface flex flex-col gap-3 p-4 text-body"
  data-testid="profile-test-panel"
  aria-label="Test on a crop"
>
  <h2 class="text-title font-semibold">Test on a crop</h2>
  <p class="text-meta text-fg-2">
    Runs this profile's detection on one stored crop and shows every candidate it found
    and what the selection kept. Nothing is written.
  </p>
  <div class="grid gap-2 sm:grid-cols-2">
    <label class="flex flex-col gap-1 text-meta text-fg-2 sm:col-span-2">
      Crop id (from review or browse)
      <input
        class="input input-sm font-mono {t.missingCropIds.length > 0
          ? 'border-danger'
          : ''}"
        bind:value={t.cropId}
        placeholder="c_123"
        aria-invalid={t.missingCropIds.length > 0}
        data-testid="profile-test-crop-id"
      />
      {#if t.missingCropIds.length > 0}
        <span class="text-danger-text" data-testid="test-missing-ids">
          Not found:
          {#each t.missingCropIds as id, i (id)}<code class="font-mono">{id}</code>{i <
            t.missingCropIds.length - 1
              ? ', '
              : ''}{/each}
        </span>
      {/if}
    </label>
    <label class="flex flex-col gap-1 text-meta text-fg-2">
      Profile version
      <select class="select select-sm" bind:value={t.source} data-testid="test-source">
        {#if !savedOnly}<option value="draft">Unsaved draft</option>{/if}
        <option value="saved"
          >{revision == null ? 'Saved profile' : `Saved revision ${revision}`}</option
        >
      </select>
    </label>
    <label class="flex flex-col gap-1 text-meta text-fg-2">
      Segmenter prompt (optional, replaces the profile's)
      <input
        class="input input-sm"
        bind:value={t.segmenterPrompt}
        data-testid="test-segmenter-prompt"
      />
    </label>
    <label class="flex items-center gap-2 text-meta text-fg-2 sm:col-span-2">
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
  <TestRunOutcome {t} onrun={run}>
    {#snippet result(r)}
      <p class="text-meta">
        <span class="text-fg-3">Profile</span>
        <span class="font-mono" data-testid="test-profile-ref">{refText(r.profile)}</span>
      </p>
      {#if !r.testable}
        <p class="text-warn-text" data-testid="test-not-testable">
          {r.reason ?? 'The server reports this item cannot be tested.'}
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
        <h3 class="mb-1 text-meta font-semibold text-fg-2">
          {PREVIEW_HEADINGS[r.preview_basis]}
        </h3>
        <div class="grid gap-3 lg:grid-cols-[minmax(0,36rem)_auto] lg:justify-start">
          <TestPreviewItem preview={r.preview} extraShapes={sourceShapes} />
          {#if parentShapes.length > 0}
            <div>
              <p class="mb-1 text-caption text-fg-3">In the crop's frame</p>
              <CropFrameShapes cropId={r.crop_id} shapes={parentShapes} />
            </div>
          {/if}
        </div>
        <details class="mt-1">
          <summary class="cursor-pointer text-caption text-fg-2">preview_item</summary>
          <pre
            class="max-h-60 overflow-auto rounded bg-surface-1 p-2 text-caption">{JSON.stringify(
              r.preview_item,
              null,
              2,
            )}</pre>
        </details>
      </div>

      {#if r.verify}
        {@const v = r.verify}
        <div
          class="flex flex-col gap-2 rounded border border-line p-3"
          data-testid="test-verify-block"
        >
          <h3 class="text-meta font-semibold text-fg-2">VLM verification</h3>
          <TestRefs
            pack={v.pack}
            packLabel="Prompt pack"
            vlm={v.vlm}
            latencyMs={v.latency_ms}
          />
          <p class="text-meta" data-testid="test-parse-status">
            {#if v.parse_ok}
              <span class="rounded border border-success/40 px-1.5 text-success-text"
                >parsed</span
              >
            {:else}
              <span class="rounded border border-danger/40 px-1.5 text-danger-text"
                >not parsed</span
              >
              {#if v.parse_error}<span class="ml-1 text-danger-text">{v.parse_error}</span
                >{/if}
            {/if}
          </p>
          <details>
            <summary class="cursor-pointer text-meta text-fg-2">Prompt sent</summary>
            <p class="mt-1 text-caption text-fg-3">System</p>
            <pre
              class="max-h-60 overflow-auto rounded bg-surface-1 p-2 text-caption whitespace-pre-wrap"
              data-testid="test-prompt-system">{v.prompt.system ?? ''}</pre>
            <p class="mt-1 text-caption text-fg-3">User</p>
            <pre
              class="max-h-60 overflow-auto rounded bg-surface-1 p-2 text-caption whitespace-pre-wrap"
              data-testid="test-prompt-user">{v.prompt.user_text ?? ''}</pre>
          </details>
          <div>
            <p class="text-meta text-fg-3">Raw reply</p>
            <pre
              class="max-h-60 overflow-auto rounded bg-surface-1 p-2 text-caption whitespace-pre-wrap"
              data-testid="test-raw-reply">{v.raw_reply ?? ''}</pre>
          </div>
          {#if v.reasoning}
            <div>
              <p class="text-meta text-fg-3">Reasoning</p>
              <pre
                class="max-h-40 overflow-auto rounded bg-surface-1 p-2 text-caption whitespace-pre-wrap">{v.reasoning}</pre>
            </div>
          {/if}
        </div>
      {/if}
    {/snippet}
  </TestRunOutcome>
</section>
