<script lang="ts">
  /**
   * The assist dialog: pick what the VLM sees (the user's selection, every
   * member, or a seeded representative sample), what to do (classify,
   * attributes, verify, segment) and which VLM and prompt pack, read the
   * server's dry run, then start. The server sizes the sample, estimates the
   * calls and refuses what it will not do; this dialog sends the request and
   * shows the answers (`assistRunController`).
   */
  import { untrack } from 'svelte';
  import Modal from '$components/Modal.svelte';
  import VlmRunPicker from '$components/vlm/VlmRunPicker.svelte';
  import AssistPlanSummary from '$components/assistRun/AssistPlanSummary.svelte';
  import AssistProposalCard from '$components/assistRun/AssistProposalCard.svelte';
  import AssistRunProgress from '$components/assistRun/AssistRunProgress.svelte';
  import {
    isLiveStatus,
    type AssistRunController,
    type SampleChoice,
  } from '$lib/assistRun/assistRunController.svelte';
  import { isPromptPackAvailable, selectableAxisEntries } from '$lib/strategies';
  import { MIN_AGREEMENT_RANGE, type AssistAction } from '$lib/types_assist';
  import { strategiesStore } from '$stores/strategies.svelte';

  interface Props {
    ctl: AssistRunController;
    /** Called once the dialog closes, so the page can reload what a run changed. */
    onclosed?: () => void;
  }

  let { ctl, onclosed }: Props = $props();

  $effect(() => {
    void strategiesStore.init();
  });

  // Re-ask the server for a dry run whenever the request changes, until a run starts.
  $effect(() => {
    if (!ctl.isOpen || ctl.run) return;
    void ctl.requestKey;
    untrack(() => ctl.schedulePlan());
  });

  const packs = $derived(selectableAxisEntries(strategiesStore.methods.prompt_packs));
  const packAvailable = $derived(
    isPromptPackAvailable(strategiesStore.methods.prompt_packs),
  );
  const live = $derived(isLiveStatus(ctl.run?.status));

  const CHOICE_LABEL: Record<SampleChoice, string> = {
    selected: 'Selected',
    all: 'All members',
    representative: 'Random representative sample',
  };
  const CHOICE_HINT: Record<SampleChoice, string> = {
    selected: 'Only the crops or boxes you picked.',
    all: 'Every member of the cluster or filter the VLM may see.',
    representative:
      'A seeded sample that grows with the size of the scope. The answers are votes for a proposed label.',
  };
  const ACTION_LABEL: Record<AssistAction, { name: string; hint: string }> = {
    classify: {
      name: 'Classify',
      hint: 'Suggest a class. Locked, validated and held-out items keep theirs.',
    },
    attributes: {
      name: 'Attributes',
      hint: 'Fill the attributes the prompt pack declares. Suggestions, not labels.',
    },
    verify: {
      name: 'Verify boxes',
      hint: 'The VLM judges each box. A human-owned box is never touched.',
    },
    segment: {
      name: 'Segment',
      hint: 'Re-queue the items for the segmenter in the region worker.',
    },
  };
  function selectedCount(): string {
    const t = ctl.opener?.selected;
    if (t?.kind === 'crops') return ` (${t.crop_ids.length})`;
    if (t?.kind === 'boxes') return ` (${t.boxes.length})`;
    return '';
  }

  function numberOrNull(e: Event): number | null {
    const v = (e.currentTarget as HTMLInputElement).valueAsNumber;
    return Number.isNaN(v) ? null : v;
  }

  function close(): void {
    ctl.close();
    onclosed?.();
  }
</script>

{#if ctl.isOpen && ctl.opener}
  <Modal
    label="Assist with the VLM"
    onclose={live ? undefined : close}
    panelClass="flex max-h-[92vh] w-full max-w-2xl flex-col rounded-lg border border-line bg-surface-0 shadow-2xl"
    data-testid="assist-dialog"
  >
    <div class="flex min-h-0 grow flex-col gap-4 overflow-y-auto p-5">
      <header class="flex flex-col gap-1">
        <h2 class="text-title font-semibold">Assist: {ctl.opener.title}</h2>
        <p class="text-meta text-fg-3">
          The VLM suggests; it never overwrites a label a human or an import set, or an
          item held out for testing.
        </p>
      </header>

      {#if !ctl.run}
        <fieldset class="flex flex-col gap-1.5">
          <legend class="mb-1 text-body font-medium text-fg-1">What the VLM sees</legend>
          {#each ctl.choices as c (c)}
            <label
              class="flex cursor-pointer items-start gap-2 rounded border px-3 py-2 text-body {ctl.choice ===
              c
                ? 'border-accent/60 bg-accent/10'
                : 'border-line hover:border-line-strong'}"
              data-testid="assist-choice-{c}"
            >
              <input
                type="radio"
                name="assist-choice"
                class="mt-1"
                checked={ctl.choice === c}
                onchange={() => ctl.setChoice(c)}
              />
              <span class="flex min-w-0 flex-col gap-0.5">
                <span class="font-medium text-fg-1"
                  >{CHOICE_LABEL[c]}{c === 'selected' ? selectedCount() : ''}</span
                >
                <span class="text-meta text-fg-2">{CHOICE_HINT[c]}</span>
              </span>
            </label>
          {/each}
          {#if ctl.choice === 'representative'}
            <div
              class="grid grid-cols-1 gap-2 sm:grid-cols-3"
              data-testid="assist-sample-opts"
            >
              <label class="flex flex-col gap-1 text-meta text-fg-2">
                Sample size
                <input
                  class="input"
                  type="number"
                  min="1"
                  placeholder="grows with the scope"
                  value={ctl.size ?? ''}
                  oninput={(e) => (ctl.size = numberOrNull(e))}
                  data-testid="assist-size"
                />
              </label>
              <label class="flex flex-col gap-1 text-meta text-fg-2">
                Seed
                <input
                  class="input"
                  type="number"
                  value={ctl.seed}
                  oninput={(e) => (ctl.seed = numberOrNull(e) ?? 0)}
                  data-testid="assist-seed"
                />
              </label>
            </div>
          {/if}
        </fieldset>

        <fieldset class="flex flex-col gap-1.5">
          <legend class="mb-1 text-body font-medium text-fg-1">What to do</legend>
          <div class="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {#each ctl.allowedActions as a (a)}
              <label
                class="flex cursor-pointer items-start gap-2 rounded border px-3 py-2 text-body {ctl.actions.includes(
                  a,
                )
                  ? 'border-accent/60 bg-accent/10'
                  : 'border-line hover:border-line-strong'}"
                data-testid="assist-action-{a}"
              >
                <input
                  type="checkbox"
                  class="mt-1"
                  checked={ctl.actions.includes(a)}
                  onchange={() => ctl.toggleAction(a)}
                />
                <span class="flex min-w-0 flex-col gap-0.5">
                  <span class="font-medium text-fg-1">{ACTION_LABEL[a].name}</span>
                  <span class="text-meta text-fg-2">{ACTION_LABEL[a].hint}</span>
                </span>
              </label>
            {/each}
          </div>
          {#if ctl.target?.kind === 'cluster' && ctl.actions.includes('classify')}
            <label class="flex flex-col gap-1 pl-1 text-meta text-fg-2">
              Agreement needed to propose a label
              <input
                class="input w-32"
                type="number"
                step="0.05"
                min={MIN_AGREEMENT_RANGE.min}
                max={MIN_AGREEMENT_RANGE.max}
                placeholder={ctl.preview ? String(ctl.preview.request.min_agreement) : ''}
                value={ctl.minAgreement ?? ''}
                oninput={(e) => (ctl.minAgreement = numberOrNull(e))}
                data-testid="assist-min-agreement"
              />
              <span class="text-fg-3"
                >Share of the answers that must name the same class. Empty uses the
                server's default.</span
              >
            </label>
          {/if}
          {#if ctl.actions.includes('segment')}
            <div
              class="flex flex-col gap-1 pl-1 text-meta text-fg-2"
              data-testid="assist-segment-verify"
            >
              <span class="text-fg-2">New boxes from the segmenter</span>
              <label class="flex items-center gap-2">
                <input
                  type="radio"
                  name="assist-segment-verify"
                  checked={ctl.segmentVerify === 'vlm'}
                  onchange={() => (ctl.segmentVerify = 'vlm')}
                  data-testid="assist-segment-verify-vlm"
                />
                Verify with the project's active VLM in the region worker
              </label>
              <label class="flex items-center gap-2">
                <input
                  type="radio"
                  name="assist-segment-verify"
                  checked={ctl.segmentVerify === 'none'}
                  onchange={() => (ctl.segmentVerify = 'none')}
                  data-testid="assist-segment-verify-none"
                />
                Accept without a VLM{ctl.actions.includes('verify')
                  ? ' (this run verifies them with its own endpoint after)'
                  : ''}
              </label>
            </div>
          {/if}
        </fieldset>

        <fieldset class="flex flex-col gap-2 text-body">
          <legend class="mb-1 text-body font-medium text-fg-1">Which VLM</legend>
          <div class="flex flex-wrap items-center gap-3 text-meta">
            <VlmRunPicker
              vlm={ctl.vlm}
              acknowledgeExternal={ctl.acknowledgeExternal}
              defaultLabel="Project's active endpoint"
              onchange={(next) => {
                ctl.vlm = next.vlm;
                ctl.acknowledgeExternal = next.acknowledgeExternal;
              }}
            />
            {#if packAvailable}
              <label class="flex items-center gap-1.5">
                <span class="text-fg-3">Prompt pack</span>
                <select
                  class="select-sm"
                  value={ctl.promptPack ?? ''}
                  onchange={(e) => {
                    const v = (e.currentTarget as HTMLSelectElement).value;
                    ctl.promptPack = v === '' ? null : v;
                  }}
                  data-testid="assist-pack"
                >
                  <option value="">Project's default</option>
                  {#each packs as p (p.id)}
                    <option value={p.id}>{p.label}</option>
                  {/each}
                </select>
              </label>
            {/if}
          </div>
        </fieldset>

        <AssistPlanSummary
          plan={ctl.preview}
          planning={ctl.planning}
          error={ctl.planError}
          errorCode={ctl.planErrorCode}
        />
      {:else}
        <AssistRunProgress run={ctl.run} />
        {#if ctl.run.result?.proposal}
          <AssistProposalCard run={ctl.run} />
        {/if}
      {/if}

      {#if ctl.startError}
        <p class="text-meta text-danger-text" data-testid="assist-start-error">
          {ctl.startError}
        </p>
      {/if}
    </div>

    <footer class="flex flex-wrap justify-end gap-2 border-t border-line px-5 py-3">
      {#if live}
        <button
          type="button"
          class="btn"
          data-testid="assist-cancel"
          onclick={() => void ctl.cancelRun()}
        >
          Cancel run
        </button>
        <button type="button" class="btn" data-testid="assist-close" onclick={close}>
          Close (the run continues)
        </button>
      {:else}
        <button type="button" class="btn" data-testid="assist-close" onclick={close}>
          {ctl.run ? 'Close' : 'Cancel'}
        </button>
        {#if !ctl.run}
          <button
            type="button"
            class="btn btn-primary"
            data-testid="assist-start"
            disabled={!ctl.canStart}
            onclick={() => void ctl.start()}
          >
            Start run
          </button>
        {/if}
      {/if}
    </footer>
  </Modal>
{/if}
