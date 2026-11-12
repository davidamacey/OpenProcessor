<script lang="ts">
  /**
   * The confirm step of a space write: restart or backfill a build, promote, retire, or cancel a
   * running build. A refusal stays in the dialog as the server worded it, with its code. `force` is
   * offered only after the server refused a promote with `embedding_evaluation_missing`, and waives
   * only that.
   */
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import EmbeddingEvaluation from '$components/embeddingSpaces/EmbeddingEvaluation.svelte';
  import type {
    EmbeddingSpacesState,
    SpaceAction,
  } from '$lib/embeddingSpaces/embeddingSpacesController.svelte';
  import {
    EMBEDDING_BUILD_MODES,
    type EmbeddingBuildMode,
    type EmbeddingSpaceView,
  } from '$lib/types_embeddingSpaces';

  interface Props {
    state: EmbeddingSpacesState;
    action: SpaceAction;
    space: EmbeddingSpaceView;
    ondone: (action: SpaceAction, space: EmbeddingSpaceView) => void;
    oncancel: () => void;
  }

  let { state: spaces, action, space, ondone, oncancel }: Props = $props();

  let mode = $state<EmbeddingBuildMode>('backfill');
  let batch = $state('');
  let force = $state(false);
  let abandon = $state(false);

  const title = $derived(
    {
      build: `Restart or backfill ${space.space_key}`,
      promote: `Promote ${space.space_key}`,
      retire: `Retire ${space.space_key}`,
      cancel: `Cancel the build of ${space.space_key}`,
    }[action],
  );
  const label = $derived(
    { build: 'Start', promote: 'Promote', retire: 'Retire', cancel: 'Cancel build' }[
      action
    ],
  );
  const offerForce = $derived(
    action === 'promote' && spaces.refusal?.code === 'embedding_evaluation_missing',
  );

  $effect(() => {
    if (action === 'promote') void spaces.loadEvaluation(space);
  });

  async function run(): Promise<void> {
    const ok =
      action === 'build'
        ? await spaces.build(space, mode, batch.trim() === '' ? null : Number(batch))
        : action === 'promote'
          ? await spaces.promote(space, force)
          : action === 'retire'
            ? await spaces.retire(space)
            : await spaces.cancel(space, abandon);
    if (ok) ondone(action, space);
  }
</script>

<ConfirmDialog
  {title}
  confirmLabel={force && action === 'promote' ? 'Promote without an evaluation' : label}
  danger={action === 'retire' || (action === 'cancel' && abandon) || force}
  busy={spaces.busy}
  onconfirm={() => void run()}
  oncancel={() => {
    spaces.clearRefusal();
    oncancel();
  }}
>
  {#if action === 'build'}
    <p class="text-xs text-fg-2">
      Restart a failed or cancelled build, or backfill the documents a ready space lacks.
    </p>
    <label class="flex flex-col gap-1 text-xs text-fg-2">
      Mode
      <select class="input input-sm" bind:value={mode} data-testid="build-mode">
        {#each EMBEDDING_BUILD_MODES as m (m)}<option value={m}>{m}</option>{/each}
      </select>
    </label>
    <label class="flex flex-col gap-1 text-xs text-fg-2">
      Batch size (optional; the server's default otherwise)
      <input
        class="input input-sm"
        inputmode="numeric"
        bind:value={batch}
        data-testid="build-batch"
      />
    </label>
  {:else if action === 'promote'}
    <p class="text-xs text-fg-2">
      Make this the primary {space.target} space. The server refuses until the build is complete
      and an evaluation against the current primary exists.
    </p>
    <EmbeddingEvaluation
      evaluation={spaces.evaluationOf(space)}
      error={spaces.evaluationErrorOf(space)}
    />
  {:else if action === 'retire'}
    <p class="text-xs text-fg-2">
      Retire this space and queue the run that clears its vectors. The primary space
      cannot be retired.
    </p>
  {:else}
    <label class="flex items-center gap-2 text-xs text-fg-2">
      <input type="checkbox" bind:checked={abandon} data-testid="cancel-abandon" />
      Also abandon the build: mark the space failed and clear its partial vectors
    </label>
    <p class="text-xs text-fg-3">
      Without it the space stays building and can be restarted as a backfill.
    </p>
  {/if}
  {#if spaces.refusal}
    <p
      class="text-danger-text"
      data-testid="space-refusal"
      data-code={spaces.refusal.code}
    >
      {spaces.refusal.message}
      <code class="ml-1 text-caption opacity-70">{spaces.refusal.code}</code>
    </p>
  {/if}
  {#if offerForce}
    <label class="flex items-center gap-2 text-xs text-warn-text">
      <input type="checkbox" bind:checked={force} data-testid="promote-force" />
      Promote without an evaluation (waives only the missing evaluation)
    </label>
  {/if}
</ConfirmDialog>
