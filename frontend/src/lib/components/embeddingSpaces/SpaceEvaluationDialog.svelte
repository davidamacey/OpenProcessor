<script lang="ts">
  /**
   * The served comparison of a space with the primary: the evaluation runs of the space, read when
   * the dialog opens. Reads only; promote is its own step.
   */
  import Modal from '$components/Modal.svelte';
  import EmbeddingEvaluation from '$components/embeddingSpaces/EmbeddingEvaluation.svelte';
  import type { EmbeddingSpacesState } from '$lib/embeddingSpaces/embeddingSpacesController.svelte';
  import type { EmbeddingSpaceView } from '$lib/types_embeddingSpaces';

  interface Props {
    state: EmbeddingSpacesState;
    space: EmbeddingSpaceView;
    onclose: () => void;
  }

  let { state: spaces, space, onclose }: Props = $props();

  $effect(() => {
    void spaces.loadEvaluation(space);
  });
</script>

<Modal
  label="Evaluation of {space.space_key}"
  {onclose}
  panelClass="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-lg border border-line bg-surface-0 p-5 shadow-2xl"
>
  <h3 class="mb-3 text-title font-semibold text-fg-1">
    Evaluation of {space.space_key}
  </h3>
  <div class="mb-4" data-testid="space-evaluation-dialog">
    <EmbeddingEvaluation
      evaluation={spaces.evaluationOf(space)}
      error={spaces.evaluationErrorOf(space)}
    />
  </div>
  <div class="flex justify-end">
    <button type="button" class="btn" onclick={onclose}>Close</button>
  </div>
</Modal>
