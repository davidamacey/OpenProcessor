<script lang="ts">
  /**
   * Above Gallery, Clusters and Detections: when the served next step is the detector-class
   * adoption (items that carry only the detector's guess), one line says so and opens the
   * existing Adopt dialog. The sentence is the step's served detail; nothing is counted here.
   */
  import AdoptDetectorDialog from '$components/detector/AdoptDetectorDialog.svelte';
  import { nextStepsStore } from '$stores/nextSteps.svelte';

  let open = $state(false);
  const step = $derived(
    nextStepsStore.next?.action === 'adopt_detector_classes' ? nextStepsStore.next : null,
  );
</script>

{#if step}
  <div
    class="flex flex-wrap items-center gap-3 border-b border-warn/30 bg-warn/10 px-4 py-2 text-body text-warn-text"
    data-testid="adopt-banner"
  >
    <span>{step.detail}</span>
    <button
      type="button"
      class="btn btn-sm"
      data-testid="adopt-banner-open"
      onclick={() => (open = true)}
    >
      Adopt detector classes…
    </button>
  </div>
{/if}
{#if open}
  <AdoptDetectorDialog
    onclose={() => (open = false)}
    ondone={() => void nextStepsStore.refresh()}
  />
{/if}
