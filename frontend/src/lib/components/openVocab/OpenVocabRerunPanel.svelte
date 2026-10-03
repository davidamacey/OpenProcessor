<script lang="ts">
  /**
   * Run the active open-vocabulary set over images already in the pool.
   * One request through the reprocess dialog (dry run first, then apply
   * behind a confirm): every image, scope `open_vocab`. Rendered only while
   * a set is active and the backend serves reprocess; the dry run's served
   * detail and any refusal (it names "activate a set first") are the
   * dialog's to show.
   */
  import ReprocessControl from '$components/datasets/ReprocessControl.svelte';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import type { ReprocessRequest } from '$lib/types_import';

  interface Props {
    /** A set is active on this project. */
    active: boolean;
  }

  let { active }: Props = $props();

  $effect(() => {
    void datasetsAvailability.init();
  });

  const ALL_IMAGES: ReprocessRequest = {
    targets: { filter: { all_images: true } },
    scopes: ['open_vocab'],
    dry_run: true,
  };
</script>

{#if active && datasetsAvailability.available === true}
  <section
    class="surface flex flex-col gap-2 p-4 text-sm"
    data-testid="open-vocab-rerun"
    aria-label="Run on existing images"
  >
    <h2 class="text-base font-semibold">Run on existing images</h2>
    <p class="text-xs text-zinc-400">
      New images are covered as they arrive. To run the active set on the images already
      in the pool, check what would run first, then confirm.
    </p>
    <div>
      <ReprocessControl
        target={{ kind: 'request', request: ALL_IMAGES }}
        buttonLabel="Run on all images…"
      />
    </div>
  </section>
{/if}
