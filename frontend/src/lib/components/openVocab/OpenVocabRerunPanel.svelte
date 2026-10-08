<script lang="ts">
  /**
   * Run the active open-vocabulary set over images already in the pool.
   * Each button opens the reprocess dialog with one request (a dry run
   * first, then the apply behind a confirm): every image, or the images
   * whose pass is in one served status. The status buttons are labelled by
   * the served vocabulary and only offered for a status it lists. Rendered
   * only while a set is active and the backend serves reprocess; the dry
   * run's served detail and any refusal (it names "activate a set first")
   * are the dialog's to show.
   */
  import ReprocessControl from '$components/datasets/ReprocessControl.svelte';
  import { datasetsAvailability } from '$lib/datasets/datasetsAvailability.svelte';
  import { optionLabel } from '$lib/openVocab/hitShapes';
  import type { OpenVocabStatus, ReprocessRequest } from '$lib/types_import';
  import type { OpenVocabVocabulary } from '$lib/types_openVocab';

  interface Props {
    /** A set is active on this project. */
    active: boolean;
    /** The served labels (`GET /open_vocab/schema`); null while unread. */
    vocabulary: OpenVocabVocabulary | null;
  }

  let { active, vocabulary }: Props = $props();

  $effect(() => {
    void datasetsAvailability.init();
  });

  const ALL_IMAGES: ReprocessRequest = {
    targets: { filter: { all_images: true } },
    scopes: ['open_vocab'],
    dry_run: true,
  };

  /** The statuses a re-run makes sense for: not `done`. */
  const RERUNNABLE: OpenVocabStatus[] = ['pending', 'skipped_gate', 'failed'];

  const statusRuns = $derived(
    vocabulary
      ? RERUNNABLE.filter((s) => vocabulary.statuses.some((o) => o.value === s)).map(
          (s) => ({
            status: s,
            label: optionLabel(vocabulary.statuses, s),
            request: {
              targets: { filter: { open_vocab_status: [s] } },
              scopes: ['open_vocab'],
              dry_run: true,
            } satisfies ReprocessRequest,
          }),
        )
      : [],
  );
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
    <div class="flex flex-wrap gap-2">
      <ReprocessControl
        target={{ kind: 'request', request: ALL_IMAGES }}
        buttonLabel="Run on all images…"
      />
      {#each statusRuns as r (r.status)}
        <ReprocessControl
          target={{ kind: 'request', request: r.request }}
          buttonLabel="Re-run: {r.label}…"
        />
      {/each}
    </div>
  </section>
{/if}
