<script lang="ts">
  /**
   * Primary-subject scope tri-state, shared by /clusters, /clusters/[id]
   * and /review. Maps to the API's `max_rank` filter — "the biggest
   * vehicle(s) in frame", which is what the business sorts on.
   *
   * Option labels differ per page (the review tabs default to top-2
   * server-side, so 0 reads as "Top 2" there), hence the `labels` prop.
   */
  interface Props {
    value: 0 | 1 | 2;
    /** Labels for scope 0 / 1 / 2. */
    labels?: [string, string, string];
    /** Optional tooltips for scope 0 / 1 / 2 — e.g. that option 0 is a
     *  tab's served default rather than "everything" (visual audit R11). */
    titles?: [string?, string?, string?];
    /** Prefix text; omit for no prefix. */
    label?: string;
    labelClass?: string;
    /** Tighter vertical padding + lighter inactive fill (cluster detail). */
    dense?: boolean;
  }

  let {
    value = $bindable(),
    labels = ['All', 'Largest', '+2nd'],
    titles = [],
    label,
    labelClass = 'text-zinc-400',
    dense = false,
  }: Props = $props();

  const options = $derived(labels.map((l, i) => ({ v: i as 0 | 1 | 2, l })));
</script>

<div class="flex shrink-0 items-center gap-1.5">
  {#if label}
    <span class={labelClass}>{label}</span>
  {/if}
  <div class="inline-flex overflow-hidden rounded border border-zinc-700">
    {#each options as opt (opt.v)}
      <button
        type="button"
        class="chip rounded-none border-0 {value === opt.v
          ? 'bg-blue-600 text-white'
          : `${dense ? 'bg-zinc-800' : 'bg-zinc-900'} text-zinc-300 hover:bg-zinc-700`}"
        title={titles[opt.v]}
        onclick={() => (value = opt.v)}
      >
        {opt.l}
      </button>
    {/each}
  </div>
</div>
