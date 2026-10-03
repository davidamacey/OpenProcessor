<script lang="ts">
  /**
   * The shared item-filter bar (OpenProcessor v0.4.0): class (by name) and
   * not-class, confidence and area bands, largest-N per image, origin,
   * embedding and review state, and, in the "Matching items" view, an
   * open-vocabulary set and prompt. Every control is a `ServedFilterField`
   * drawn from a spec; a route's served `filter_specs` entry for a param
   * replaces the local spec. A control the route does not honour is absent
   * (`visible`), never disabled. Nothing is range-checked here: a malformed
   * band is the server's 400, shown under the bar through `error`.
   */
  import ServedFilterField from './ServedFilterField.svelte';
  import { resolveControls } from '$lib/itemFilter/itemFilterControls';
  import type { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';
  import type { ReviewFilterSpec } from '$lib/api';

  interface Props {
    state: ItemFilterState;
    /** Whether the route honours a param; default: all. */
    visible?: (param: string) => boolean;
    /** The route's served `filter_specs`, to override local labels/options/bounds. */
    served?: readonly ReviewFilterSpec[];
    showOpenVocab?: boolean;
    /** Params drawn inline; every other control sits in a "More filters"
     *  disclosure so the bar stays one row tall. */
    inline?: readonly string[];
    /** The server's refusal of the current filter (a 400). */
    error?: string | null;
    onchange?: () => void;
    /** The root's classes; a host that already lays its children out in a flex
     *  row passes `contents` so the controls join that row directly. */
    rootClass?: string;
  }

  let {
    state: filter,
    visible = () => true,
    served = [],
    showOpenVocab = false,
    inline = ['class_name', 'open_vocab_set', 'source_prompt'],
    error = null,
    onchange,
    rootClass = 'flex min-w-0 flex-wrap items-center gap-3 text-xs',
  }: Props = $props();

  const controls = $derived(
    resolveControls(served, showOpenVocab).filter((c) => visible(c.param)),
  );
  const inlineControls = $derived(controls.filter((c) => inline.includes(c.param)));
  const moreControls = $derived(controls.filter((c) => !inline.includes(c.param)));
  const moreActive = $derived(
    filter.chips().filter((c) => !inline.includes(c.param) && visible(c.param)).length,
  );
  // Names and the free-text pair have no toggle that shows them being set,
  // so they get a removable chip; enum and number controls show their own value.
  const CHIP_PARAMS = new Set([
    'class_name',
    'exclude_class_name',
    'open_vocab_set',
    'source_prompt',
  ]);
  const chips = $derived(
    filter.chips().filter((c) => CHIP_PARAMS.has(c.param) && visible(c.param)),
  );

  function set(param: string, value: string | string[]): void {
    filter.setValue(param, value);
    onchange?.();
  }
</script>

<div class={rootClass} data-testid="item-filter-bar">
  {#each inlineControls as spec (spec.param)}
    <ServedFilterField {spec} value={filter.valueOf(spec.param)} onchange={set} />
  {/each}

  {#if moreControls.length > 0}
    <details
      class="shrink-0 open:basis-full"
      data-testid="item-filter-more"
      open={moreActive > 0}
    >
      <summary class="chip cursor-pointer list-none text-zinc-300">
        More filters{moreActive > 0 ? ` (${moreActive})` : ''}
      </summary>
      <div class="mt-2 flex flex-wrap items-center gap-3 text-xs">
        {#each moreControls as spec (spec.param)}
          <ServedFilterField {spec} value={filter.valueOf(spec.param)} onchange={set} />
        {/each}
      </div>
    </details>
  {/if}

  {#each chips as chip (chip.label)}
    <button
      type="button"
      class="chip border-blue-500/60 bg-blue-500/15 text-blue-100"
      data-testid="item-filter-chip"
      title="Remove this filter"
      onclick={() => {
        chip.clear();
        onchange?.();
      }}
    >
      {chip.label} ×
    </button>
  {/each}

  {#if !filter.isEmpty}
    <button
      type="button"
      class="btn-sm"
      data-testid="item-filter-clear"
      onclick={() => {
        filter.clear();
        onchange?.();
      }}
    >
      Clear filters
    </button>
  {/if}
</div>
{#if error}
  <p
    class="mt-1 w-full text-xs text-red-300"
    role="alert"
    data-testid="item-filter-error"
  >
    {error}
  </p>
{/if}
