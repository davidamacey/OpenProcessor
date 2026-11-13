<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  import Popover from '$components/ui/Popover.svelte';
  /**
   * The shared item-filter bar (OpenProcessor v0.4.0): class (by name) and
   * not-class, confidence and area bands, largest-N per image, origin,
   * embedding and review state, and, in the "Matching items" view, an
   * open-vocabulary set and prompt. Every control is a `ServedFilterField`
   * drawn from a spec; a route's served `filter_specs` entry for a param
   * replaces the local spec. A control the route does not honour is absent
   * (`visible`), never disabled. The rarely used controls sit in a "More filters" popover. Nothing is range-checked here: a malformed
   * band is the server's 400, shown under the bar through `error`.
   */
  import ServedFilterField from './ServedFilterField.svelte';
  import { resolveControls } from '$lib/itemFilter/itemFilterControls';
  import type { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';
  import type { ReviewFilterSpec } from '$lib/api_review';

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
    rootClass = 'flex min-w-0 flex-wrap items-center gap-3 text-meta',
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
    'taken_from',
    'class_name',
    'exclude_class_name',
    'open_vocab_set',
    'source_prompt',
    'geo_bbox',
    'geo_polygon',
    'has_location',
    'viz_box',
    'import_id',
    'audit_outcome',
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
    <!-- A popover over the view, so opening it moves neither the grid nor the map. -->
    <div class="shrink-0" data-testid="item-filter-more">
      <Popover
        label="More filters"
        panelClass="w-[36rem] max-h-[70vh] overflow-y-auto p-3"
      >
        {#snippet trigger({ open, attrs })}
          <button
            type="button"
            class="chip {moreActive > 0 || open ? 'chip-accent' : 'text-fg-2'}"
            data-testid="item-filter-more-toggle"
            {...attrs}
          >
            More filters{moreActive > 0 ? ` (${moreActive})` : ''}
          </button>
        {/snippet}
        <div class="flex flex-wrap items-center gap-3 text-meta">
          {#each moreControls as spec (spec.param)}
            <ServedFilterField {spec} value={filter.valueOf(spec.param)} onchange={set} />
          {/each}
        </div>
      </Popover>
    </div>
  {/if}

  {#each chips as chip, i (`${chip.param}:${i}`)}
    <button
      type="button"
      class="chip border-accent/60 bg-accent/15 text-accent-text"
      data-testid="item-filter-chip"
      title="Remove this filter"
      onclick={() => {
        chip.clear();
        onchange?.();
      }}
    >
      {chip.label}<Icon name="x" size={12} class="ml-1 inline-block" />
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
    class="mt-1 w-full text-meta text-danger-text"
    role="alert"
    data-testid="item-filter-error"
  >
    {error}
  </p>
{/if}
