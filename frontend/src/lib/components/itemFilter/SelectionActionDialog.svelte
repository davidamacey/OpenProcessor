<script lang="ts">
  /**
   * Confirm step for an action on every item a filter matches. The count in
   * the sentence is the server's dry run (`selected`); changing the limit,
   * sample or seed re-runs it. Refusals are shown as the server worded them.
   */
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { SELECTION_SAMPLES } from '$lib/types_itemFilter';
  import { humanizeId } from '$lib/humanizeId';
  import type {
    SelectionActionController,
    SelectionActionId,
  } from '$lib/itemFilter/selectionActionController.svelte';

  interface Props {
    controller: SelectionActionController;
    /** Called after the write landed, so the host can reload its view. */
    onapplied?: () => void;
  }

  let { controller: c, onapplied }: Props = $props();

  const TITLES: Record<SelectionActionId, string> = {
    exclude: 'Ignore matching items',
    unexclude: 'Restore matching items',
    label: 'Label matching items',
    move: 'Move matching items to a cluster',
  };

  const classOptions = $derived(classesStore.classes.filter((cls) => !cls.deprecated));

  function intOrNull(raw: string): number | null {
    if (raw.trim() === '') return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  }

  async function confirm(): Promise<void> {
    await c.confirm();
    if (c.confirmed) {
      c.cancel();
      onapplied?.();
    }
  }
</script>

{#if c.action}
  <ConfirmDialog
    title={TITLES[c.action]}
    confirmLabel="Apply"
    danger={c.action === 'exclude'}
    busy={c.loading && c.selected != null}
    confirmDisabled={!c.canConfirm}
    onconfirm={confirm}
    oncancel={() => c.cancel()}
  >
    {#if c.action === 'label'}
      <label class="flex items-center gap-2">
        <span class="text-zinc-400">Class</span>
        <select
          class="select-sm"
          data-testid="selection-class"
          value={c.classId ?? ''}
          onchange={(e) => {
            c.classId = intOrNull(e.currentTarget.value);
            void c.refresh();
          }}
        >
          <option value="">choose…</option>
          {#each classOptions as cls (cls.id)}
            <option value={cls.id}>{cls.name}</option>
          {/each}
        </select>
      </label>
    {:else if c.action === 'move'}
      <label class="flex items-center gap-2">
        <span class="text-zinc-400">Cluster id</span>
        <input
          type="number"
          class="input-sm w-28"
          data-testid="selection-cluster"
          value={c.clusterId ?? ''}
          onchange={(e) => {
            c.clusterId = intOrNull(e.currentTarget.value);
            void c.refresh();
          }}
        />
      </label>
    {/if}

    <p data-testid="selection-count">
      {#if c.loading && c.selected == null}
        Counting…
      {:else if c.selected != null}
        This will change <strong class="text-zinc-100">{c.selected}</strong>
        {c.selected === 1 ? 'item' : 'items'}.
      {:else if !c.error}
        Pick the missing choice to see how many items this changes.
      {/if}
    </p>

    <div class="flex flex-wrap items-center gap-3 text-xs">
      <label class="flex items-center gap-1.5">
        <span class="text-zinc-400">Limit to</span>
        <input
          type="number"
          min="1"
          class="input-sm w-24"
          placeholder="all"
          data-testid="selection-limit"
          value={c.limit ?? ''}
          onchange={(e) => {
            c.limit = intOrNull(e.currentTarget.value);
            void c.refresh();
          }}
        />
      </label>
      <label class="flex items-center gap-1.5">
        <span class="text-zinc-400">Sample</span>
        <select
          class="select-sm"
          data-testid="selection-sample"
          value={c.sample ?? ''}
          onchange={(e) => {
            const v = e.currentTarget.value;
            c.sample = (SELECTION_SAMPLES as readonly string[]).includes(v)
              ? (v as (typeof SELECTION_SAMPLES)[number])
              : null;
            void c.refresh();
          }}
        >
          <option value="">server order</option>
          {#each SELECTION_SAMPLES as s (s)}
            <option value={s}>{humanizeId(s)}</option>
          {/each}
        </select>
      </label>
      {#if c.sample === 'random'}
        <label class="flex items-center gap-1.5">
          <span class="text-zinc-400">Seed</span>
          <input
            type="number"
            class="input-sm w-24"
            data-testid="selection-seed"
            value={c.seed ?? ''}
            onchange={(e) => {
              c.seed = intOrNull(e.currentTarget.value);
              void c.refresh();
            }}
          />
        </label>
      {/if}
    </div>

    {#if c.error}
      <p class="text-red-300" role="alert" data-testid="selection-error">{c.error}</p>
    {/if}
  </ConfirmDialog>
{/if}
