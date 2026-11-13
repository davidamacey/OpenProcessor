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
  import { opStatusLine } from '$lib/labelOps/opStatusLine';
  import type {
    SelectionActionController,
    SelectionActionId,
  } from '$lib/itemFilter/selectionActionController.svelte';

  interface Props {
    controller: SelectionActionController;
    /** Called after the write landed, so the host can reload its view. */
    onapplied?: () => void;
    /** Who the action acts on, in words ("the 3 selected items"), shown above the count. */
    subject?: string;
    /** The whole-cluster label offers to open the next cluster once it is written. */
    openNext?: { checked: boolean; onchange: (checked: boolean) => void };
    /** A caution the host wants read before the write (a mixed cluster). */
    warning?: string | null;
    /** The detector classes of a mixed cluster: offered first, because labelling one at a time
     *  is cleaner than labelling the whole. Choosing one closes this dialog and narrows the view. */
    split?: {
      classes: { value: string; count: number }[];
      onsplit: (value: string) => void;
    };
  }

  let { controller: c, onapplied, subject, openNext, warning, split }: Props = $props();

  const TITLES: Record<SelectionActionId, string> = {
    exclude: 'Ignore matching items',
    unexclude: 'Restore matching items',
    label: 'Label matching items',
    move: 'Move matching items to a cluster',
  };

  const VERBS: Partial<Record<SelectionActionId, string>> = {
    label: 'Labeling',
    move: 'Moving',
  };
  const statusLine = $derived(
    c.action && c.loading ? opStatusLine(c.job, VERBS[c.action] ?? 'Working') : null,
  );

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

{#if c.action && !c.detached}
  <ConfirmDialog
    title={TITLES[c.action]}
    confirmLabel="Apply"
    danger={c.action === 'exclude'}
    busy={c.loading && c.selected != null}
    busyLabel={statusLine ?? 'Working…'}
    confirmDisabled={!c.canConfirm}
    focusConfirm={!(warning && c.action === 'label')}
    onconfirm={confirm}
    oncancel={() => (c.job ? c.runInBackground() : c.cancel())}
  >
    {#snippet actions()}
      {#if c.loading && c.job}
        <button
          type="button"
          class="btn"
          data-testid="selection-background"
          onclick={() => c.runInBackground()}
        >
          Run in background
        </button>
      {/if}
    {/snippet}
    {#if warning && c.action === 'label'}
      <div
        class="flex flex-col gap-2 rounded border border-warn/40 bg-warn/10 px-3 py-2 text-warn-text"
        role="note"
        data-testid="selection-warning"
      >
        <p>{warning}</p>
        {#if split && split.classes.length > 1}
          <div class="flex flex-wrap items-center gap-1.5" data-testid="selection-split">
            <span class="text-fg-1">Split by detector class:</span>
            {#each split.classes as f (f.value)}
              <button
                type="button"
                class="btn-sm border border-line-strong bg-surface-1 text-meta text-fg-1 hover:border-accent/40"
                data-testid="selection-split-class"
                onclick={() => {
                  c.cancel();
                  split.onsplit(f.value);
                }}
              >
                {f.value}
                <span class="ml-1 tabular-nums text-fg-2">{f.count.toLocaleString()}</span
                >
              </button>
            {/each}
          </div>
        {/if}
        <p class="text-meta text-fg-2">Or choose a class below to label all of them.</p>
      </div>
    {/if}

    {#if c.action === 'label'}
      <label class="flex items-center gap-2">
        <span class="text-fg-2">Class</span>
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
        <span class="text-fg-2">Cluster</span>
        <select
          class="select-sm"
          data-testid="selection-cluster"
          value={c.clusterUid ?? ''}
          onchange={(e) => {
            c.clusterUid = e.currentTarget.value || null;
            void c.refresh();
          }}
        >
          <option value="">choose…</option>
          {#each c.clusterOptions as cl (cl.uid)}
            <option value={cl.uid}>
              {cl.uid.slice(0, 8)} · {cl.dominant_class_name ?? 'unlabeled'} · {cl.size.toLocaleString()}
            </option>
          {/each}
        </select>
      </label>
    {/if}

    {#if subject}
      <p class="text-fg-2" data-testid="selection-subject">Acting on {subject}.</p>
    {/if}
    <p data-testid="selection-count">
      {#if c.loading && c.selected == null}
        Counting…
      {:else if c.selected != null}
        This will change <strong class="text-fg-1">{c.selected}</strong>
        {c.selected === 1 ? 'item' : 'items'}.
      {:else if !c.error}
        Pick the missing choice to see how many items this changes.
      {/if}
    </p>

    <div class="flex flex-wrap items-center gap-3 text-meta">
      <label class="flex items-center gap-1.5">
        <span class="text-fg-2">Limit to</span>
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
        <span class="text-fg-2">Sample</span>
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
          <span class="text-fg-2">Seed</span>
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

    {#if openNext && c.action === 'label'}
      <label class="flex items-center gap-2 text-meta text-fg-2">
        <input
          type="checkbox"
          checked={openNext.checked}
          data-testid="open-next-after-label"
          onchange={(e) => openNext.onchange(e.currentTarget.checked)}
        />
        Open the next cluster after labelling
      </label>
    {/if}

    {#if statusLine}
      <p class="text-fg-1" data-testid="selection-job">
        {statusLine}.
        {#if c.job?.status === 'queued'}
          <span class="text-fg-2"
            >You can close this: the change runs in the background and shows in Label
            history.</span
          >
        {/if}
      </p>
    {/if}

    {#if c.error}
      <p class="text-danger-text" role="alert" data-testid="selection-error">{c.error}</p>
    {/if}
  </ConfirmDialog>
{/if}
