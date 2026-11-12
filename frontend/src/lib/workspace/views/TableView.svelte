<script lang="ts">
  /**
   * Table: detections or images as rows with the served columns and cells. The server
   * sorts, pages and formats; this view sends the choice and shows what comes back.
   * Rows, columns and the sort live in the page URL.
   */
  import { onMount, untrack } from 'svelte';
  import DataTable from '$components/table/DataTable.svelte';
  import ColumnPicker from '$components/table/ColumnPicker.svelte';
  import HiddenItemsNote from '$components/itemFilter/HiddenItemsNote.svelte';
  import BulkBar from '$components/workspace/BulkBar.svelte';
  import ViewState from '$components/state/ViewState.svelte';
  import { infiniteScroll } from '$lib/actions/infiniteScroll';
  import {
    createTableController,
    rowKey,
    type TableUrlState,
  } from '$lib/table/tableController.svelte';
  import type { TableRow } from '$lib/types_table';
  import { NO_SELECTION } from '$lib/workspace/selection';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
  import { writeWorkspaceUrl } from '$lib/workspace/workspaceUrl';
  import { keyboardStore } from '$stores/keyboard.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  // `ws` is fixed for the life of the view.
  // svelte-ignore state_referenced_locally
  const ctl = createTableController({
    initial: {
      rows: ws.url.rows,
      columns: ws.url.columns ?? [],
      sort: ws.url.sort.table ?? null,
    },
    itemFilter: () => ws.itemQuery(),
    writeUrl: (s: TableUrlState) =>
      ws.patchUrl((params) =>
        writeWorkspaceUrl(params, {
          rows: s.rows,
          columns: s.columns.length > 0 ? s.columns : null,
          sort: { table: s.sort ?? undefined },
        }),
      ),
  });

  let anchor: string | null = null;

  const queryKey = $derived(JSON.stringify(ws.itemQuery()));
  $effect(() => {
    void queryKey;
    untrack(() => void ctl.load());
  });

  // The selection feeds the bulk bar: crop ids on detection rows, image ids on image rows.
  $effect(() => {
    const ids = [...ctl.selectedIds];
    if (ids.length === 0) {
      if (ws.selection.kind === 'items' || ws.selection.kind === 'images') {
        ws.select(NO_SELECTION);
      }
      return;
    }
    ws.select(
      ctl.rowsKind === 'images'
        ? { kind: 'images', imageIds: ids }
        : { kind: 'items', cropIds: ids },
    );
  });

  $effect(() =>
    ws.registerPage({
      get kind() {
        return ctl.rowsKind === 'images' ? 'images' : 'items';
      },
      ids: () => ctl.rows.map(rowKey),
      total: () => ctl.total,
      apply: (keys) => ctl.select(keys, true),
    }),
  );

  $effect(() => {
    if (!ctl.loading) ws.matching = { total: ctl.total, hidden: ctl.hidden };
    return () => {
      ws.matching = null;
    };
  });

  const status = $derived(
    ctl.error
      ? 'error'
      : ctl.loading && ctl.rows.length === 0
        ? 'loading'
        : ctl.rows.length === 0
          ? 'empty'
          : 'ready',
  );

  function select(key: string, range: boolean): void {
    if (range && anchor !== null) {
      const keys = ctl.rows.map(rowKey);
      const a = keys.indexOf(anchor);
      const b = keys.indexOf(key);
      if (a >= 0 && b >= 0) {
        ctl.select(keys.slice(Math.min(a, b), Math.max(a, b) + 1), true);
        ctl.focus(key);
        return;
      }
    }
    ctl.select([key], !ctl.selectedIds.has(key));
    anchor = key;
    ctl.focus(key);
  }

  function open(row: TableRow): void {
    if (ctl.rowsKind === 'images') {
      ws.setFocus({ kind: 'image', id: rowKey(row) });
      ws.setView('review');
    } else {
      // Review opens on this item, and Esc returns to this table.
      ws.openReview(rowKey(row));
    }
  }

  function move(by: 1 | -1, extend: boolean): void {
    const keys = ctl.rows.map(rowKey);
    const at = ctl.focusedId === null ? -1 : keys.indexOf(ctl.focusedId);
    const to = keys[Math.min(Math.max(at + by, 0), keys.length - 1)];
    if (to === undefined) return;
    if (extend) {
      if (ctl.focusedId !== null) ctl.select([ctl.focusedId], true);
      ctl.select([to], true);
    }
    ctl.focus(to);
    document
      .querySelector(`[data-row-key="${CSS.escape(to)}"]`)
      ?.scrollIntoView({ block: 'nearest' });
  }

  onMount(() => {
    const scope = 'table';
    const offs = [
      keyboardStore.registerAction('table.prev_row', () => move(-1, false), scope),
      keyboardStore.registerAction('table.next_row', () => move(1, false), scope),
      keyboardStore.registerAction('table.extend_prev', () => move(-1, true), scope),
      keyboardStore.registerAction('table.extend_next', () => move(1, true), scope),
      // Registered ahead of the page's card handler: this view's own scope is searched first.
      keyboardStore.registerAction(
        'workspace.select_toggle',
        () => {
          if (ctl.focusedId === null) return false;
          select(ctl.focusedId, false);
        },
        scope,
      ),
      keyboardStore.registerAction(
        'workspace.open_review',
        () => {
          const row = ctl.rows.find((r) => rowKey(r) === ctl.focusedId);
          if (!row) return false;
          open(row);
        },
        scope,
      ),
    ];
    return () => offs.forEach((off) => off());
  });
</script>

<div class="flex min-h-0 flex-1 flex-col" data-testid="table-view">
  <h1 class="sr-only">Table</h1>
  <BulkBar {ws} onapplied={() => void ctl.reload()} />
  <div class="flex items-center gap-3 px-4 pt-3 text-xs text-fg-2">
    <div role="group" aria-label="Rows" class="flex gap-1">
      {#each ['detections', 'images'] as const as kind (kind)}
        <button
          type="button"
          class="rounded border px-2 py-1 {ctl.rowsKind === kind
            ? 'border-info text-fg-1'
            : 'border-line-strong'}"
          aria-pressed={ctl.rowsKind === kind}
          data-testid="table-rows-{kind}"
          onclick={() => void ctl.setRows(kind)}
          >{kind === 'images' ? 'Images' : 'Detections'}</button
        >
      {/each}
    </div>
    <p data-testid="table-total">
      <strong class="text-fg-1">{ctl.total.toLocaleString()}</strong> rows
      <HiddenItemsNote hidden={ctl.hidden} />
    </p>
    <div class="ml-auto">
      {#if ctl.vocabulary}
        <ColumnPicker
          vocabulary={ctl.vocabulary.columns}
          chosen={ctl.chosenIds}
          onchange={(ids) => void ctl.setColumns(ids)}
        />
      {/if}
    </div>
  </div>
  <div class="min-h-0 flex-1 overflow-hidden p-4">
    <ViewState
      {status}
      message={ctl.error ?? ctl.emptyMessage ?? 'No rows match this filter.'}
      onretry={() => void ctl.reload()}
    >
      <DataTable
        columns={ctl.columns}
        rows={ctl.rows}
        sortApplied={ctl.sortApplied}
        sortOptions={ctl.sortOptions}
        selectedIds={ctl.selectedIds}
        focusedId={ctl.focusedId}
        onsort={(id) => void ctl.sortBy(id)}
        onselect={select}
        onfocus={(k) => ctl.focus(k)}
        onopen={open}
      />
      <div
        use:infiniteScroll={{
          onload: () => ctl.loadMore(),
          disabled: !ctl.hasMore || ctl.loading || ctl.loadingMore,
        }}
        class="h-1"
        aria-hidden="true"
      ></div>
    </ViewState>
  </div>
</div>
