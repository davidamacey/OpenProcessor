<!--
  One operation's detail and its undo. Undo is two steps: the served dry run
  is shown first (what would be restored and what would be kept, with the
  conflicts), and only then does Confirm write.
-->
<script lang="ts">
  import Modal from '$lib/components/Modal.svelte';
  import { formatCount } from '$lib/format/count';
  import { formatTimestamp } from '$lib/format/date';
  import { humanizeId } from '$lib/humanizeId';
  import type { LabelOpsController } from '$lib/labelOps/labelOpsController.svelte';
  import type { LabelOp, LabelOpUndoResponse } from '$lib/types_labelOps';

  interface Props {
    ctl: LabelOpsController;
    op: LabelOp;
  }
  let { ctl, op }: Props = $props();

  const busy = $derived(ctl.phase === 'dry' || ctl.phase === 'writing');

  const OUTCOMES: [keyof LabelOpUndoResponse, string][] = [
    ['restored', 'Restored'],
    ['already_undone', 'Already undone'],
    ['kept_changed', 'Kept: changed since'],
    ['kept_holdout', 'Kept: test holdout'],
    ['kept_stale_class', 'Kept: restore class renamed'],
    ['conflicts', 'Conflicts'],
  ];

  const facts = $derived<[string, string][]>([
    ['Kind', humanizeId(op.kind)],
    ['Recorded', formatTimestamp(op.at)],
    ['Finished', formatTimestamp(op.finished_at)],
    ['Status', humanizeId(op.status)],
    ['Class', op.class_name ?? '—'],
    ['Label source', op.label_source ? humanizeId(op.label_source) : '—'],
    ['Target', humanizeId(op.target)],
    ['Selected', formatCount(op.selected)],
    ['Items written', formatCount(op.count)],
    ['Conflicts when written', formatCount(op.conflicts)],
    ['Still carrying the label', formatCount(op.current)],
  ]);
</script>

<Modal
  label="Label operation"
  onclose={busy ? undefined : () => ctl.close()}
  panelClass="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
  data-testid="label-op-dialog"
>
  <h2 class="text-lg font-semibold">
    {op.kind === 'label' ? 'Label' : 'Move'} operation
  </h2>
  <p class="break-all font-mono text-xs text-zinc-500">{op.op_id}</p>

  <dl class="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-sm" data-testid="label-op-facts">
    {#each facts as [k, v] (k)}
      <dt class="text-zinc-400">{k}</dt>
      <dd class="font-mono text-zinc-200">{v}</dd>
    {/each}
  </dl>

  {#if op.last_undo}
    <section class="mt-4 text-sm" data-testid="label-op-last-undo">
      <h3 class="font-medium">Last undo, {formatTimestamp(op.last_undo.at)}</h3>
      <ul class="mt-1 text-xs text-zinc-300">
        <li>Restored: <span class="font-mono">{op.last_undo.restored}</span></li>
        <li>
          Already undone: <span class="font-mono">{op.last_undo.already_undone}</span>
        </li>
        <li>
          Kept, changed since: <span class="font-mono">{op.last_undo.kept_changed}</span>
        </li>
        <li>
          Kept, test holdout: <span class="font-mono">{op.last_undo.kept_holdout}</span>
        </li>
        <li>
          Kept, restore class renamed:
          <span class="font-mono">{op.last_undo.kept_stale_class}</span>
        </li>
        <li>Conflicts: <span class="font-mono">{op.last_undo.conflicts}</span></li>
      </ul>
    </section>
  {/if}

  {#if ctl.undoError}
    <p class="mt-4 text-sm text-red-300" role="alert" data-testid="label-op-undo-error">
      {ctl.undoError}
    </p>
  {/if}

  {#if ctl.result && (ctl.phase === 'ready' || ctl.phase === 'done')}
    {@const r = ctl.result}
    <section
      class="mt-4 rounded border border-zinc-700 p-3 text-sm"
      data-testid="label-op-result"
    >
      <h3 class="font-medium">
        {r.dry_run ? 'Dry run: nothing has been written' : 'Undo finished'}
      </h3>
      <p class="mt-1 text-xs text-zinc-400">
        Matched <span class="font-mono" data-testid="label-op-matched">{r.matched}</span> items.
      </p>
      <ul class="mt-2 flex flex-col gap-0.5 text-xs">
        {#each OUTCOMES as [key, label] (key)}
          <li
            class="flex justify-between gap-3 text-zinc-300"
            data-testid={`label-op-${key}`}
          >
            <span>{label}</span><span class="font-mono">{r[key]}</span>
          </li>
        {/each}
      </ul>
      {#if r.conflict_ids.length}
        <div class="mt-2 text-xs" data-testid="label-op-conflict-ids">
          <p class="text-amber-300">Conflicting items:</p>
          <ul class="max-h-24 overflow-y-auto break-all font-mono text-zinc-400">
            {#each r.conflict_ids as id (id)}
              <li>{id}</li>
            {/each}
          </ul>
        </div>
      {/if}
    </section>
  {/if}

  <div class="mt-5 flex flex-wrap justify-end gap-2">
    {#if ctl.phase === 'ready'}
      <button type="button" class="btn" onclick={() => ctl.cancelUndo()}>Back</button>
      <button
        type="button"
        class="btn btn-danger"
        data-testid="label-op-confirm"
        onclick={() => void ctl.confirmUndo()}>Confirm undo</button
      >
    {:else if op.undoable && ctl.phase !== 'done'}
      <button
        type="button"
        class="btn btn-primary"
        disabled={busy}
        data-testid="label-op-undo"
        onclick={() => void ctl.startUndo()}
        >{ctl.phase === 'dry' ? 'Checking…' : 'Undo…'}</button
      >
    {/if}
    {#if ctl.phase === 'writing'}
      <span class="self-center text-xs text-zinc-400" role="status">Undoing…</span>
    {/if}
    <button type="button" class="btn" disabled={busy} onclick={() => ctl.close()}
      >Close</button
    >
  </div>
</Modal>
