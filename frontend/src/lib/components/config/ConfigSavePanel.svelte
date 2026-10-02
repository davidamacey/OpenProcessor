<script lang="ts">
  /**
   * The editor's save column (§7.6 items 1, 2 and 4), shared by the
   * prompt-pack and region-profile editors: the served report's counts,
   * Save / Discard, the served `revision_conflict` with "Reload" and "Keep
   * my edits", the "changed on the server" notice, and Activate. Saving
   * writes a new revision and never changes what is active.
   */
  import type { Snippet } from 'svelte';
  import type { ConfigEditorView } from '$lib/config/configEditor.svelte';
  import type { ValidationReport } from '$lib/types_config';

  interface Props {
    ed: ConfigEditorView;
    /** The report on screen (a viewed revision's, else the live one). */
    report: ValidationReport | null;
    /** "pack" / "profile". */
    noun: string;
    onsave: () => void;
    onactivate: () => void;
    /** Extra controls between Save and Activate. */
    extra?: Snippet;
  }

  let { ed, report, noun, onsave, onactivate, extra }: Props = $props();

  const counts = $derived({
    errors: report?.errors.length ?? 0,
    warnings: (report?.warnings ?? []).filter((i) => i.severity === 'warning').length,
    info: (report?.warnings ?? []).filter((i) => i.severity === 'info').length,
  });
</script>

<section class="surface flex flex-col gap-2 p-4 text-sm" aria-label="Save">
  <div class="flex flex-wrap items-center gap-2 text-xs" data-testid="report-counts">
    <span class={counts.errors > 0 ? 'text-red-300' : 'text-zinc-400'}
      >{counts.errors} error{counts.errors === 1 ? '' : 's'}</span
    >
    <span class={counts.warnings > 0 ? 'text-amber-300' : 'text-zinc-400'}
      >{counts.warnings} warning{counts.warnings === 1 ? '' : 's'}</span
    >
    <span class="text-zinc-400">{counts.info} note{counts.info === 1 ? '' : 's'}</span>
    {#if ed.validating}<span class="text-zinc-500">checking…</span>{/if}
  </div>
  {#if ed.validateError}
    <p class="text-xs text-red-300">Could not check the draft: {ed.validateError}</p>
  {/if}
  {#if ed.editable}
    <div class="flex flex-wrap gap-2">
      <button
        type="button"
        class="btn btn-primary btn-sm"
        disabled={!ed.canSave}
        onclick={onsave}
        data-testid="config-save">{ed.saving ? 'Saving…' : 'Save'}</button
      >
      <button
        type="button"
        class="btn btn-sm"
        disabled={!ed.dirty || ed.saving}
        onclick={() => void ed.reloadLatest()}>Discard edits</button
      >
    </div>
    <p class="text-xs text-zinc-500">
      {ed.dirty ? 'Unsaved edits.' : 'No unsaved edits.'} Saving writes a new revision; it doesn't
      change the active {noun}.
    </p>
  {/if}
  {#if ed.saveError}
    <p class="text-xs text-red-300" data-testid="save-error">{ed.saveError}</p>
  {/if}
  {#if ed.conflict}
    <div
      class="space-y-2 rounded border border-amber-500/40 bg-amber-500/10 p-2 text-xs text-amber-100"
      data-testid="save-conflict"
    >
      <p>{ed.conflict.message}</p>
      <div class="flex flex-wrap gap-2">
        <button type="button" class="btn btn-sm" onclick={() => void ed.reloadLatest()}
          >Reload{ed.conflict.currentRevision != null
            ? ` revision ${ed.conflict.currentRevision}`
            : ''} (discard my edits)</button
        >
        {#if ed.conflict.currentRevision != null}
          <button
            type="button"
            class="btn btn-sm"
            onclick={() => ed.keepMine()}
            data-testid="keep-mine">Keep my edits</button
          >
        {/if}
      </div>
    </div>
  {/if}
  {#if ed.remoteChanged}
    <div
      class="space-y-2 rounded border border-sky-500/40 bg-sky-500/10 p-2 text-xs text-sky-100"
      data-testid="remote-changed"
    >
      <p>This {noun} changed on the server while you were editing.</p>
      <button type="button" class="btn btn-sm" onclick={() => void ed.reloadLatest()}
        >Load it (discard my edits)</button
      >
    </div>
  {/if}
  {@render extra?.()}
  {#if ed.doc}
    <div class="border-t border-zinc-800 pt-2">
      <button
        type="button"
        class="btn btn-sm"
        onclick={onactivate}
        data-testid="config-activate"
        >Activate{ed.doc.revision != null ? ` revision ${ed.doc.revision}` : ''}</button
      >
    </div>
  {/if}
</section>
