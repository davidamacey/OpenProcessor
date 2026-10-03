<script lang="ts">
  /**
   * `/settings/open-vocab`: the project's SAM 3 open-vocabulary sets
   * (OpenProcessor v0.4.0). Everything listed is served; `OpenVocabListState`
   * follows it. Absent (one line) until the backend serves the routes.
   */
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import ConfigActivePanel from '$components/config/ConfigActivePanel.svelte';
  import ConfigCloneDialog from '$components/config/ConfigCloneDialog.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import OpenVocabRerunPanel from '$components/openVocab/OpenVocabRerunPanel.svelte';
  import SegmenterNotice from '$components/openVocab/SegmenterNotice.svelte';
  import type { CloneSource } from '$lib/config/configList.svelte';
  import { formatTimestamp } from '$lib/formatDate';
  import { openVocabAvailability } from '$lib/openVocab/openVocabAvailability.svelte';
  import { OPEN_VOCAB_ACTIVE_COPY } from '$lib/openVocab/openVocabCopy';
  import { createOpenVocabList } from '$lib/openVocab/openVocabListController.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { OpenVocabSummary } from '$lib/types_openVocab';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const list = createOpenVocabList();

  $effect(() => {
    if (openVocabAvailability.available !== true) return;
    list.start();
    return () => list.stop();
  });

  let cloneFrom = $state<CloneSource | null>(null);
  let deleting = $state<OpenVocabSummary | null>(null);
  let creating = $state(false);
  let newName = $state('');

  const editorPath = (name: string) =>
    projectHref(`/settings/open-vocab/${encodeURIComponent(name)}`);

  function openClone(from: CloneSource): void {
    list.clearClone();
    cloneFrom = from;
  }

  async function doClone(name: string, description: string): Promise<void> {
    if (!cloneFrom) return;
    const doc = await list.clone(cloneFrom, name, description);
    if (!doc) return;
    cloneFrom = null;
    toastStore.success(`Created ${doc.name}`);
    await goto(resolve(editorPath(doc.name)));
  }

  async function doCreate(): Promise<void> {
    const doc = await list.create(newName);
    if (!doc) return;
    creating = false;
    newName = '';
    toastStore.success(`Created ${doc.name}`);
    await goto(resolve(editorPath(doc.name)));
  }

  async function doDelete(): Promise<void> {
    const row = deleting;
    if (!row) return;
    if (await list.remove(row)) {
      deleting = null;
      toastStore.success(`Deleted ${row.name}`);
    }
  }
</script>

<div class="mx-auto flex max-w-6xl flex-col gap-4 p-6">
  <header class="flex flex-wrap items-baseline gap-3">
    <h1 class="text-2xl font-semibold tracking-tight">Open-vocabulary sets</h1>
    <span class="grow"></span>
    <a
      class="text-xs text-blue-300 hover:underline"
      href={resolve(projectHref('/settings'))}>Back to settings</a
    >
  </header>
  <p class="text-sm text-zinc-400">
    An open-vocabulary set lists things to find by describing them in words: each target
    has a prompt, and optionally the class its hits are stored as. One set is active per
    project. Open a set to edit and test it.
  </p>

  <ConfigGate
    store={openVocabAvailability}
    what="open-vocabulary sets"
    unavailableText="Open-vocabulary sets are not available on this backend."
    testid="open-vocab-unavailable"
  >
    <SegmenterNotice segmenter={list.list?.segmenter ?? null} />

    <ConfigActivePanel
      ctl={list.active}
      copy={OPEN_VOCAB_ACTIVE_COPY}
      onrollback={() => list.rollback()}
      ondeactivate={() => list.deactivate()}
    />

    {#if list.loadError && !list.list}
      <p class="text-sm text-red-300" data-testid="open-vocab-load-error">
        {list.loadError}
      </p>
    {:else if !list.list}
      <p class="text-sm text-zinc-500">Loading…</p>
    {:else}
      {@const l = list.list}
      <section class="surface overflow-x-auto p-4" aria-label="Sets">
        <div class="mb-2 flex items-center gap-3">
          <h2 class="text-base font-semibold">Sets</h2>
          <span class="grow"></span>
          <button
            type="button"
            class="btn btn-sm"
            data-testid="open-vocab-new"
            onclick={() => {
              list.createError = null;
              creating = true;
            }}>New set</button
          >
        </div>
        <table class="w-full text-left text-sm" data-testid="open-vocab-table">
          <thead class="text-xs text-zinc-500">
            <tr>
              <th class="py-1 pr-3 font-normal">Name</th>
              <th class="py-1 pr-3 font-normal">Display name</th>
              <th class="py-1 pr-3 font-normal">Revision</th>
              <th class="py-1 pr-3 font-normal">Targets</th>
              <th class="py-1 pr-3 font-normal">Runs on ingest</th>
              <th class="py-1 pr-3 font-normal">Updated</th>
              <th class="py-1 font-normal"></th>
            </tr>
          </thead>
          <tbody>
            {#each l.sets as s (s.name)}
              <tr
                class="border-t border-zinc-800 align-top"
                data-testid="open-vocab-row"
                data-name={s.name}
              >
                <td class="py-1.5 pr-3">
                  <a
                    class="font-mono text-blue-300 hover:underline"
                    href={resolve(editorPath(s.name))}>{s.name}</a
                  >
                  {#if s.active}
                    <span
                      class="ml-1 rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 text-[11px] text-emerald-300"
                      data-testid="open-vocab-active-chip"
                      >active{s.active_revision != null
                        ? ` r${s.active_revision}`
                        : ''}</span
                    >
                    {#if s.revision != null && s.active_revision != null && s.revision !== s.active_revision}
                      <span class="ml-1 block text-[11px] text-amber-300"
                        >saved r{s.revision}, active r{s.active_revision}</span
                      >
                    {/if}
                  {/if}
                </td>
                <td class="py-1.5 pr-3 text-xs text-zinc-200">{s.display_name || '—'}</td>
                <td class="py-1.5 pr-3 font-mono text-xs">
                  {s.revision ?? '—'}{s.read_only ? ' · read-only' : ''}
                </td>
                <td class="py-1.5 pr-3 text-xs">
                  {s.n_enabled_targets} enabled of {s.n_targets}
                </td>
                <td class="py-1.5 pr-3 text-xs">{s.run_on_ingest ? 'yes' : 'no'}</td>
                <td class="py-1.5 pr-3 text-xs text-zinc-500">
                  {s.updated_at ? formatTimestamp(s.updated_at) : '—'}
                </td>
                <td class="py-1.5 text-right whitespace-nowrap">
                  <a class="btn btn-sm" href={resolve(editorPath(s.name))}>Open</a>
                  <button
                    type="button"
                    class="btn btn-sm"
                    onclick={() => openClone({ name: s.name, source: null })}
                    >Clone</button
                  >
                  {#if !s.read_only}
                    <button
                      type="button"
                      class="btn btn-sm"
                      onclick={() => {
                        list.deleteError = null;
                        deleting = s;
                      }}>Delete</button
                    >
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
        {#if l.sets.length === 0}
          <p class="mt-2 text-sm text-zinc-400" data-testid="open-vocab-empty">
            No sets yet. Create one, or clone a template below.
          </p>
        {/if}
      </section>

      {#if l.templates.length > 0}
        <section class="surface flex flex-col gap-2 p-4" aria-label="Templates">
          <h2 class="text-base font-semibold">Templates</h2>
          <p class="text-xs text-zinc-400">
            Starting points. Clone one to get a set you can edit and activate.
          </p>
          <ul class="space-y-1 text-sm" data-testid="open-vocab-templates">
            {#each l.templates as t (t.name + t.path)}
              <li class="flex flex-wrap items-center gap-2" data-testid="template-row">
                <span class="font-mono">{t.name}</span>
                {#if t.display_name}<span class="text-zinc-300">{t.display_name}</span
                  >{/if}
                <span class="text-xs text-zinc-500">{t.n_targets} targets</span>
                <code class="text-xs text-zinc-500">{t.path}</code>
                <button
                  type="button"
                  class="btn btn-sm"
                  onclick={() => openClone({ name: t.name, source: 'template' })}
                  >Clone</button
                >
              </li>
            {/each}
          </ul>
        </section>
      {/if}

      <OpenVocabRerunPanel
        active={list.active.active?.active.name != null}
        vocabulary={list.vocabulary}
      />
    {/if}
  </ConfigGate>
</div>

{#if creating}
  <ConfirmDialog
    title="New open-vocabulary set"
    confirmLabel="Create"
    busy={list.busy}
    confirmDisabled={newName.trim() === ''}
    onconfirm={() => void doCreate()}
    oncancel={() => (creating = false)}
  >
    <label class="flex flex-col gap-1 text-xs text-zinc-400">
      Set name
      <input
        class="input input-sm font-mono"
        bind:value={newName}
        data-testid="open-vocab-new-name"
      />
    </label>
    <p class="text-xs text-zinc-500">
      The set starts with the server's defaults; add targets in the editor.
    </p>
    {#if list.createError}<p class="text-red-300" data-testid="open-vocab-create-error">
        {list.createError}
      </p>{/if}
  </ConfirmDialog>
{/if}

{#if cloneFrom}
  <ConfigCloneDialog
    title="Clone {cloneFrom.name}{cloneFrom.source === 'template' ? ' (template)' : ''}"
    nameLabel="New set name"
    withDescription
    busy={list.busy}
    error={list.cloneError}
    report={list.cloneReport}
    onconfirm={(name, description) => void doClone(name, description)}
    oncancel={() => (cloneFrom = null)}
  />
{/if}

{#if deleting}
  <ConfirmDialog
    title="Delete {deleting.name}"
    confirmLabel="Delete"
    danger
    busy={list.busy}
    onconfirm={() => void doDelete()}
    oncancel={() => (deleting = null)}
  >
    <p>
      Deletes the set at revision {deleting.revision}. Items it already found keep their
      results.
    </p>
    {#if list.deleteError}<p class="text-red-300" data-testid="delete-error">
        {list.deleteError}
      </p>{/if}
  </ConfirmDialog>
{/if}
