<script lang="ts">
  /**
   * `/settings/prompt-packs` — the project's VLM prompt packs
   * (any_domain_plan.md §7.2, §7.6 items 1 and 4; docs/design/
   * w3-pack-editor-ui-plan-2026-09-27.md §2). Everything listed is served;
   * `PackList` follows it. Absent (one line) until the backend serves W3.
   */
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import PackActivePanel from '$components/packs/PackActivePanel.svelte';
  import ConfigCloneDialog from '$components/config/ConfigCloneDialog.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import { formatTimestamp } from '$lib/formatDate';
  import { packsAvailability } from '$lib/packs/packsAvailability.svelte';
  import type { CloneSource } from '$lib/config/configList.svelte';
  import { createPackList } from '$lib/packs/packListController.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { PromptPackSummary } from '$lib/types_packs';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const list = createPackList();

  $effect(() => {
    if (packsAvailability.available !== true) return;
    list.start();
    return () => list.stop();
  });

  let cloneFrom = $state<CloneSource | null>(null);
  let deleting = $state<PromptPackSummary | null>(null);

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
    await goto(
      resolve(projectHref(`/settings/prompt-packs/${encodeURIComponent(doc.name)}`)),
    );
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
    <h1 class="text-2xl font-semibold tracking-tight">Prompt packs</h1>
    <span class="grow"></span>
    <a
      class="text-xs text-blue-300 hover:underline"
      href={resolve(projectHref('/settings'))}>Back to settings</a
    >
  </header>
  <p class="text-sm text-zinc-400">
    A prompt pack holds the instructions the VLM gets for classifying items and checking
    regions. Open a pack to edit it, test it on a crop and make it the active one.
  </p>

  <ConfigGate
    store={packsAvailability}
    what="prompt packs"
    unavailableText="Prompt-pack editing is not available on this backend."
    testid="packs-unavailable"
  >
    <PackActivePanel ctl={list.active} onrollback={() => list.rollback()} />

    {#if list.loadError && !list.list}
      <p class="text-sm text-red-300" data-testid="packs-load-error">{list.loadError}</p>
    {:else if !list.list}
      <p class="text-sm text-zinc-500">Loading…</p>
    {:else}
      {@const l = list.list}
      <section class="surface overflow-x-auto p-4" aria-label="Packs">
        <table class="w-full text-left text-sm" data-testid="packs-table">
          <thead class="text-xs text-zinc-500">
            <tr>
              <th class="py-1 pr-3 font-normal">Name</th>
              <th class="py-1 pr-3 font-normal">Source</th>
              <th class="py-1 pr-3 font-normal">Revision</th>
              <th class="py-1 pr-3 font-normal">Description</th>
              <th class="py-1 pr-3 font-normal">Updated</th>
              <th class="py-1 font-normal"></th>
            </tr>
          </thead>
          <tbody>
            {#each l.packs as p (p.name)}
              <tr
                class="border-t border-zinc-800"
                data-testid="pack-row"
                data-name={p.name}
              >
                <td class="py-1.5 pr-3">
                  <a
                    class="font-mono text-blue-300 hover:underline"
                    href={resolve(
                      projectHref(`/settings/prompt-packs/${encodeURIComponent(p.name)}`),
                    )}>{p.name}</a
                  >
                  {#if p.active}
                    <span
                      class="ml-1 rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 text-[11px] text-emerald-300"
                      data-testid="pack-active-chip"
                      >active{p.active_revision != null
                        ? ` r${p.active_revision}`
                        : ''}</span
                    >
                  {/if}
                  {#if p.asks_region_text}
                    <span class="ml-1 text-[11px] text-zinc-500"
                      >asks for region text</span
                    >
                  {/if}
                </td>
                <td class="py-1.5 pr-3 text-xs text-zinc-400">
                  {p.source}{p.read_only ? ' · read-only' : ''}
                </td>
                <td class="py-1.5 pr-3 font-mono text-xs">{p.revision ?? '—'}</td>
                <td class="py-1.5 pr-3 text-xs text-zinc-300">{p.description ?? ''}</td>
                <td class="py-1.5 pr-3 text-xs text-zinc-500">
                  {p.updated_at ? formatTimestamp(p.updated_at) : '—'}
                </td>
                <td class="py-1.5 text-right whitespace-nowrap">
                  <button
                    type="button"
                    class="btn btn-sm"
                    onclick={() => openClone({ name: p.name, source: null })}
                    >Clone</button
                  >
                  {#if !p.read_only}
                    <button
                      type="button"
                      class="btn btn-sm"
                      onclick={() => {
                        list.deleteError = null;
                        deleting = p;
                      }}>Delete</button
                    >
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </section>

      {#if l.templates.length > 0}
        <section class="surface flex flex-col gap-2 p-4" aria-label="Templates">
          <h2 class="text-base font-semibold">Templates</h2>
          <p class="text-xs text-zinc-400">
            Starting points. Clone one to get a pack you can edit.
          </p>
          <ul class="space-y-1 text-sm" data-testid="templates">
            {#each l.templates as t (t.name + t.path)}
              <li class="flex flex-wrap items-center gap-2" data-testid="template-row">
                <span class="font-mono">{t.name}</span>
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
    {/if}
  </ConfigGate>
</div>

{#if cloneFrom}
  <ConfigCloneDialog
    title="Clone {cloneFrom.name}{cloneFrom.source === 'template' ? ' (template)' : ''}"
    nameLabel="New pack name"
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
      Deletes the pack at revision {deleting.revision}. Its revision history is kept.
    </p>
    {#if list.deleteError}<p class="text-red-300" data-testid="delete-error">
        {list.deleteError}
      </p>{/if}
  </ConfirmDialog>
{/if}
