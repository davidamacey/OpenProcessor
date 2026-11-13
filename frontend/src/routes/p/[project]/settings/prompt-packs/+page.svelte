<script lang="ts">
  import PageHeader from '$components/ui/PageHeader.svelte';
  /**
   * `/settings/prompt-packs` — the project's VLM prompt packs.
   * Everything listed is served; `PackList` follows it. Absent (one line)
   * until the backend serves prompt packs.
   */
  import { resolve } from '$app/paths';
  import PackActivePanel from '$components/packs/PackActivePanel.svelte';
  import ConfigRowTail from '$components/config/ConfigRowTail.svelte';
  import ConfigListDialogs from '$components/config/ConfigListDialogs.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import { packsAvailability } from '$lib/packs/packsAvailability.svelte';
  import { createCloneDeleteFlow } from '$lib/config/cloneDeleteFlow.svelte';
  import { createPackList } from '$lib/packs/packListController.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { PromptPackSummary } from '$lib/types_packs';
  import { keyboardStore } from '$stores/keyboard.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const list = createPackList();

  $effect(() => {
    if (packsAvailability.available !== true) return;
    list.start();
    return () => list.stop();
  });

  const flow = createCloneDeleteFlow<PromptPackSummary>(list, (name) =>
    projectHref(`/settings/prompt-packs/${encodeURIComponent(name)}`),
  );
</script>

<div class="flex flex-col gap-4">
  <PageHeader title="Prompt packs">
    {#snippet actions()}
      <a
        class="text-meta text-accent-text hover:underline"
        href={resolve(projectHref('/settings'))}>Back to settings</a
      >
    {/snippet}
  </PageHeader>
  <p class="text-body text-fg-2">
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
      <p class="text-body text-danger-text" data-testid="packs-load-error">
        {list.loadError}
      </p>
    {:else if !list.list}
      <p class="text-body text-fg-3">Loading…</p>
    {:else}
      {@const l = list.list}
      <section class="surface overflow-x-auto p-4" aria-label="Packs">
        <table class="w-full text-left text-body" data-testid="packs-table">
          <thead class="text-meta text-fg-3">
            <tr>
              <th class="py-1 pr-3 font-normal">Name</th>
              <th class="py-1 pr-3 font-normal">Source</th>
              <th class="py-1 pr-3 font-normal">Revision</th>
              <th class="py-1 pr-3 font-normal">Description</th>
              <th class="py-1 pr-3 font-normal">Updated</th>
              <th class="py-1 font-normal"><span class="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            {#each l.packs as p (p.name)}
              <tr class="border-t border-line" data-testid="pack-row" data-name={p.name}>
                <td class="py-1.5 pr-3">
                  <a
                    class="text-accent-text hover:underline"
                    href={resolve(
                      projectHref(`/settings/prompt-packs/${encodeURIComponent(p.name)}`),
                    )}><code>{p.name}</code></a
                  >
                  {#if p.active}
                    <span
                      class="ml-1 rounded border border-success/40 bg-success/10 px-1.5 text-caption text-success-text"
                      data-testid="pack-active-chip"
                      >active{p.active_revision != null
                        ? ` r${p.active_revision}`
                        : ''}</span
                    >
                  {/if}
                  {#if p.asks_region_text}
                    <span class="ml-1 text-caption text-fg-3">asks for region text</span>
                  {/if}
                </td>
                <td class="py-1.5 pr-3 text-meta text-fg-2">
                  {p.source}{p.read_only ? ' · read-only' : ''}
                </td>
                <td class="py-1.5 pr-3 text-meta tabular-nums">{p.revision ?? '—'}</td>
                <td class="py-1.5 pr-3 text-meta text-fg-2">{p.description ?? ''}</td>
                <ConfigRowTail row={p} {flow} />
              </tr>
            {/each}
          </tbody>
        </table>
      </section>

      {#if l.templates.length > 0}
        <section class="surface flex flex-col gap-2 p-4" aria-label="Templates">
          <h2 class="text-title font-semibold">Templates</h2>
          <p class="text-meta text-fg-2">
            Starting points. Clone one to get a pack you can edit.
          </p>
          <ul class="space-y-1 text-body" data-testid="templates">
            {#each l.templates as t (t.name + t.path)}
              <li class="flex flex-wrap items-center gap-2" data-testid="template-row">
                <code>{t.name}</code>
                <code class="text-meta text-fg-3">{t.path}</code>
                <button
                  type="button"
                  class="btn btn-sm"
                  onclick={() => flow.openClone({ name: t.name, source: 'template' })}
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

<ConfigListDialogs {flow} {list} nameLabel="New pack name">
  {#snippet deleteNote(row)}
    Deletes the pack at revision {row.revision}. Its revision history is kept.
  {/snippet}
</ConfigListDialogs>
