<script lang="ts">
  /**
   * `/settings/presets/<kind>`: the documents of one research preset kind (serving profiles,
   * cluster profiles, assist-loop presets, near-duplicate profiles), the active ref, New, Clone and
   * Delete. Everything listed is served; the server validates every write.
   */
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import ConfigListDialogs from '$components/config/ConfigListDialogs.svelte';
  import ConfigRowTail from '$components/config/ConfigRowTail.svelte';
  import NearDupSettingsSections from '$lib/components/nearDup/NearDupSettingsSections.svelte';
  import PresetActiveSummary from '$components/presets/PresetActiveSummary.svelte';
  import PresetNewDialog from '$components/presets/PresetNewDialog.svelte';
  import { createCloneDeleteFlow } from '$lib/config/cloneDeleteFlow.svelte';
  import { createPresetList } from '$lib/presets/presetListController.svelte';
  import {
    activationReach,
    presetKindBySlug,
    presetPath,
  } from '$lib/presets/presetKinds';
  import { projectHref } from '$lib/projectPaths';
  import type { PresetBody, PresetDoc } from '$lib/types_presets';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const kind = $derived(presetKindBySlug(page.params.kind));
  const list = $derived(kind ? createPresetList(kind) : null);

  $effect(() => {
    const l = list;
    if (!l) return;
    l.start();
    return () => l.stop();
  });

  let creating = $state(false);
  const editorPath = (name: string) => presetPath(kind!, name);
  const flow = $derived(
    list
      ? createCloneDeleteFlow<PresetDoc>(list, (n) => projectHref(editorPath(n)))
      : null,
  );

  async function doCreate(name: string, description: string, body: PresetBody) {
    const doc = await list?.create(name, description, body);
    if (!doc) return;
    creating = false;
    toastStore.success(`Created ${doc.name}`);
    await goto(resolve(projectHref(editorPath(doc.name))));
  }

  const axisLabel = (axis: string | null): string => list?.active.axisLabel(axis) ?? '';
</script>

<div class="flex flex-col gap-4">
  {#if !kind || !list || !flow}
    <p class="text-body text-fg-2" data-testid="preset-unknown-kind">
      There is no such kind of preset.
    </p>
    <a
      class="text-meta text-accent-text hover:underline"
      href={resolve(projectHref('/settings'))}>Back to settings</a
    >
  {:else}
    <header class="flex flex-wrap items-baseline gap-3">
      <h1 class="text-h1 font-semibold text-fg-1">{kind.title}</h1>
      <span
        class="rounded border border-line-strong px-1.5 text-caption text-fg-2"
        data-testid="preset-scope"
        >{kind.scope === 'deployment' ? 'deployment-wide' : 'this project'}</span
      >
      <span class="grow"></span>
      <a
        class="text-meta text-accent-text hover:underline"
        href={resolve(projectHref('/settings'))}>Back to settings</a
      >
    </header>
    <p class="text-body text-fg-2">{kind.blurb} {activationReach(kind)}</p>

    <PresetActiveSummary active={list.active} />

    {#if list.loadError && !list.list}
      <p class="text-body text-danger-text" data-testid="preset-load-error">
        {list.loadError}
      </p>
    {:else if !list.list}
      <p class="text-body text-fg-3">Loading…</p>
    {:else}
      {@const items = list.list.items}
      <section
        class="surface relative min-w-0 overflow-x-auto p-4"
        aria-label="Documents"
      >
        <div class="mb-2 flex items-center gap-3">
          <h2 class="text-title font-semibold">Documents</h2>
          <span class="grow"></span>
          <button
            type="button"
            class="btn btn-sm"
            data-testid="preset-new"
            onclick={() => {
              list.createError = null;
              creating = true;
            }}>New {kind.noun}</button
          >
        </div>
        <table class="w-full text-left text-body" data-testid="preset-table">
          <thead class="text-meta text-fg-3">
            <tr>
              <th class="py-1 pr-3 font-normal">Name</th>
              <th class="py-1 pr-3 font-normal">Description</th>
              <th class="py-1 pr-3 font-normal">Revision</th>
              <th class="py-1 pr-3 font-normal">Updated</th>
              <th class="py-1 font-normal"><span class="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            {#each items as d (d.name)}
              <tr
                class="border-t border-line align-top"
                data-testid="preset-row"
                data-name={d.name}
              >
                <td class="py-1.5 pr-3">
                  <a
                    class="text-accent-text hover:underline"
                    href={resolve(projectHref(editorPath(d.name)))}
                    ><code>{d.name}</code></a
                  >
                  {#each list.active.entries.filter((e) => e.state === 'active' && e.name === d.name) as e (e.axis ?? 'deployment')}
                    <div
                      class="mt-0.5 flex flex-col items-start gap-0.5 whitespace-nowrap"
                    >
                      <span
                        class="rounded border border-success/40 bg-success/10 px-1.5 text-caption text-success-text"
                        data-testid="preset-active-chip"
                        >active{axisLabel(e.axis) ? ` · ${axisLabel(e.axis)}` : ''} r{e.revision}</span
                      >
                      {#if e.revision !== d.revision}
                        <span class="text-caption text-warn-text"
                          >saved r{d.revision}, active r{e.revision}</span
                        >
                      {/if}
                    </div>
                  {/each}
                </td>
                <td class="py-1.5 pr-3 text-meta text-fg-2">{d.description || '—'}</td>
                <td class="py-1.5 pr-3 text-meta tabular-nums">{d.revision}</td>
                <ConfigRowTail row={d} {flow}>
                  {#snippet lead()}
                    <a class="btn btn-sm" href={resolve(projectHref(editorPath(d.name)))}
                      >Open</a
                    >
                  {/snippet}
                </ConfigRowTail>
              </tr>
            {/each}
          </tbody>
        </table>
        {#if items.length === 0}
          <p class="mt-2 text-body text-fg-2" data-testid="preset-empty">
            No {kind.title.toLowerCase()} yet.
          </p>
        {/if}
      </section>
    {/if}

    {#if creating}
      <PresetNewDialog
        {list}
        oncreate={(n, d, b) => void doCreate(n, d, b)}
        oncancel={() => (creating = false)}
      />
    {/if}

    {#if kind.id === 'near_dup_profile'}
      <NearDupSettingsSections />
    {/if}

    <ConfigListDialogs {flow} {list} nameLabel="New name" offerProjects={false}>
      {#snippet deleteNote(row)}
        Deletes the {kind.noun} at revision {row.revision}.
      {/snippet}
    </ConfigListDialogs>
  {/if}
</div>
