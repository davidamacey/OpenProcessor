<script lang="ts">
  /**
   * `/settings/region-profiles` — the project's region profiles
   * (any_domain_plan.md §4, §7.3, §7.4, §7.6 items 2 and 4; docs/design/
   * w4-profile-editor-ui-plan-2026-09-27.md §3). Everything listed is
   * served; `ProfileList` follows it. Absent (one line) until the backend
   * serves W4.
   */
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import ConfigActivePanel from '$components/config/ConfigActivePanel.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import ConfigCloneDialog from '$components/config/ConfigCloneDialog.svelte';
  import ProfileImpactPanel from '$components/profiles/ProfileImpactPanel.svelte';
  import ProfileVocabularyPanel from '$components/profiles/ProfileVocabularyPanel.svelte';
  import type { CloneSource } from '$lib/config/configList.svelte';
  import { formatTimestamp } from '$lib/formatDate';
  import { PROFILE_ACTIVE_COPY } from '$lib/profiles/profileCopy';
  import { createProfileList } from '$lib/profiles/profileListController.svelte';
  import { profilesAvailability } from '$lib/profiles/profilesAvailability.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { RegionProfileSummary } from '$lib/types_profiles';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const list = createProfileList();

  $effect(() => {
    if (profilesAvailability.available !== true) return;
    list.start();
    return () => list.stop();
  });

  let cloneFrom = $state<CloneSource | null>(null);
  let deleting = $state<RegionProfileSummary | null>(null);

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
      resolve(projectHref(`/settings/region-profiles/${encodeURIComponent(doc.name)}`)),
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
    <h1 class="text-2xl font-semibold tracking-tight">Region profiles</h1>
    <span class="grow"></span>
    <a
      class="text-xs text-blue-300 hover:underline"
      href={resolve(projectHref('/settings'))}>Back to settings</a
    >
  </header>
  <p class="text-sm text-zinc-400">
    A region profile says which part of an item to find (and whether to read text on it):
    which item classes to search, the detector and segmenter that propose boxes, and how
    many to keep. Open a profile to edit it and make it the active one.
  </p>

  <ConfigGate
    store={profilesAvailability}
    what="region profiles"
    unavailableText="Region-profile editing is not available on this backend."
    testid="profiles-unavailable"
  >
    <ConfigActivePanel
      ctl={list.active}
      copy={PROFILE_ACTIVE_COPY}
      onrollback={() => list.rollback()}
      ondeactivate={() => list.deactivate()}
    >
      {#snippet actions()}
        <button
          type="button"
          class="btn btn-sm"
          disabled={list.impactLoading}
          data-testid="impact-open"
          onclick={() => void list.loadImpact()}
          >{list.impact ? 'Refresh impact' : 'Show impact'}</button
        >
      {/snippet}
    </ConfigActivePanel>

    {#if list.impactError}
      <p class="text-sm text-red-300" data-testid="impact-error">{list.impactError}</p>
    {/if}
    {#if list.impact}
      <ProfileImpactPanel impact={list.impact} />
    {/if}

    {#if list.loadError && !list.list}
      <p class="text-sm text-red-300" data-testid="profiles-load-error">
        {list.loadError}
      </p>
    {:else if !list.list}
      <p class="text-sm text-zinc-500">Loading…</p>
    {:else}
      {@const l = list.list}
      <section class="surface overflow-x-auto p-4" aria-label="Profiles">
        <table class="w-full text-left text-sm" data-testid="profiles-table">
          <thead class="text-xs text-zinc-500">
            <tr>
              <th class="py-1 pr-3 font-normal">Name</th>
              <th class="py-1 pr-3 font-normal">Region</th>
              <th class="py-1 pr-3 font-normal">Source</th>
              <th class="py-1 pr-3 font-normal">Revision</th>
              <th class="py-1 pr-3 font-normal">Finds regions with</th>
              <th class="py-1 pr-3 font-normal">Item classes</th>
              <th class="py-1 pr-3 font-normal">Updated</th>
              <th class="py-1 font-normal"></th>
            </tr>
          </thead>
          <tbody>
            {#each l.profiles as p (p.name)}
              <tr
                class="border-t border-zinc-800 align-top"
                data-testid="profile-row"
                data-name={p.name}
              >
                <td class="py-1.5 pr-3">
                  <a
                    class="font-mono text-blue-300 hover:underline"
                    href={resolve(
                      projectHref(
                        `/settings/region-profiles/${encodeURIComponent(p.name)}`,
                      ),
                    )}>{p.name}</a
                  >
                  {#if p.active}
                    <span
                      class="ml-1 rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 text-[11px] text-emerald-300"
                      data-testid="profile-active-chip"
                      >active{p.active_revision != null
                        ? ` r${p.active_revision}`
                        : ''}</span
                    >
                    {#if p.revision != null && p.active_revision != null && p.revision !== p.active_revision}
                      <span
                        class="ml-1 block text-[11px] text-amber-300"
                        data-testid="profile-saved-vs-active"
                        >saved r{p.revision}, active r{p.active_revision}</span
                      >
                    {/if}
                  {/if}
                </td>
                <td class="py-1.5 pr-3 text-xs">
                  <span class="text-zinc-200">{p.display_name ?? '—'}</span>
                  {#if p.region_class_name}
                    <code class="ml-1 font-mono text-zinc-500">{p.region_class_name}</code
                    >
                  {/if}
                  <span class="block text-zinc-500"
                    >{p.reads_text ? 'reads text' : 'no text'}</span
                  >
                </td>
                <td class="py-1.5 pr-3 text-xs text-zinc-400">
                  {p.source}{p.read_only ? ' · read-only' : ''}
                </td>
                <td class="py-1.5 pr-3 font-mono text-xs">{p.revision ?? '—'}</td>
                <td class="py-1.5 pr-3 text-xs">
                  <span class="block"
                    >detector:
                    <span class="font-mono">{p.detector_model || 'none'}</span></span
                  >
                  <span class="block"
                    >segmenter prompt:
                    <span class="font-mono">{p.segmenter_text_prompt || 'none'}</span
                    ></span
                  >
                  {#if p.max_regions_per_item != null}
                    <span class="block text-zinc-500"
                      >up to {p.max_regions_per_item} per item</span
                    >
                  {/if}
                </td>
                <td class="py-1.5 pr-3 font-mono text-xs text-zinc-300">
                  {p.parent_classes.length > 0 ? p.parent_classes.join(', ') : '—'}
                </td>
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
        {#if l.profiles.length === 0}
          <p class="mt-2 text-sm text-zinc-400" data-testid="profiles-empty">
            No profiles yet. Clone a template below to start one.
          </p>
        {/if}
      </section>

      {#if (l.templates ?? []).length > 0}
        <section class="surface flex flex-col gap-2 p-4" aria-label="Templates">
          <h2 class="text-base font-semibold">Templates</h2>
          <p class="text-xs text-zinc-400">
            Starting points. Clone one to get a profile you can edit and activate.
          </p>
          <ul class="space-y-1 text-sm" data-testid="profile-templates">
            {#each l.templates ?? [] as t (t.name + t.path)}
              <li class="flex flex-wrap items-center gap-2" data-testid="template-row">
                <span class="font-mono">{t.name}</span>
                {#if t.display_name}<span class="text-zinc-300">{t.display_name}</span
                  >{/if}
                <span class="text-xs text-zinc-500"
                  >{t.reads_text ? 'reads text' : 'no text'}</span
                >
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

      <details
        class="surface p-4"
        data-testid="vocabulary-details"
        ontoggle={(e) => {
          if ((e.currentTarget as HTMLDetailsElement).open) void list.loadVocabulary();
        }}
      >
        <summary class="cursor-pointer text-base font-semibold"
          >Models and sources</summary
        >
        <p class="mt-1 text-xs text-zinc-400">
          What this deployment serves for a profile to use. Read only; a profile picks
          from these.
        </p>
        <div class="mt-3">
          {#if list.vocabularyError}
            <p class="text-sm text-red-300">{list.vocabularyError}</p>
          {:else if list.vocabulary}
            <ProfileVocabularyPanel vocab={list.vocabulary} />
          {:else}
            <p class="text-sm text-zinc-500">Loading…</p>
          {/if}
        </div>
      </details>
    {/if}
  </ConfigGate>
</div>

{#if cloneFrom}
  <ConfigCloneDialog
    title="Clone {cloneFrom.name}{cloneFrom.source === 'template' ? ' (template)' : ''}"
    nameLabel="New profile name"
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
      Deletes the profile at revision {deleting.revision}. Items it already produced keep
      their results.
    </p>
    {#if list.deleteError}<p class="text-red-300" data-testid="delete-error">
        {list.deleteError}
      </p>{/if}
  </ConfirmDialog>
{/if}
