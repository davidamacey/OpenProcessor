<script lang="ts">
  import PageHeader from '$components/ui/PageHeader.svelte';
  /**
   * `/settings/region-profiles` — the project's region profiles
   * Everything listed is served; `ProfileList` follows it. Absent (one line)
   * until the backend serves region profiles.
   */
  import { resolve } from '$app/paths';
  import ConfigActivePanel from '$components/config/ConfigActivePanel.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import ConfigRowTail from '$components/config/ConfigRowTail.svelte';
  import ConfigListDialogs from '$components/config/ConfigListDialogs.svelte';
  import ProfileImpactPanel from '$components/profiles/ProfileImpactPanel.svelte';
  import ProfileVocabularyPanel from '$components/profiles/ProfileVocabularyPanel.svelte';
  import { createCloneDeleteFlow } from '$lib/config/cloneDeleteFlow.svelte';
  import { PROFILE_ACTIVE_COPY } from '$lib/profiles/profileCopy';
  import { createProfileList } from '$lib/profiles/profileListController.svelte';
  import { profilesAvailability } from '$lib/profiles/profilesAvailability.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { RegionProfileSummary } from '$lib/types_profiles';
  import { keyboardStore } from '$stores/keyboard.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const list = createProfileList();

  $effect(() => {
    if (profilesAvailability.available !== true) return;
    list.start();
    return () => list.stop();
  });

  const flow = createCloneDeleteFlow<RegionProfileSummary>(list, (name) =>
    projectHref(`/settings/region-profiles/${encodeURIComponent(name)}`),
  );
</script>

<div class="flex flex-col gap-4">
  <PageHeader title="Region profiles">
    {#snippet actions()}
      <a
        class="text-meta text-accent-text hover:underline"
        href={resolve(projectHref('/settings'))}>Back to settings</a
      >
    {/snippet}
  </PageHeader>
  <p class="text-body text-fg-2">
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
      <p class="text-body text-danger-text" data-testid="impact-error">
        {list.impactError}
      </p>
    {/if}
    {#if list.impact}
      <ProfileImpactPanel impact={list.impact} />
    {/if}

    {#if list.loadError && !list.list}
      <p class="text-body text-danger-text" data-testid="profiles-load-error">
        {list.loadError}
      </p>
    {:else if !list.list}
      <p class="text-body text-fg-3">Loading…</p>
    {:else}
      {@const l = list.list}
      <section class="surface overflow-x-auto p-4" aria-label="Profiles">
        <table class="w-full text-left text-body" data-testid="profiles-table">
          <thead class="text-meta text-fg-3">
            <tr>
              <th class="py-1 pr-3 font-normal">Name</th>
              <th class="py-1 pr-3 font-normal">Region</th>
              <th class="py-1 pr-3 font-normal">Source</th>
              <th class="py-1 pr-3 font-normal">Revision</th>
              <th class="py-1 pr-3 font-normal">Finds regions with</th>
              <th class="py-1 pr-3 font-normal">Item classes</th>
              <th class="py-1 pr-3 font-normal">Updated</th>
              <th class="py-1 font-normal"><span class="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            {#each l.profiles as p (p.name)}
              <tr
                class="border-t border-line align-top"
                data-testid="profile-row"
                data-name={p.name}
              >
                <td class="py-1.5 pr-3">
                  <a
                    class="text-accent-text hover:underline"
                    href={resolve(
                      projectHref(
                        `/settings/region-profiles/${encodeURIComponent(p.name)}`,
                      ),
                    )}><code>{p.name}</code></a
                  >
                  {#if p.active}
                    <span
                      class="ml-1 rounded border border-success/40 bg-success/10 px-1.5 text-caption text-success-text"
                      data-testid="profile-active-chip"
                      >active{p.active_revision != null
                        ? ` r${p.active_revision}`
                        : ''}</span
                    >
                    {#if p.revision != null && p.active_revision != null && p.revision !== p.active_revision}
                      <span
                        class="ml-1 block text-caption text-warn-text"
                        data-testid="profile-saved-vs-active"
                        >saved r{p.revision}, active r{p.active_revision}</span
                      >
                    {/if}
                  {/if}
                </td>
                <td class="py-1.5 pr-3 text-meta">
                  <span class="text-fg-1">{p.display_name ?? '—'}</span>
                  {#if p.region_class_name}
                    <code class="ml-1 text-fg-3">{p.region_class_name}</code>
                  {/if}
                  <span class="block text-fg-3"
                    >{p.reads_text ? 'reads text' : 'no text'}</span
                  >
                </td>
                <td class="py-1.5 pr-3 text-meta text-fg-2">
                  {p.source}{p.read_only ? ' · read-only' : ''}
                </td>
                <td class="py-1.5 pr-3 text-meta tabular-nums">{p.revision ?? '—'}</td>
                <td class="py-1.5 pr-3 text-meta">
                  <span class="block"
                    >detector:
                    <code>{p.detector_model || 'none'}</code></span
                  >
                  <span class="block"
                    >segmenter prompt:
                    <span>{p.segmenter_text_prompt || 'none'}</span></span
                  >
                  {#if p.max_regions_per_item != null}
                    <span class="block text-fg-3"
                      >up to {p.max_regions_per_item} per item</span
                    >
                  {/if}
                </td>
                <td class="py-1.5 pr-3 text-meta text-fg-2">
                  {p.parent_classes.length > 0 ? p.parent_classes.join(', ') : '—'}
                </td>
                <ConfigRowTail row={p} {flow} />
              </tr>
            {/each}
          </tbody>
        </table>
        {#if l.profiles.length === 0}
          <p class="mt-2 text-body text-fg-2" data-testid="profiles-empty">
            No profiles yet. Clone a template below to start one.
          </p>
        {/if}
      </section>

      {#if (l.templates ?? []).length > 0}
        <section class="surface flex flex-col gap-2 p-4" aria-label="Templates">
          <h2 class="text-title font-semibold">Templates</h2>
          <p class="text-meta text-fg-2">
            Starting points. Clone one to get a profile you can edit and activate.
          </p>
          <ul class="space-y-1 text-body" data-testid="profile-templates">
            {#each l.templates ?? [] as t (t.name + t.path)}
              <li class="flex flex-wrap items-center gap-2" data-testid="template-row">
                <code>{t.name}</code>
                {#if t.display_name}<span class="text-fg-2">{t.display_name}</span>{/if}
                <span class="text-meta text-fg-3"
                  >{t.reads_text ? 'reads text' : 'no text'}</span
                >
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

      <details
        class="surface p-4"
        data-testid="vocabulary-details"
        ontoggle={(e) => {
          if ((e.currentTarget as HTMLDetailsElement).open) void list.loadVocabulary();
        }}
      >
        <summary class="cursor-pointer text-title font-semibold"
          >Models and sources</summary
        >
        <p class="mt-1 text-meta text-fg-2">
          What this deployment serves for a profile to use. Read only; a profile picks
          from these.
        </p>
        <div class="mt-3">
          {#if list.vocabularyError}
            <p class="text-body text-danger-text">{list.vocabularyError}</p>
          {:else if list.vocabulary}
            <ProfileVocabularyPanel vocab={list.vocabulary} />
          {:else}
            <p class="text-body text-fg-3">Loading…</p>
          {/if}
        </div>
      </details>
    {/if}
  </ConfigGate>
</div>

<ConfigListDialogs {flow} {list} nameLabel="New profile name">
  {#snippet deleteNote(row)}
    Deletes the profile at revision {row.revision}. Items it already produced keep their
    results.
  {/snippet}
</ConfigListDialogs>
