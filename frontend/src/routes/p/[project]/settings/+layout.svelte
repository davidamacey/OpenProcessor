<script lang="ts">
  /**
   * Settings: one left sidebar (a select below 768 px) over one route per section, the content
   * at 6xl. The sidebar search (`/` focuses it) filters the static navigation index in
   * `$lib/settings/settingsNav`; each page shows its scope as a badge here, so a deployment-wide
   * page is never mistaken for a project one.
   *
   * Docs read: SvelteKit routing, nested layouts (svelte.dev/docs/kit/routing#layout).
   */
  import type { Snippet } from 'svelte';
  import { page } from '$app/state';
  import Badge from '$components/ui/Badge.svelte';
  import Kbd from '$components/ui/Kbd.svelte';
  import PageShell from '$components/ui/PageShell.svelte';
  import SidebarNav, { type SidebarGroup } from '$components/ui/SidebarNav.svelte';
  import { projectHref } from '$lib/projectPaths';
  import {
    SETTINGS_GROUP_LABELS,
    SETTINGS_SECTIONS,
    searchSections,
    sectionOfPath,
    type SettingsNeed,
  } from '$lib/settings/settingsNav';
  import { openVocabAvailability } from '$lib/openVocab/openVocabAvailability.svelte';
  import { packsAvailability } from '$lib/packs/packsAvailability.svelte';
  import { profilesAvailability } from '$lib/profiles/profilesAvailability.svelte';
  import { vlmAvailability } from '$lib/vlm/vlmAvailability.svelte';

  let { children }: { children: Snippet } = $props();

  $effect(() => {
    void packsAvailability.init();
    void profilesAvailability.init();
    void vlmAvailability.init();
    void openVocabAvailability.init();
  });

  // A feature the backend does not serve has no section (absent, not disabled).
  const served: Record<SettingsNeed, () => boolean | null> = {
    packs: () => packsAvailability.available,
    profiles: () => profilesAvailability.available,
    vlm: () => vlmAvailability.available,
    openVocab: () => openVocabAvailability.available,
  };
  const offered = $derived(
    SETTINGS_SECTIONS.filter((s) => !s.needs || served[s.needs]() === true),
  );

  let query = $state('');
  let search: HTMLInputElement | undefined = $state();
  const shown = $derived(searchSections(offered, query));
  const groups = $derived<SidebarGroup[]>(
    (Object.keys(SETTINGS_GROUP_LABELS) as (keyof typeof SETTINGS_GROUP_LABELS)[])
      .map((g) => ({
        heading: SETTINGS_GROUP_LABELS[g],
        items: shown
          .filter((s) => s.group === g)
          .map((s) => ({ id: s.id, label: s.label, href: projectHref(s.path) })),
      }))
      .filter((g) => g.items.length > 0),
  );
  const current = $derived(sectionOfPath(page.url.pathname));

  function focusSearch(e: KeyboardEvent): void {
    if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target as HTMLElement | null;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    e.preventDefault();
    search?.focus();
  }
</script>

<svelte:window onkeydown={focusSearch} />

<PageShell class="md:flex md:gap-8">
  <aside class="mb-4 shrink-0 md:mb-0 md:w-60" data-testid="settings-sidebar">
    <label class="relative mb-3 block">
      <span class="sr-only">Search settings</span>
      <input
        bind:this={search}
        bind:value={query}
        type="search"
        class="input w-full pr-8"
        placeholder="Search settings"
        data-testid="settings-search"
        onkeydown={(e) => {
          if (e.key === 'Escape') {
            query = '';
            search?.blur();
          }
        }}
      />
      <span class="pointer-events-none absolute top-1/2 right-2 -translate-y-1/2">
        <Kbd>/</Kbd>
      </span>
    </label>
    {#if groups.length > 0}
      <SidebarNav {groups} current={current?.id ?? ''} label="Settings sections" />
    {:else}
      <p class="px-3 text-meta text-fg-3" data-testid="settings-search-empty">
        No setting matches “{query}”.
      </p>
    {/if}
  </aside>

  <div class="min-w-0 flex-1 overflow-x-auto" data-testid="settings-content">
    {#if current}
      <div class="mb-3 flex items-center gap-2 text-meta text-fg-3">
        <span>{SETTINGS_GROUP_LABELS[current.group]}</span>
        <Badge tone={current.scope === 'Deployment' ? 'warn' : 'neutral'}
          >{current.scope}</Badge
        >
      </div>
    {/if}
    {@render children()}
  </div>
</PageShell>
