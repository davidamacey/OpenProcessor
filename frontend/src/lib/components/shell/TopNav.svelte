<script lang="ts">
  /**
   * The app's top bar: 48px, on the nav surface with elevation e2 and a backdrop blur. Left to
   * right: the menu button (below 768px), the logo, the project switcher, the breadcrumb, the
   * grouped nav (Dashboard | Data | Model | Quality | Settings; a group with several pages is a
   * menu), Resources and the API chip. Below 768px the nav moves into a left Drawer. Every
   * label, group and path comes from the one table in `$lib/projectPaths`.
   */
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import AboutModal from '$components/AboutModal.svelte';
  import AppLogo from '$components/AppLogo.svelte';
  import ProjectSwitcher from '$components/ProjectSwitcher.svelte';
  import ResourcesMenu from '$components/ResourcesMenu.svelte';
  import Button from '$components/ui/Button.svelte';
  import Drawer from '$components/ui/Drawer.svelte';
  import Icon from '$components/ui/Icon.svelte';
  import Menu from '$components/ui/Menu.svelte';
  import {
    NAV_GROUPS,
    breadcrumbOf,
    navGroupOf,
    projectHref,
    sectionOf,
  } from '$lib/projectPaths';
  import { HEALTH_CHIP_TEXT, healthChip, healthStore } from '$stores/health.svelte';

  interface Props {
    apiBase: string;
    appName: string;
    appBadge: string;
  }
  let { apiBase, appName, appBadge }: Props = $props();

  let aboutOpen = $state(false);
  let drawerOpen = $state(false);

  const section = $derived(sectionOf(page.url.pathname));
  const activeGroup = $derived(section ? navGroupOf(section) : null);
  const crumbs = $derived(breadcrumbOf(page.url.pathname));

  const ITEM =
    'relative inline-flex h-12 items-center gap-1 px-3 text-control whitespace-nowrap hover:text-fg-1 after:absolute after:inset-x-2 after:bottom-0 after:h-0.5 after:rounded-full';
  const itemClass = (active: boolean): string =>
    `${ITEM} ${active ? 'text-fg-1 after:bg-accent' : 'text-fg-2 after:bg-transparent'}`;

  const chipState = $derived(healthChip(healthStore.ok, healthStore.lastChecked));
  const dotClass = $derived(
    chipState === 'ok' ? 'bg-success' : chipState === 'down' ? 'bg-danger' : 'bg-fg-3',
  );
  const where = $derived(apiBase || window.location.host);
  const dotTitle = $derived(
    chipState === 'checking'
      ? `Checking API at ${where}…`
      : healthStore.ok
        ? `API OK at ${where} (last checked ${healthStore.lastChecked ? new Date(healthStore.lastChecked).toLocaleTimeString() : '—'})`
        : `API unavailable at ${where}: ${healthStore.error ?? 'no response'}`,
  );
</script>

<header
  class="relative z-40 flex h-12 shrink-0 items-center gap-3 border-b border-line bg-surface-nav px-4 shadow-e2 backdrop-blur-md md:px-gutter"
>
  <div class="md:hidden">
    <Button
      variant="ghost"
      icon="menu"
      aria-label="Open navigation"
      data-testid="nav-open"
      onclick={() => (drawerOpen = true)}
    />
  </div>

  <div class="flex shrink-0 items-center gap-2 text-body font-semibold tracking-tight">
    <button
      type="button"
      class="flex shrink-0 items-center justify-center rounded-control border border-line-strong transition-transform duration-150 hover:scale-110"
      onclick={() => (aboutOpen = true)}
      aria-label="About {appName}"
      title="About {appName}"
    >
      <AppLogo class="h-6 w-6" label={appBadge} />
    </button>
    <a href={resolve(projectHref('/dashboard'))} class="hidden hover:text-white lg:inline"
      >{appName}</a
    >
  </div>

  <ProjectSwitcher />

  <AboutModal open={aboutOpen} onclose={() => (aboutOpen = false)} {appName} />

  {#if crumbs.length}
    <nav
      class="hidden shrink-0 items-center gap-1 text-control whitespace-nowrap lg:flex"
      aria-label="Breadcrumb"
      data-testid="breadcrumb"
    >
      {#each crumbs as crumb, i (i)}
        {#if i > 0}<Icon name="chevron-right" size={14} class="text-fg-3" />{/if}
        {#if crumb.href}
          <a
            href={resolve(crumb.href as '/')}
            class="shrink-0 rounded-chip px-1.5 py-0.5 hover:bg-surface-2 hover:text-fg-1 {i ===
            crumbs.length - 1
              ? 'text-fg-1'
              : 'text-fg-2'}"
            aria-current={i === crumbs.length - 1 ? 'page' : undefined}>{crumb.label}</a
          >
        {:else}
          <span
            class="max-w-48 truncate px-1.5 {i === crumbs.length - 1
              ? 'text-fg-1'
              : 'text-fg-3'}"
            aria-current={i === crumbs.length - 1 ? 'page' : undefined}
            >{crumb.label}</span
          >
        {/if}
      {/each}
    </nav>
  {/if}

  <span class="grow"></span>

  <nav aria-label="Primary" class="hidden items-center md:flex" data-testid="primary-nav">
    {#each NAV_GROUPS as group (group.id)}
      {#if group.sections.length === 1}
        {@const only = group.sections[0]!}
        <a
          href={resolve(projectHref(`/${only.section}`))}
          class={itemClass(activeGroup === group.id)}
          aria-current={section === only.section ? 'page' : undefined}>{only.label}</a
        >
      {:else}
        <Menu
          label={group.label}
          align="start"
          current={section ?? undefined}
          triggerClass={itemClass(activeGroup === group.id)}
          items={group.sections.map((s) => ({
            id: s.section,
            label: s.label,
            href: projectHref(`/${s.section}`),
          }))}
        />
      {/if}
    {/each}
  </nav>

  <ResourcesMenu />

  <span
    class="chip shrink-0 gap-1.5 rounded-full bg-surface-1 text-fg-2"
    title={dotTitle}
    data-testid="api-health"
  >
    <span class="size-2 rounded-full {dotClass}"></span>
    <span class="hidden font-mono lg:inline" data-testid="api-health-chip"
      >{HEALTH_CHIP_TEXT[chipState]}</span
    >
    <span class="sr-only lg:hidden">{HEALTH_CHIP_TEXT[chipState]}</span>
  </span>
</header>

{#if drawerOpen}
  <Drawer label="Navigation" onclose={() => (drawerOpen = false)}>
    <nav aria-label="Primary" class="flex flex-col py-2" data-testid="drawer-nav">
      {#each NAV_GROUPS as group (group.id)}
        {#if group.sections.length > 1}
          <p
            class="px-4 pt-3 pb-1 text-caption font-medium tracking-wide text-fg-3 uppercase"
          >
            {group.label}
          </p>
        {:else if group.id !== 'home'}
          <div class="my-1 border-t border-line" role="separator"></div>
        {/if}
        {#each group.sections as s (s.section)}
          <a
            href={resolve(projectHref(`/${s.section}`))}
            class="flex items-center border-l-2 px-4 py-2.5 text-body {section ===
            s.section
              ? 'border-accent bg-accent-soft text-fg-1'
              : 'border-transparent text-fg-2 hover:bg-surface-2 hover:text-fg-1'}"
            aria-current={section === s.section ? 'page' : undefined}
            onclick={() => (drawerOpen = false)}>{s.label}</a
          >
        {/each}
      {/each}
    </nav>
  </Drawer>
{/if}
