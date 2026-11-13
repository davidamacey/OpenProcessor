<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  /**
   * Top-bar project switcher (review §7.3). Lists the served `selectable`
   * projects with their `display_name` and a served status badge for
   * anything that isn't `active`; picking one navigates to the same
   * section under the new slug (`switchProjectHref`, which drops ids that
   * don't carry across projects). The switch itself is the `/p/[project]`
   * layout's job — this component only navigates. Nothing is persisted.
   *
   * The list is one bounded, scrollable page the server filters as the operator types
   * (`ProjectPickerController`, `GET /projects?q=&limit=`). It follows the WAI-ARIA APG
   * combobox pattern with a listbox popup
   * (https://www.w3.org/WAI/ARIA/apg/patterns/combobox/): focus stays in the filter box,
   * Up/Down/Home/End move the highlighted option (`aria-activedescendant`), Enter picks it,
   * Escape closes.
   *
   * The "paused" chip is the ACTIVE project's served pause state: the
   * `GET {prefix}/pause` answer (`projectPauseStore`, read by the
   * `/p/[project]` layout and refreshed on a `project.paused`/`resumed`
   * event — it includes a global GPU-training claim, with its served
   * `paused_by` and `reason` in the tooltip), falling back to the served
   * summary's own `paused` until that read lands.
   *
   * The "custom keys" badge is the ACTIVE project's served keymap
   * `is_default === false` (already loaded, so it costs nothing); other
   * projects' keymaps aren't loaded, so they carry no badge.
   */
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { page } from '$app/state';
  import { switchProjectHref } from '$lib/projectPaths';
  import { keymapStore } from '$stores/keymap.svelte';
  import { projectPauseStore } from '$stores/projectPause.svelte';
  import { projectsStore } from '$stores/projects.svelte';
  import { ProjectPickerController } from '$lib/projects/projectPickerController.svelte';
  import { onDestroy, tick } from 'svelte';

  let open = $state(false);
  let root = $state<HTMLDivElement | null>(null);

  const current = $derived(projectsStore.current);
  const picker = new ProjectPickerController();
  let filterInput = $state<HTMLInputElement | null>(null);
  onDestroy(() => picker.dispose());
  const customKeys = $derived(keymapStore.source === 'served' && !keymapStore.isDefault);
  const pauseState = $derived(
    current ? projectPauseStore.stateFor(current.slug) : undefined,
  );
  const paused = $derived(pauseState ? pauseState.paused : (current?.paused ?? false));
  /** The served cause of the pause, verbatim: who holds it and the
   *  server's own sentence for a claim-only pause. */
  const pausedTitle = $derived.by(() => {
    const base =
      "This project's pipeline is paused: workers skip it until it's resumed on the Projects page";
    if (!pauseState) return base;
    const parts = [base];
    if (pauseState.paused_by.length > 0)
      parts.push(`Paused by: ${pauseState.paused_by.join(', ')}`);
    if (pauseState.reason) parts.push(pauseState.reason);
    return parts.join('. ');
  });

  async function toggle(): Promise<void> {
    open = !open;
    if (!open) return;
    picker.open();
    await tick();
    filterInput?.focus();
  }

  function onFilterKeydown(e: KeyboardEvent): void {
    if (e.key === 'ArrowDown') picker.move(1);
    else if (e.key === 'ArrowUp') picker.move(-1);
    else if (e.key === 'Home') picker.home();
    else if (e.key === 'End') picker.end();
    else if (e.key === 'Enter') {
      const p = picker.highlighted;
      if (p) choose(p.slug);
    } else return;
    e.preventDefault();
    void tick().then(() =>
      root
        ?.querySelector(`#project-option-${CSS.escape(picker.highlighted?.slug ?? '')}`)
        ?.scrollIntoView?.({ block: 'nearest' }),
    );
  }

  function choose(slug: string): void {
    open = false;
    const picked =
      picker.projects.find((p) => p.slug === slug) ??
      picker.recents.find((r) => r.slug === slug);
    if (picked) picker.remember(picked);
    if (slug === current?.slug) return;
    void goto(resolve(switchProjectHref(page.url, slug)));
  }

  function onWindowPointer(e: PointerEvent): void {
    if (open && root && !root.contains(e.target as Node)) open = false;
  }

  function onKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape' && open) {
      e.stopPropagation();
      open = false;
    }
  }
</script>

<svelte:window onpointerdown={onWindowPointer} />

<div class="relative min-w-0 shrink" bind:this={root} data-testid="project-switcher">
  <button
    type="button"
    class="flex max-w-[min(16rem,55vw)] min-w-0 items-center gap-1.5 rounded border border-line px-2 py-1 text-body text-fg-1 hover:border-line-strong"
    aria-haspopup="listbox"
    aria-expanded={open}
    title="Switch project"
    data-testid="project-switcher-trigger"
    onclick={toggle}
    onkeydown={onKeydown}
  >
    <span class="truncate" data-testid="project-switcher-current"
      >{current?.display_name ?? '—'}</span
    >
    {#if current && current.status !== 'active'}
      <span
        class="shrink-0 rounded bg-warn-soft px-1 text-caption text-warn-text"
        data-testid="project-switcher-status"
        >{projectsStore.statusLabel(current.status)}</span
      >
    {/if}
    {#if paused}
      <span
        class="shrink-0 rounded bg-warn-soft px-1 text-caption text-warn-text"
        title={pausedTitle}
        data-testid="project-switcher-paused">paused</span
      >
    {/if}
    {#if customKeys}
      <span
        class="shrink-0 rounded bg-surface-2 px-1 text-caption text-fg-2"
        title="This project's keyboard shortcuts differ from the defaults"
        data-testid="project-switcher-custom-keys">custom keys</span
      >
    {/if}
    <Icon name="chevron-down" size={14} class="text-fg-3" />
  </button>

  {#if open}
    <div
      class="absolute left-0 top-full z-60 mt-1 w-72 max-w-[calc(100vw-1rem)] rounded border border-line-strong bg-surface-1 shadow-xl"
      role="presentation"
      data-testid="project-switcher-menu"
      onkeydown={onKeydown}
    >
      <input
        bind:this={filterInput}
        type="text"
        class="block w-full rounded-t border-b border-line-strong bg-surface-0 px-3 py-1.5 text-body text-fg-1 placeholder:text-fg-3 focus:outline-none"
        placeholder="Find a project…"
        role="combobox"
        aria-label="Find a project"
        aria-expanded="true"
        aria-controls="project-switcher-listbox"
        aria-autocomplete="list"
        aria-activedescendant={picker.highlighted
          ? `project-option-${picker.highlighted.slug}`
          : undefined}
        autocomplete="off"
        data-testid="project-switcher-filter"
        value={picker.query}
        oninput={(e) => picker.type(e.currentTarget.value)}
        onkeydown={onFilterKeydown}
      />
      <div
        id="project-switcher-listbox"
        class="max-h-[min(18rem,50vh)] overflow-y-auto py-1"
        role="listbox"
        aria-label="Projects"
        data-testid="project-switcher-list"
      >
        {#if picker.query.trim() === '' && picker.recents.length > 0}
          <p class="px-3 pt-1 text-meta text-fg-2 font-semibold">Recent</p>
          {#each picker.recents as r (r.slug)}
            <button
              type="button"
              class="block w-full truncate px-3 py-1 text-left text-body text-fg-2 hover:bg-surface-2"
              data-testid="project-recent-{r.slug}"
              onclick={() => choose(r.slug)}>{r.display_name}</button
            >
          {/each}
          <p class="px-3 pt-1 text-meta text-fg-2 font-semibold">All</p>
        {/if}
        {#each picker.projects as p, i (p.slug)}
          <div
            id="project-option-{p.slug}"
            role="option"
            tabindex="-1"
            aria-selected={i === picker.active}
            class="flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-left text-body {i ===
            picker.active
              ? 'bg-surface-2'
              : ''} {p.slug === current?.slug ? 'text-white' : 'text-fg-2'}"
            data-testid="project-option-{p.slug}"
            onclick={() => choose(p.slug)}
            onkeydown={() => {}}
          >
            <span class="min-w-0 flex-1">
              <span class="block truncate">{p.display_name}</span>
              <code class="block truncate text-caption text-fg-3">{p.slug}</code>
            </span>
            {#if p.status !== 'active'}
              <span class="shrink-0 rounded bg-warn-soft px-1 text-caption text-warn-text"
                >{projectsStore.statusLabel(p.status)}</span
              >
            {/if}
            {#if p.slug === current?.slug}
              <span class="shrink-0 text-meta text-fg-3">current</span>
            {/if}
          </div>
        {:else}
          <p class="px-3 py-2 text-meta text-fg-3" role="status">
            {#if picker.error}{picker.error}{:else if picker.loading}Loading…{:else}No
              project matches "{picker.query}".{/if}
          </p>
        {/each}
      </div>
      {#if picker.projects.length > 0}
        <div
          class="flex items-center gap-2 px-3 py-1 text-caption text-fg-3"
          data-testid="project-switcher-more"
        >
          <span>Showing {picker.projects.length} of {picker.total}</span>
          {#if picker.hasMore}
            <button
              type="button"
              class="rounded border border-line-strong px-1.5 py-0.5 text-fg-2 hover:border-fg-3"
              disabled={picker.loading}
              data-testid="project-switcher-more-button"
              onclick={() => void picker.more()}>Show more</button
            >
          {/if}
        </div>
      {/if}
      <div class="my-1 border-t border-line"></div>
      <a
        href={resolve('/projects')}
        class="block px-3 py-1.5 text-body text-fg-2 hover:bg-surface-2 hover:text-white"
        data-testid="project-switcher-manage"
        onclick={() => (open = false)}>Manage projects…</a
      >
    </div>
  {/if}
</div>
