<script lang="ts">
  /**
   * Top-bar project switcher (review §7.3). Lists the served `selectable`
   * projects with their `display_name` and a served status badge for
   * anything that isn't `active`; picking one navigates to the same
   * section under the new slug (`switchProjectHref`, which drops ids that
   * don't carry across projects). The switch itself is the `/p/[project]`
   * layout's job — this component only navigates. Nothing is persisted.
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

  let open = $state(false);
  let root = $state<HTMLDivElement | null>(null);

  const current = $derived(projectsStore.current);
  const options = $derived(projectsStore.selectable);
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

  function choose(slug: string): void {
    open = false;
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

<div class="relative shrink-0" bind:this={root} data-testid="project-switcher">
  <button
    type="button"
    class="flex max-w-[16rem] items-center gap-1.5 rounded border border-zinc-800 px-2 py-1 text-sm text-zinc-200 hover:border-zinc-600"
    aria-haspopup="listbox"
    aria-expanded={open}
    title="Switch project"
    data-testid="project-switcher-trigger"
    onclick={() => (open = !open)}
    onkeydown={onKeydown}
  >
    <span class="truncate" data-testid="project-switcher-current"
      >{current?.display_name ?? '—'}</span
    >
    {#if current && current.status !== 'active'}
      <span
        class="shrink-0 rounded bg-amber-950/60 px-1 text-[10px] uppercase tracking-wide text-amber-300"
        data-testid="project-switcher-status"
        >{projectsStore.statusLabel(current.status)}</span
      >
    {/if}
    {#if paused}
      <span
        class="shrink-0 rounded bg-amber-950/60 px-1 text-[10px] uppercase tracking-wide text-amber-300"
        title={pausedTitle}
        data-testid="project-switcher-paused">paused</span
      >
    {/if}
    {#if customKeys}
      <span
        class="shrink-0 rounded bg-zinc-800 px-1 text-[10px] text-zinc-300"
        title="This project's keyboard shortcuts differ from the defaults"
        data-testid="project-switcher-custom-keys">custom keys</span
      >
    {/if}
    <span class="shrink-0 text-zinc-500" aria-hidden="true">▾</span>
  </button>

  {#if open}
    <div
      class="absolute left-0 top-full z-50 mt-1 w-72 rounded border border-zinc-700 bg-zinc-900 py-1 shadow-xl"
      role="listbox"
      aria-label="Projects"
      tabindex="-1"
      data-testid="project-switcher-menu"
      onkeydown={onKeydown}
    >
      {#each options as p (p.slug)}
        <button
          type="button"
          role="option"
          aria-selected={p.slug === current?.slug}
          class="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-zinc-800 {p.slug ===
          current?.slug
            ? 'text-white'
            : 'text-zinc-300'}"
          data-testid="project-option-{p.slug}"
          onclick={() => choose(p.slug)}
        >
          <span class="min-w-0 flex-1">
            <span class="block truncate">{p.display_name}</span>
            <span class="block truncate font-mono text-[11px] text-zinc-500"
              >{p.slug}</span
            >
          </span>
          {#if p.status !== 'active'}
            <span
              class="shrink-0 rounded bg-amber-950/60 px-1 text-[10px] uppercase tracking-wide text-amber-300"
              >{projectsStore.statusLabel(p.status)}</span
            >
          {/if}
          {#if p.slug === current?.slug}
            <span class="shrink-0 text-xs text-zinc-500">current</span>
          {/if}
        </button>
      {/each}
      <div class="my-1 border-t border-zinc-800"></div>
      <a
        href={resolve('/projects')}
        class="block px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800 hover:text-white"
        data-testid="project-switcher-manage"
        onclick={() => (open = false)}>Manage projects…</a
      >
    </div>
  {/if}
</div>
