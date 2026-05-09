<script lang="ts">
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';

  const shortcuts = $derived(keyboardStore.shortcutsForCurrentScope());

  // Per-class hotkeys are routed through the layout-level keydown
  // listener (not registered with keyboardStore) so the user wouldn't
  // see them in the page-scope list. Surface them as their own panel.
  const classHotkeys = $derived(
    classesStore.classes
      .filter((c) => !!c.hotkey_letter)
      .sort((a, b) => (a.hotkey_letter ?? '').localeCompare(b.hotkey_letter ?? '')),
  );

  // Friendly label for the page-scope tag in the header.
  const scopeLabel = $derived(
    keyboardStore.scope === 'cluster'
      ? 'Cluster'
      : keyboardStore.scope === 'review'
        ? 'Review'
        : keyboardStore.scope === 'classes'
          ? 'Classes'
          : keyboardStore.scope === 'global'
            ? 'Global'
            : keyboardStore.scope,
  );
</script>

{#if keyboardStore.overlayOpen}
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Keyboard shortcuts"
    tabindex="-1"
    onclick={() => keyboardStore.closeOverlay()}
    onkeydown={(e) => e.key === 'Escape' && keyboardStore.closeOverlay()}
  >
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      role="document"
      class="max-h-[85vh] w-full max-w-3xl overflow-y-auto rounded-lg border border-zinc-700 bg-zinc-950 p-6 shadow-2xl"
      onclick={(e) => e.stopPropagation()}
      onkeydown={(e) => e.stopPropagation()}
      tabindex="-1"
    >
      <div class="mb-4 flex items-center justify-between">
        <h2 class="text-lg font-semibold text-white">Keyboard Shortcuts</h2>
        <span class="rounded-full border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-[11px] text-zinc-300">
          {scopeLabel} page
        </span>
      </div>

      <!-- Page-scope shortcuts (registered via keyboardStore) -->
      <section class="mb-5">
        <h3 class="mb-2 text-xs font-semibold tracking-wide text-zinc-400 uppercase">
          {scopeLabel} actions
        </h3>
        {#if shortcuts.length === 0}
          <p class="text-sm text-zinc-500">No shortcuts on this page.</p>
        {:else}
          <ul class="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {#each shortcuts as s (s.scope + ':' + s.key)}
              <li class="flex items-center justify-between gap-3 text-sm">
                <span class="text-zinc-300">{s.description}</span>
                <kbd class="font-mono text-[11px]">{s.key}</kbd>
              </li>
            {/each}
          </ul>
        {/if}
      </section>

      <!-- Per-class label hotkeys (set on /classes; work on every page
           that registers a dropOnClass dispatcher — currently /clusters/[id]
           and /review). -->
      <section class="mb-5">
        <h3 class="mb-2 flex items-center gap-2 text-xs font-semibold tracking-wide text-zinc-400 uppercase">
          Class hotkeys
          <span class="text-[10px] font-normal normal-case text-zinc-600">
            (configure on /classes)
          </span>
        </h3>
        {#if classHotkeys.length === 0}
          <p class="text-sm text-zinc-500">
            No class hotkeys bound. Visit
            <a href="/classes" class="text-blue-400 hover:underline">/classes</a>
            and assign a single letter to your most-used classes.
          </p>
        {:else}
          <ul class="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {#each classHotkeys as cls (cls.id)}
              <li class="flex items-center justify-between gap-3 text-sm">
                <span class="truncate text-zinc-300">
                  {cls.name}
                  {#if cls.group}<span class="text-zinc-500"> · {cls.group}</span>{/if}
                </span>
                <kbd class="font-mono text-[11px] uppercase text-blue-300">
                  {cls.hotkey_letter}
                </kbd>
              </li>
            {/each}
          </ul>
        {/if}
      </section>

      <!-- Always-on shortcuts (built into the keyboardStore dispatcher). -->
      <section>
        <h3 class="mb-2 text-xs font-semibold tracking-wide text-zinc-400 uppercase">
          Always
        </h3>
        <ul class="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
          <li class="flex items-center justify-between gap-3 text-sm">
            <span class="text-zinc-300">Toggle this panel</span>
            <kbd class="font-mono text-[11px]">~</kbd>
          </li>
          <li class="flex items-center justify-between gap-3 text-sm">
            <span class="text-zinc-300">Close this panel / cancel</span>
            <kbd class="font-mono text-[11px]">Esc</kbd>
          </li>
        </ul>
      </section>
    </div>
  </div>
{/if}
