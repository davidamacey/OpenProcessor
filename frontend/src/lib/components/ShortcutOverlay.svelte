<script lang="ts">
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  import { setClassHotkey } from '$lib/classHotkey';
  import { isAssignableClass } from '$lib/classVisibility';
  import { formatShortcutKey } from '$lib/keyboardDisplay';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { keymapAvailability, keymapStore } from '$stores/keymap.svelte';
  import type { RegistryClass } from '$lib/types';

  const shortcuts = $derived(keyboardStore.shortcutsForCurrentScope());

  // Show every non-deprecated class — sorted by validated_count desc so
  // the operator's most-labelled classes float to the top. Lets the
  // overlay double as a hotkey editor without leaving /review or
  // /clusters.
  const editableClasses = $derived(
    classesStore.classes
      .filter(isAssignableClass)
      .slice()
      .sort((a, b) => {
        // Bound hotkeys first, then largest classes
        const aHas = a.hotkey_letter ? 0 : 1;
        const bHas = b.hotkey_letter ? 0 : 1;
        if (aHas !== bHas) return aHas - bHas;
        return (b.validated_count ?? 0) - (a.validated_count ?? 0);
      }),
  );

  let pending = $state<Record<number, boolean>>({});

  async function setHotkey(cls: RegistryClass, raw: string): Promise<void> {
    pending[cls.id] = true;
    try {
      await setClassHotkey(cls, raw);
    } finally {
      pending[cls.id] = false;
    }
  }

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
      class="max-h-[88vh] w-full max-w-3xl overflow-y-auto rounded-lg border border-zinc-700 bg-zinc-950 p-6 shadow-2xl"
      onclick={(e) => e.stopPropagation()}
      onkeydown={(e) => e.stopPropagation()}
      tabindex="-1"
    >
      <div class="mb-4 flex items-center justify-between">
        <h2 class="text-lg font-semibold text-white">Keyboard Shortcuts</h2>
        <div class="flex items-center gap-2">
          {#if keymapAvailability.available !== false}
            <a
              href={resolve(projectHref('/settings#keyboard'))}
              class="text-[11px] text-blue-400 hover:text-blue-300 hover:underline"
              onclick={() => keyboardStore.closeOverlay()}
            >
              Edit shortcuts…
            </a>
          {/if}
          <span
            class="rounded-full border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-[11px] text-zinc-300"
          >
            {scopeLabel} page
          </span>
        </div>
      </div>

      <section class="mb-5">
        <h3 class="mb-2 text-xs font-semibold tracking-wide text-zinc-400 uppercase">
          {scopeLabel} actions
        </h3>
        {#if shortcuts.length === 0}
          <p class="text-sm text-zinc-500">No shortcuts on this page.</p>
        {:else}
          <ul class="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {#each shortcuts as s (s.scope + ':' + s.description)}
              <li class="flex items-center justify-between gap-3 text-sm">
                <span class="text-zinc-300">{s.description}</span>
                <span class="flex gap-1">
                  {#each s.keys as k (k)}
                    <kbd class="font-mono text-[11px]">{formatShortcutKey(k)}</kbd>
                  {/each}
                </span>
              </li>
            {/each}
          </ul>
        {/if}
      </section>

      <section class="mb-5">
        <h3
          class="mb-2 flex items-center gap-2 text-xs font-semibold tracking-wide text-zinc-400 uppercase"
        >
          Class hotkeys
          <span class="text-[10px] font-normal normal-case text-zinc-600">
            (single letter; tab to next class; Enter to save)
          </span>
        </h3>
        {#if editableClasses.length === 0}
          <p class="text-sm text-zinc-500">No classes defined yet.</p>
        {:else}
          <ul class="grid grid-cols-1 gap-y-1 sm:grid-cols-2 sm:gap-x-4">
            {#each editableClasses as cls (cls.id)}
              <li class="flex items-center justify-between gap-3 text-sm">
                <span
                  class="truncate text-zinc-300"
                  title={cls.group ? `${cls.group} / ${cls.name}` : cls.name}
                >
                  {cls.name}
                  {#if cls.validated_count != null}
                    <span class="ml-1 font-mono text-[10px] text-zinc-500">
                      {cls.validated_count.toLocaleString()}
                    </span>
                  {/if}
                </span>
                <input
                  type="text"
                  maxlength="1"
                  value={cls.hotkey_letter ?? ''}
                  disabled={pending[cls.id]}
                  onkeydown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      (e.currentTarget as HTMLInputElement).blur();
                    } else if (e.key === 'Escape') {
                      e.preventDefault();
                      (e.currentTarget as HTMLInputElement).value =
                        cls.hotkey_letter ?? '';
                      (e.currentTarget as HTMLInputElement).blur();
                    }
                  }}
                  onblur={(e) => {
                    void setHotkey(cls, (e.currentTarget as HTMLInputElement).value);
                  }}
                  class="w-10 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-center font-mono text-xs uppercase text-blue-300 focus:border-blue-500 focus:outline-none disabled:opacity-50"
                  aria-label={`hotkey for ${cls.name}`}
                />
              </li>
            {/each}
          </ul>
        {/if}
      </section>

      <section>
        <h3 class="mb-2 text-xs font-semibold tracking-wide text-zinc-400 uppercase">
          Always
        </h3>
        <ul class="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
          <li class="flex items-center justify-between gap-3 text-sm">
            <span class="text-zinc-300"
              >{keymapStore.label('global.shortcuts_overlay')}</span
            >
            <kbd class="font-mono text-[11px]"
              >{keymapStore.glyph('global.shortcuts_overlay')}</kbd
            >
          </li>
          <li class="flex items-center justify-between gap-3 text-sm">
            <span class="text-zinc-300">{keymapStore.label('global.close_overlay')}</span>
            <kbd class="font-mono text-[11px]"
              >{keymapStore.glyph('global.close_overlay')}</kbd
            >
          </li>
        </ul>
      </section>
    </div>
  </div>
{/if}
