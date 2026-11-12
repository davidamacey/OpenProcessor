<script lang="ts">
  import { resolve } from '$app/paths';
  import Modal from '$components/Modal.svelte';
  import { projectHref } from '$lib/projectPaths';
  import { setClassHotkey } from '$lib/classHotkey';
  import { isAssignableClass } from '$lib/classVisibility';
  import { formatShortcutKey } from '$lib/keyboardDisplay';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { keymapAvailability, keymapStore } from '$stores/keymap.svelte';
  import type { RegistryClass } from '$lib/types_classes';

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
  <Modal
    label="Keyboard shortcuts"
    onclose={() => keyboardStore.closeOverlay()}
    layer="top"
    scrim="bg-black/70"
    panelClass="max-h-[88vh] w-full max-w-3xl overflow-y-auto rounded-lg border border-line-strong bg-surface-0 p-6 shadow-2xl"
  >
    <div class="mb-4 flex items-center justify-between">
      <h2 class="text-lg font-semibold text-white">Keyboard Shortcuts</h2>
      <div class="flex items-center gap-2">
        {#if keymapAvailability.available !== false}
          <a
            href={resolve(projectHref('/settings#keyboard'))}
            class="text-caption text-accent-text hover:text-accent-text hover:underline"
            onclick={() => keyboardStore.closeOverlay()}
          >
            Edit shortcuts…
          </a>
        {/if}
        <span
          class="rounded-full border border-line-strong bg-surface-1 px-2 py-0.5 text-caption text-fg-2"
        >
          {scopeLabel} page
        </span>
      </div>
    </div>

    <section class="mb-5">
      <h3 class="mb-2 text-xs font-semibold tracking-wide text-fg-2 uppercase">
        {scopeLabel} actions
      </h3>
      {#if shortcuts.length === 0}
        <p class="text-sm text-fg-3">No shortcuts on this page.</p>
      {:else}
        <ul class="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
          {#each shortcuts as s (s.scope + ':' + s.description)}
            <li class="flex items-center justify-between gap-3 text-sm">
              <span class="text-fg-2">{s.description}</span>
              <span class="flex gap-1">
                {#each s.keys as k (k)}
                  <kbd class="font-mono text-caption">{formatShortcutKey(k)}</kbd>
                {/each}
              </span>
            </li>
          {/each}
        </ul>
      {/if}
    </section>

    <section class="mb-5">
      <h3
        class="mb-2 flex items-center gap-2 text-xs font-semibold tracking-wide text-fg-2 uppercase"
      >
        Class hotkeys
        <span class="text-caption font-normal normal-case text-fg-3">
          (single letter; tab to next class; Enter to save)
        </span>
      </h3>
      {#if editableClasses.length === 0}
        <p class="text-sm text-fg-3">No classes defined yet.</p>
      {:else}
        <ul class="grid grid-cols-1 gap-y-1 sm:grid-cols-2 sm:gap-x-4">
          {#each editableClasses as cls (cls.id)}
            <li class="flex items-center justify-between gap-3 text-sm">
              <span
                class="truncate text-fg-2"
                title={cls.group ? `${cls.group} / ${cls.name}` : cls.name}
              >
                {cls.name}
                {#if cls.validated_count != null}
                  <span class="ml-1 font-mono text-caption text-fg-3">
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
                    (e.currentTarget as HTMLInputElement).value = cls.hotkey_letter ?? '';
                    (e.currentTarget as HTMLInputElement).blur();
                  }
                }}
                onblur={(e) => {
                  void setHotkey(cls, (e.currentTarget as HTMLInputElement).value);
                }}
                class="w-10 rounded border border-line-strong bg-surface-1 px-1.5 py-0.5 text-center font-mono text-xs uppercase text-accent-text focus:border-accent focus:outline-none disabled:opacity-50"
                aria-label={`hotkey for ${cls.name}`}
              />
            </li>
          {/each}
        </ul>
      {/if}
    </section>

    <section>
      <h3 class="mb-2 text-xs font-semibold tracking-wide text-fg-2 uppercase">Always</h3>
      <ul class="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
        <li class="flex items-center justify-between gap-3 text-sm">
          <span class="text-fg-2">{keymapStore.label('global.shortcuts_overlay')}</span>
          <kbd class="font-mono text-caption"
            >{keymapStore.glyph('global.shortcuts_overlay')}</kbd
          >
        </li>
        <li class="flex items-center justify-between gap-3 text-sm">
          <span class="text-fg-2">{keymapStore.label('global.close_overlay')}</span>
          <kbd class="font-mono text-caption"
            >{keymapStore.glyph('global.close_overlay')}</kbd
          >
        </li>
      </ul>
    </section>
  </Modal>
{/if}
