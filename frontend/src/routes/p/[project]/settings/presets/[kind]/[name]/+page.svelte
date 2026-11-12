<script lang="ts">
  /**
   * `/settings/presets/<kind>/<name>`: one preset document. The body is edited as JSON; Save sends
   * `expected_revision` and the server judges the body (a refusal lists each field), a stale save is
   * the served 409 with "Reload" and "Keep my edits", and Activate asks the server to move the
   * active ref.
   */
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import ConfigDescriptionField from '$components/config/ConfigDescriptionField.svelte';
  import ConfigSavePanel from '$components/config/ConfigSavePanel.svelte';
  import PresetActivateDialog from '$components/presets/PresetActivateDialog.svelte';
  import PresetActiveSummary from '$components/presets/PresetActiveSummary.svelte';
  import { createPresetEditor } from '$lib/presets/presetEditorController.svelte';
  import { presetKindBySlug, presetPath } from '$lib/presets/presetKinds';
  import { projectHref } from '$lib/projectPaths';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const kind = $derived(presetKindBySlug(page.params.kind));
  const name = $derived(page.params.name ?? '');
  const ed = $derived(kind ? createPresetEditor(kind, name) : null);

  $effect(() => {
    const e = ed;
    if (!e) return;
    e.start();
    return () => e.stop();
  });

  let activating = $state(false);

  async function doSave(): Promise<void> {
    if (ed && (await ed.save())) toastStore.success(`Saved revision ${ed.doc?.revision}`);
  }
</script>

<div class="mx-auto flex w-full max-w-7xl flex-col gap-4 p-6">
  {#if !kind || !ed}
    <p class="text-sm text-zinc-400" data-testid="preset-unknown-kind">
      There is no such kind of preset.
    </p>
    <a
      class="text-xs text-blue-300 hover:underline"
      href={resolve(projectHref('/settings'))}>Back to settings</a
    >
  {:else}
    <header class="flex flex-wrap items-baseline gap-3">
      <h1 class="font-mono text-2xl font-semibold tracking-tight">{name}</h1>
      {#if ed.doc}
        <span class="text-sm text-zinc-400" data-testid="config-meta">
          {kind.noun} · revision {ed.doc.revision}
        </span>
        {#each ed.active.entries.filter((e) => e.state === 'active' && e.name === name) as e (e.axis ?? 'deployment')}
          <span
            class="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 text-xs text-emerald-300"
            data-testid="preset-active-chip"
            >active{kind.axes.find((a) => a.id === e.axis)
              ? ` · ${kind.axes.find((a) => a.id === e.axis)!.label}`
              : ''} r{e.revision}</span
          >
        {/each}
      {/if}
      <span class="grow"></span>
      <a
        class="text-xs text-blue-300 hover:underline"
        href={resolve(projectHref(presetPath(kind)))}>All {kind.title.toLowerCase()}</a
      >
    </header>

    {#if ed.loadError && !ed.doc}
      <p class="text-sm text-red-300" data-testid="preset-load-error">{ed.loadError}</p>
    {:else if !ed.doc}
      <p class="text-sm text-zinc-500">Loading…</p>
    {:else}
      <PresetActiveSummary active={ed.active} />

      <div class="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <section class="surface flex min-w-0 flex-col gap-3 p-4" aria-label="Body">
          <ConfigDescriptionField {ed} />
          <label class="flex flex-col gap-1 text-xs text-zinc-400">
            Body (JSON)
            <textarea
              class="input min-h-96 font-mono text-xs"
              spellcheck="false"
              value={ed.bodyText}
              oninput={(e) => ed.setBodyText(e.currentTarget.value)}
              data-testid="preset-body"></textarea>
          </label>
          {#if ed.bodyError}
            <p class="text-xs text-amber-300" data-testid="preset-body-error">
              {ed.bodyError}
            </p>
          {/if}
          <div>
            <button
              type="button"
              class="btn btn-sm"
              disabled={ed.bodyError !== null}
              onclick={() => ed.format()}>Format</button
            >
          </div>
        </section>

        <ConfigSavePanel
          {ed}
          report={null}
          noun={kind.noun}
          onsave={() => void doSave()}
          onactivate={() => {
            ed.active.clearAction();
            activating = true;
          }}
        />
      </div>

      {#if activating && ed.doc}
        <PresetActivateDialog
          active={ed.active}
          target={{ name, revision: ed.doc.revision }}
          unsaved={ed.dirty}
          onclose={() => (activating = false)}
          onactivated={(t) => toastStore.success(`Activated ${t.name} r${t.revision}`)}
        />
      {/if}
    {/if}
  {/if}
</div>
