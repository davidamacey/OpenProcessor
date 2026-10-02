<script lang="ts">
  /**
   * Step 1 — sources and target. Sources come from the served project
   * list (`selectable` and `status === 'active'` only), ordered by
   * priority with up/down buttons (the first source wins a label
   * conflict and donates the image of a duplicate). The slug pattern is a
   * hint only: the preview's `slug_*` errors are the gate.
   */
  import type { CombineWizard } from '$lib/combine/combineWizardController.svelte';
  import type { CombineLabelStates } from '$lib/types_combine';
  import type { ProjectSummary } from '$lib/types_projects';
  import { combineLabel } from '$lib/combine/combineText';

  interface Props {
    wizard: CombineWizard;
    candidates: ProjectSummary[];
    slugPattern?: string | null;
  }
  let { wizard, candidates, slugPattern = null }: Props = $props();

  const LABEL_STATES: CombineLabelStates[] = ['all', 'validated_only'];
  const available = $derived(candidates.filter((p) => !wizard.hasSource(p.slug)));

  function nameOf(slug: string): string {
    return candidates.find((p) => p.slug === slug)?.display_name ?? slug;
  }
</script>

<section class="space-y-3" data-testid="combine-step-sources">
  <h2 class="text-sm font-semibold text-zinc-200">1. Sources and target</h2>

  <ol class="space-y-1" data-testid="combine-source-list">
    {#each wizard.sources as s, i (s.project)}
      <li
        class="flex flex-wrap items-center gap-2 rounded border border-zinc-800 bg-zinc-900/40 px-2 py-1.5 text-sm"
        data-testid="combine-source-{s.project}"
      >
        <span class="w-5 text-right text-xs text-zinc-500">{i + 1}.</span>
        <span class="text-zinc-100">{nameOf(s.project)}</span>
        <span class="font-mono text-[11px] text-zinc-500">{s.project}</span>
        <span class="grow"></span>
        <label class="flex items-center gap-1 text-xs text-zinc-400">
          Labels
          <select
            class="rounded border border-zinc-700 bg-zinc-900 px-1 py-0.5 text-xs text-zinc-100"
            data-testid="combine-label-states-{s.project}"
            value={s.label_states ?? 'all'}
            onchange={(e) =>
              wizard.setLabelStates(
                s.project,
                (e.currentTarget as HTMLSelectElement).value as CombineLabelStates,
              )}
          >
            {#each LABEL_STATES as v (v)}
              <option value={v}>{combineLabel(v)}</option>
            {/each}
          </select>
        </label>
        <button
          type="button"
          class="btn btn-sm"
          aria-label="Move {s.project} up"
          data-testid="combine-source-up-{s.project}"
          disabled={i === 0}
          onclick={() => wizard.moveSource(s.project, -1)}>Up</button
        >
        <button
          type="button"
          class="btn btn-sm"
          aria-label="Move {s.project} down"
          data-testid="combine-source-down-{s.project}"
          disabled={i === wizard.sources.length - 1}
          onclick={() => wizard.moveSource(s.project, 1)}>Down</button
        >
        <button
          type="button"
          class="btn btn-sm text-red-300"
          data-testid="combine-source-remove-{s.project}"
          onclick={() => wizard.removeSource(s.project)}>Remove</button
        >
      </li>
    {:else}
      <li class="text-xs text-zinc-500">No sources yet. Add at least one.</li>
    {/each}
  </ol>
  {#if wizard.sources.length > 1}
    <p class="text-xs text-zinc-500">
      Order is priority: the first source wins a label conflict and donates the image of a
      duplicate.
    </p>
  {/if}

  <label class="block text-sm">
    <span class="mb-1 block text-xs text-zinc-400">Add a source project</span>
    <select
      class="w-full max-w-xs rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
      data-testid="combine-add-source"
      value=""
      disabled={available.length === 0}
      onchange={(e) => {
        const el = e.currentTarget as HTMLSelectElement;
        if (el.value) wizard.addSource(el.value);
        el.value = '';
      }}
    >
      <option value="">{available.length === 0 ? 'No more projects' : 'Choose…'}</option>
      {#each available as p (p.slug)}
        <option value={p.slug}>{p.display_name} ({p.slug})</option>
      {/each}
    </select>
  </label>

  <div class="grid gap-3 sm:grid-cols-2">
    <label class="block text-sm">
      <span class="mb-1 block text-xs text-zinc-400">Target slug</span>
      <input
        type="text"
        class="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 font-mono text-sm text-zinc-100"
        data-testid="combine-target-slug"
        autocomplete="off"
        value={wizard.slug}
        oninput={(e) =>
          wizard.setTarget({ slug: (e.currentTarget as HTMLInputElement).value })}
      />
      {#if slugPattern}
        <span
          class="mt-0.5 block text-[11px] text-zinc-500"
          data-testid="combine-slug-hint"
          >Pattern: <code class="font-mono">{slugPattern}</code></span
        >
      {/if}
    </label>
    <label class="block text-sm">
      <span class="mb-1 block text-xs text-zinc-400">Display name</span>
      <input
        type="text"
        class="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
        data-testid="combine-target-name"
        value={wizard.displayName}
        oninput={(e) =>
          wizard.setTarget({ displayName: (e.currentTarget as HTMLInputElement).value })}
      />
    </label>
  </div>
  <label class="block text-sm">
    <span class="mb-1 block text-xs text-zinc-400">Description (optional)</span>
    <input
      type="text"
      class="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
      data-testid="combine-target-description"
      value={wizard.description}
      oninput={(e) =>
        wizard.setTarget({ description: (e.currentTarget as HTMLInputElement).value })}
    />
  </label>
</section>
