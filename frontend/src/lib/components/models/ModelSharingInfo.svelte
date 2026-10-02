<script lang="ts">
  /**
   * The sharing and class-mapping part of one `/models` card (projects P2,
   * §5.5), all from served fields:
   *
   * - another project's shared model: a project chip (the served project
   *   list's display name for that slug, else the slug) and "shared";
   * - the active project's own promoted model (served `owned`): its served
   *   sharing state and, when the served sharing revision is present, the
   *   owner-only toggle (`onshare` opens the confirm dialog);
   * - the served `class_mapping`: "N classes map", the served unmapped
   *   names, and a lazily loaded name-by-name mapping. Class ids never
   *   cross projects, so none is rendered.
   */
  import {
    canToggleSharing,
    mappingText,
    sharingRole,
    unmappedText,
  } from '$lib/modelSharing';
  import type { ModelSharing } from '$lib/models/modelSharingController.svelte';
  import type { ModelInfo } from '$lib/types';

  interface Props {
    model: ModelInfo;
    /** Display name for a served project slug, when the list carries it. */
    projectName: (slug: string) => string;
    sharing: ModelSharing;
    onshare: (m: ModelInfo) => void;
  }
  let { model, projectName, sharing, onshare }: Props = $props();

  const role = $derived(sharingRole(model));
  const mappingState = $derived(sharing.mapping(model.name));
</script>

{#if role !== 'none' || model.class_mapping}
  <div
    class="mt-3 space-y-2 border-t border-zinc-800 pt-3 text-xs"
    data-testid="model-sharing-{model.name}"
  >
    {#if role === 'foreign' && model.project}
      <div class="flex flex-wrap items-center gap-1.5">
        <span
          class="rounded border border-sky-800 bg-sky-950/50 px-1.5 py-0.5 text-sky-200"
          title="Promoted in project {model.project}"
          data-testid="model-project-chip">from {projectName(model.project)}</span
        >
        <span
          class="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400"
          >shared</span
        >
      </div>
    {:else if role === 'owner'}
      <div class="flex flex-wrap items-center justify-between gap-2">
        <span class="text-zinc-400" data-testid="model-sharing-state"
          >{model.shared ? 'Shared with other projects' : 'Not shared'}</span
        >
        {#if canToggleSharing(model)}
          <button
            type="button"
            class="btn btn-sm"
            data-testid="model-share-toggle-{model.name}"
            disabled={sharing.pending === model.name}
            onclick={() => onshare(model)}
            >{model.shared ? 'Stop sharing' : 'Share with other projects'}</button
          >
        {/if}
      </div>
    {/if}

    {#if model.class_mapping}
      <div data-testid="model-class-mapping">
        <span class="text-zinc-300" data-testid="model-mapped-count"
          >{mappingText(model.class_mapping)}</span
        >
        {#if model.class_mapping.unmapped.length}
          <details class="mt-1" data-testid="model-unmapped">
            <summary class="cursor-pointer text-amber-300"
              >{unmappedText(model.class_mapping)}</summary
            >
            <p class="mt-1 break-words text-zinc-400" data-testid="model-unmapped-names">
              {model.class_mapping.unmapped.join(', ')}
            </p>
          </details>
        {/if}
        <details
          class="mt-1"
          data-testid="model-mapping-details"
          ontoggle={(e) => {
            if ((e.currentTarget as HTMLDetailsElement).open)
              void sharing.loadMapping(model.name);
          }}
        >
          <summary class="cursor-pointer text-zinc-400 hover:text-zinc-200"
            >Class mapping</summary
          >
          {#if mappingState?.status === 'loading'}
            <p class="mt-1 text-zinc-500">Loading…</p>
          {:else if mappingState?.status === 'error'}
            <p class="mt-1 text-red-300" data-testid="model-mapping-error">
              {mappingState.message}
            </p>
          {:else if mappingState?.status === 'ok'}
            {@const m = mappingState.mapping}
            <table class="mt-1 w-full" data-testid="model-mapping-table">
              <thead class="text-left text-[10px] uppercase tracking-wide text-zinc-500">
                <tr>
                  <th class="py-0.5 pr-2 font-normal">Model class</th>
                  <th class="py-0.5 pr-2 font-normal">This project</th>
                  <th class="py-0.5 font-normal">Match</th>
                </tr>
              </thead>
              <tbody>
                {#each m.entries as e (e.model_id)}
                  <tr class="border-t border-zinc-800/60">
                    <td class="py-0.5 pr-2 text-zinc-200">{e.model_name}</td>
                    <td
                      class="py-0.5 pr-2 {e.class_name
                        ? 'text-zinc-200'
                        : 'text-zinc-500'}">{e.class_name ?? '—'}</td
                    >
                    <td class="py-0.5 text-zinc-400"
                      >{m.labels?.match?.[e.match] ?? e.match}</td
                    >
                  </tr>
                {/each}
              </tbody>
            </table>
            {#if m.not_covered.length}
              <p class="mt-1 text-zinc-500" data-testid="model-not-covered">
                Not predicted by this model: {m.not_covered.join(', ')}
              </p>
            {/if}
          {/if}
        </details>
      </div>
    {/if}
  </div>
{/if}
