<script lang="ts">
  import PageHeader from '$components/ui/PageHeader.svelte';
  /**
   * `/settings/embedding-spaces`: the project's embedding spaces per target, the encoders the
   * deployment can serve, and the writes that register, restart or backfill, promote, retire and
   * cancel. Every build, wherever it was started, is followed through the served re-index run. The
   * comparison of a space with the primary is the served evaluation; this page computes none.
   */
  import { resolve } from '$app/paths';
  import EmbeddingModelsTable from '$components/embeddingSpaces/EmbeddingModelsTable.svelte';
  import EmbeddingSpacesTable from '$components/embeddingSpaces/EmbeddingSpacesTable.svelte';
  import SpaceActionDialog from '$components/embeddingSpaces/SpaceActionDialog.svelte';
  import SpaceEvaluationDialog from '$components/embeddingSpaces/SpaceEvaluationDialog.svelte';
  import SpaceRegisterDialog from '$components/embeddingSpaces/SpaceRegisterDialog.svelte';
  import {
    createEmbeddingSpaces,
    type SpaceAction,
  } from '$lib/embeddingSpaces/embeddingSpacesController.svelte';
  import { projectHref } from '$lib/projectPaths';
  import { EMBEDDING_TARGETS, type EmbeddingSpaceView } from '$lib/types_embeddingSpaces';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const st = createEmbeddingSpaces();
  $effect(() => {
    void st.load();
    return () => st.stop();
  });

  let registering = $state(false);
  let acting = $state<{ action: SpaceAction; space: EmbeddingSpaceView } | null>(null);
  let viewing = $state<EmbeddingSpaceView | null>(null);

  const DONE: Record<SpaceAction, string> = {
    build: 'Build started',
    promote: 'Promoted',
    retire: 'Retiring',
    cancel: 'Build cancelled',
  };
</script>

<div class="flex flex-col gap-4">
  <PageHeader title="Embedding spaces">
    {#snippet actions()}
      <button type="button" class="btn btn-sm" onclick={() => void st.load()}
        >Refresh</button
      >
      <button
        type="button"
        class="btn btn-sm btn-primary"
        data-testid="space-register"
        onclick={() => {
          st.clearRefusal();
          registering = true;
        }}>Register a space</button
      >
      <a
        class="text-meta text-accent-text hover:underline"
        href={resolve(projectHref('/settings'))}>Back to settings</a
      >
    {/snippet}
  </PageHeader>
  <p class="text-body text-fg-2">
    A space is one set of vectors for a target made by one encoder version. A project
    always has a primary space; register a candidate to build another, then promote it
    when the server allows.
  </p>

  {#if st.loadError && !st.spaces}
    <p class="text-body text-danger-text" data-testid="space-load-error">
      {st.loadError}
    </p>
  {:else if !st.spaces}
    <p class="text-body text-fg-3">Loading…</p>
  {:else}
    {@const all = st.spaces}
    {#each EMBEDDING_TARGETS as t (t)}
      {@const rows = all.filter((s) => s.target === t)}
      {#if rows.length > 0}
        <EmbeddingSpacesTable
          state={st}
          target={t}
          spaces={rows}
          onaction={(action, space) => {
            st.clearRefusal();
            acting = { action, space };
          }}
          onevaluation={(space) => (viewing = space)}
        />
      {/if}
    {/each}
    {#if all.length === 0}
      <p class="text-body text-fg-2" data-testid="space-empty">
        This project has no spaces yet.
      </p>
    {/if}
  {/if}

  {#if st.models}
    <EmbeddingModelsTable models={st.models} />
  {:else if st.modelsError}
    <p class="text-body text-danger-text" data-testid="models-error">{st.modelsError}</p>
  {/if}
</div>

{#if registering}
  <SpaceRegisterDialog
    state={st}
    onregistered={(what) => {
      registering = false;
      toastStore.success(`Registered ${what}; the build started`);
    }}
    oncancel={() => (registering = false)}
  />
{/if}

{#if viewing}
  <SpaceEvaluationDialog state={st} space={viewing} onclose={() => (viewing = null)} />
{/if}

{#if acting}
  <SpaceActionDialog
    state={st}
    action={acting.action}
    space={acting.space}
    ondone={(action, space) => {
      acting = null;
      toastStore.success(`${DONE[action]}: ${space.space_key}`);
    }}
    oncancel={() => (acting = null)}
  />
{/if}
