<script lang="ts">
  /**
   * `/settings/models` (OpenProcessor W9; any_domain_plan.md §7.8.5;
   * docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md §3.3): which VLM this
   * project uses, the deployment-wide endpoint registry, the local-model
   * catalog and every model choice the deployment exposes. Everything
   * listed is served; `VlmModels` follows it. Absent (one line) until the
   * backend serves W9.
   */
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import ConfigActivateDialog from '$components/config/ConfigActivateDialog.svelte';
  import ConfigActivePanel from '$components/config/ConfigActivePanel.svelte';
  import ConfigCloneDialog from '$components/config/ConfigCloneDialog.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import LocalVlmPanel from '$components/vlm/LocalVlmPanel.svelte';
  import ModelChoicesTable from '$components/vlm/ModelChoicesTable.svelte';
  import VlmEndpointsTable from '$components/vlm/VlmEndpointsTable.svelte';
  import type { CloneSource } from '$lib/config/configList.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { VlmEndpointSummary } from '$lib/types_vlm';
  import { VLM_ACTIVE_COPY } from '$lib/vlm/vlmCopy';
  import { vlmAvailability } from '$lib/vlm/vlmAvailability.svelte';
  import { createVlmModels } from '$lib/vlm/vlmModelsController.svelte';
  import { healthStore } from '$stores/health.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { projectsStore } from '$stores/projects.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const models = createVlmModels();

  $effect(() => {
    if (vlmAvailability.available !== true) return;
    models.start();
    return () => models.stop();
  });

  let activating = $state<VlmEndpointSummary | null>(null);
  let cloneFrom = $state<CloneSource | null>(null);
  let deleting = $state<VlmEndpointSummary | null>(null);

  const health = $derived(healthStore.scopedHealth?.vlm ?? null);

  function openActivate(row: VlmEndpointSummary): void {
    models.active.clearAction();
    activating = row;
  }

  function openClone(row: VlmEndpointSummary): void {
    models.clearClone();
    cloneFrom = { name: row.name, source: null };
  }

  async function doClone(name: string, description: string): Promise<void> {
    if (!cloneFrom) return;
    const doc = await models.clone(cloneFrom, name, description);
    if (!doc) return;
    cloneFrom = null;
    toastStore.success(`Created ${doc.name}`);
    await goto(
      resolve(projectHref(`/settings/models/vlm/${encodeURIComponent(doc.name)}`)),
    );
  }

  async function doDelete(): Promise<void> {
    const row = deleting;
    if (!row) return;
    if (await models.remove(row)) {
      deleting = null;
      toastStore.success(`Deleted ${row.name}`);
    }
  }
</script>

<div class="mx-auto flex max-w-7xl flex-col gap-4 p-6">
  <header class="flex flex-wrap items-baseline gap-3">
    <h1 class="text-2xl font-semibold tracking-tight">Models</h1>
    <span class="grow"></span>
    <a
      class="text-xs text-blue-300 hover:underline"
      href={resolve(projectHref('/settings'))}>Back to settings</a
    >
  </header>
  <p class="text-sm text-zinc-400">
    The VLM reads crops and proposes labels. Register the endpoints it can run on, test
    them, and choose which one this project uses.
  </p>

  <ConfigGate
    store={vlmAvailability}
    what="VLM models"
    unavailableText="VLM endpoint management is not available on this backend."
    testid="vlm-unavailable"
  >
    <section id="active" class="flex flex-col gap-2">
      <ConfigActivePanel
        ctl={models.active}
        copy={VLM_ACTIVE_COPY}
        sourceLabels={models.list?.labels.source ?? null}
        onrollback={() => models.rollback()}
        ondeactivate={() => models.deactivate()}
      />
      <p class="text-xs text-zinc-400" data-testid="vlm-health">
        {#if health}
          VLM service for this project: {health.reachable ? 'reachable' : 'not reachable'}
          {#if health.model}· model <span class="font-mono">{health.model}</span>{/if}
          {#if health.detail}· {health.detail}{/if}
          {#if health.last_error}
            · <span class="text-red-300">{health.last_error}</span>
          {/if}
        {:else}
          No health read for this project yet.
        {/if}
      </p>
    </section>

    <section id="endpoints" class="surface flex flex-col gap-3 p-4">
      <div class="flex flex-wrap items-center gap-3">
        <h2 class="text-base font-semibold">Endpoints (deployment-wide)</h2>
        <span class="grow"></span>
        <a
          class="btn btn-sm"
          data-testid="vlm-new-endpoint"
          href={resolve(projectHref('/settings/models/new-endpoint'))}>New endpoint</a
        >
      </div>
      {#if models.loadError && !models.list}
        <p class="text-sm text-red-300" data-testid="vlm-load-error">
          {models.loadError}
        </p>
      {:else if !models.list}
        <p class="text-sm text-zinc-500">Loading…</p>
      {:else}
        <VlmEndpointsTable
          list={models.list}
          currentSlug={projectsStore.current?.slug ?? null}
          probes={models.probes}
          probeErrors={models.probeErrors}
          probing={models.probing}
          busy={models.busy || models.active.busy}
          onactivate={openActivate}
          onprobe={(row) => void models.probe(row.name)}
          onclone={openClone}
          ondelete={(row) => {
            models.deleteError = null;
            models.deleteProjects = [];
            deleting = row;
          }}
        />
      {/if}
    </section>

    <section id="local-model" class="surface flex flex-col gap-3 p-4">
      <h2 class="text-base font-semibold">Local model</h2>
      {#if models.catalogError && !models.catalog}
        <p class="text-sm text-red-300" data-testid="vlm-catalog-error">
          {models.catalogError}
        </p>
      {:else if !models.catalog}
        <p class="text-sm text-zinc-500">Loading…</p>
      {:else}
        <LocalVlmPanel
          catalog={models.catalog}
          busy={models.localBusy}
          error={models.localError}
          errorCode={models.localErrorCode}
          onselect={(id, force) => models.selectLocal(id, force)}
          onclear={() => models.clearLocal()}
        />
      {/if}
    </section>

    <section id="model-choices" class="surface flex flex-col gap-3 p-4">
      <h2 class="text-base font-semibold">All model choices</h2>
      {#if models.modelChoicesError}
        <p class="text-sm text-red-300" data-testid="model-choices-error">
          {models.modelChoicesError}
        </p>
      {:else if !models.modelChoices}
        <p class="text-sm text-zinc-500">Loading…</p>
      {:else}
        <ModelChoicesTable
          choices={models.modelChoices}
          scopeLabels={models.modelChoiceLabels}
        />
      {/if}
    </section>
  </ConfigGate>
</div>

{#if activating}
  <ConfigActivateDialog
    ed={{ active: models.active, dirty: false, viewing: null }}
    target={{ name: activating.name, revision: activating.revision, validation: null }}
    ack={{
      warning: activating.warning,
      required: activating.sends_images_externally,
    }}
    blurb="Runs that start after this use this endpoint for this project. Labels the VLM already wrote keep their provenance."
    onclose={() => (activating = null)}
    onactivated={(t) => {
      void models.loadRegistry();
      toastStore.success(
        `Activated ${t.name}${t.revision == null ? '' : ` r${t.revision}`}`,
      );
    }}
  />
{/if}

{#if cloneFrom}
  <ConfigCloneDialog
    title="Clone {cloneFrom.name}"
    nameLabel="New endpoint name"
    withDescription
    busy={models.busy}
    error={models.cloneError}
    report={models.cloneReport}
    onconfirm={(name, description) => void doClone(name, description)}
    oncancel={() => (cloneFrom = null)}
  />
{/if}

{#if deleting}
  <ConfirmDialog
    title="Delete {deleting.name}"
    confirmLabel="Delete"
    danger
    busy={models.busy}
    onconfirm={() => void doDelete()}
    oncancel={() => (deleting = null)}
  >
    <p class="text-xs text-zinc-400">
      Deletes the stored endpoint and its revisions. A project that runs it blocks this.
    </p>
    {#if models.deleteError}
      <p class="text-red-300" data-testid="vlm-delete-error">{models.deleteError}</p>
    {/if}
    {#if models.deleteProjects.length > 0}
      <p class="text-xs text-zinc-300" data-testid="vlm-delete-projects">
        Used by: <span class="font-mono">{models.deleteProjects.join(', ')}</span>
      </p>
    {/if}
  </ConfirmDialog>
{/if}
