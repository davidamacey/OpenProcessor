<script lang="ts">
  /**
   * `/settings/models/vlm/[name]` — one VLM endpoint's editor
   * (any_domain_plan.md §7.8.5; docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md
   * §3.3). The form, its groups, types, ranges and help, every list, every
   * validation issue, the revisions and the probe results are served;
   * `VlmEndpointEditor` holds the draft and follows the server. There is no
   * key input: a key is a host secret, named by reference.
   */
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import ConfigActivateDialog from '$components/config/ConfigActivateDialog.svelte';
  import ConfigActivePanel from '$components/config/ConfigActivePanel.svelte';
  import ConfigCloneDialog from '$components/config/ConfigCloneDialog.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import ConfigRestoreDialog from '$components/config/ConfigRestoreDialog.svelte';
  import ConfigRevisions from '$components/config/ConfigRevisions.svelte';
  import ConfigSavePanel from '$components/config/ConfigSavePanel.svelte';
  import ConfigViewingBanner from '$components/config/ConfigViewingBanner.svelte';
  import VlmEndpointForm from '$components/vlm/VlmEndpointForm.svelte';
  import VlmProbePanel from '$components/vlm/VlmProbePanel.svelte';
  import { createVlmModels } from '$lib/vlm/vlmModelsController.svelte';
  import { projectHref } from '$lib/projectPaths';
  import { VLM_ACTIVE_COPY } from '$lib/vlm/vlmCopy';
  import { vlmAvailability } from '$lib/vlm/vlmAvailability.svelte';
  import { createVlmEndpointEditor } from '$lib/vlm/vlmEndpointEditorController.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import type { VlmEndpointDoc } from '$lib/types_vlm';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const endpointName = $derived(page.params.name ?? '');
  const ed = $derived(createVlmEndpointEditor(endpointName));
  // Clone-to-edit reuses the list controller's clone call.
  const cloner = createVlmModels();

  $effect(() => {
    if (vlmAvailability.available !== true) return;
    const e = ed;
    e.start();
    return () => e.stop();
  });

  /** What the form shows: a revision being viewed, else the draft. */
  const shownDoc = $derived<VlmEndpointDoc | null>(ed.viewing ?? ed.doc);
  const shown = $derived(ed.viewing ? ed.viewing.body : ed.draftBody);
  const shownReport = $derived(ed.viewing ? ed.viewing.validation : ed.report);
  const external = $derived(
    ed.checks.facts?.sends_images_externally ??
      shownDoc?.sends_images_externally ??
      false,
  );
  const activeHere = $derived(ed.active.active?.active.name === endpointName);
  const lastProbe = $derived(ed.probeResult ?? ed.doc?.last_probe ?? null);

  let activating = $state(false);
  let restoring = $state(false);
  let cloning = $state(false);

  async function doSave(): Promise<void> {
    if (await ed.save()) toastStore.success(`Saved revision ${ed.doc?.revision}`);
  }

  async function doClone(name: string, description: string): Promise<void> {
    const doc = await cloner.clone(
      { name: endpointName, source: null },
      name,
      description,
    );
    if (!doc) return;
    cloning = false;
    toastStore.success(`Created ${doc.name}`);
    await goto(
      resolve(projectHref(`/settings/models/vlm/${encodeURIComponent(doc.name)}`)),
    );
  }
</script>

<div class="mx-auto flex max-w-7xl flex-col gap-4 p-6">
  <header class="flex flex-wrap items-baseline gap-3">
    <h1 class="font-mono text-2xl font-semibold tracking-tight">{endpointName}</h1>
    {#if ed.doc}
      {@const d = ed.doc}
      <span class="text-sm text-zinc-400" data-testid="config-meta">
        {d.source}{d.revision != null ? ` · revision ${d.revision}` : ''}{d.read_only
          ? ' · read-only'
          : ''}
      </span>
      {#if activeHere}
        <span
          class="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 text-xs text-emerald-300"
          data-testid="vlm-active-here">active in this project</span
        >
      {/if}
      {#each d.active_in ?? [] as slug (slug)}
        <span
          class="rounded border border-zinc-700 px-1.5 font-mono text-xs text-zinc-300"
          >{slug}</span
        >
      {/each}
      {#if d.cloned_from}
        <span class="font-mono text-xs text-zinc-500">cloned from {d.cloned_from}</span>
      {/if}
    {/if}
    <span class="grow"></span>
    <a
      class="text-xs text-blue-300 hover:underline"
      href={resolve(projectHref('/settings/models'))}>All models</a
    >
  </header>

  <ConfigGate
    store={vlmAvailability}
    what="VLM models"
    unavailableText="VLM endpoint management is not available on this backend."
    testid="vlm-unavailable"
  >
    {#if ed.loadError && !ed.doc}
      <p class="text-sm text-red-300" data-testid="vlm-load-error">{ed.loadError}</p>
    {:else if !ed.doc || !ed.schema}
      <p class="text-sm text-zinc-500">Loading…</p>
    {:else}
      {@const d = ed.doc}
      <ConfigActivePanel
        ctl={ed.active}
        copy={VLM_ACTIVE_COPY}
        sourceLabels={ed.list?.labels.source ?? null}
        onrollback={() => ed.active.rollback()}
        ondeactivate={() => ed.active.deactivate()}
      />

      <div class="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <section
          class="surface flex min-w-0 flex-col gap-4 p-4"
          aria-label="Endpoint fields"
        >
          <ConfigViewingBanner
            {ed}
            onrestore={() => (restoring = true)}
            onactivate={() => (activating = true)}
          />

          {#if d.read_only}
            <div class="flex flex-wrap items-center gap-2 text-sm text-zinc-400">
              <span>This endpoint is read-only. Clone it to make an editable copy.</span>
              <button
                type="button"
                class="btn btn-sm"
                data-testid="vlm-clone-to-edit"
                onclick={() => {
                  cloner.clearClone();
                  cloning = true;
                }}>Clone to edit</button
              >
            </div>
          {/if}

          {#if external}
            <div
              class="rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200"
              data-testid="vlm-external-banner"
            >
              {shownDoc?.warning ?? 'This endpoint sends crops outside this deployment.'}
            </div>
          {/if}

          <label class="flex flex-col gap-1 text-xs text-zinc-400">
            Description
            <input
              class="input input-sm text-sm"
              value={ed.viewing ? ed.viewing.description : ed.draftDescription}
              readonly={!ed.editable}
              oninput={(e) =>
                ed.setDescription((e.currentTarget as HTMLInputElement).value)}
            />
          </label>

          {#if ed.extrasError}
            <p class="text-xs text-red-300" data-testid="vlm-extras-error">
              Could not load the pickers' lists: {ed.extrasError}
            </p>
          {/if}

          <VlmEndpointForm
            schema={ed.schema}
            body={shown}
            report={shownReport}
            list={ed.list}
            catalog={ed.catalog}
            readonly={!ed.editable}
            onchange={(field, value) => ed.setField(field, value)}
          />
        </section>

        <aside
          class="order-first flex min-w-0 flex-col gap-4 lg:sticky lg:top-4 lg:order-none lg:self-start"
        >
          <ConfigSavePanel
            {ed}
            report={shownReport}
            noun="endpoint"
            onsave={() => void doSave()}
            onactivate={() => (activating = true)}
          >
            {#snippet extra()}
              {#if !ed.viewing}
                <div class="flex flex-col gap-2 border-t border-zinc-800 pt-2">
                  <div class="flex flex-wrap gap-2">
                    <button
                      type="button"
                      class="btn btn-sm"
                      disabled={ed.checks.testing}
                      data-testid="vlm-test-connection"
                      onclick={() => void ed.testConnection()}
                      >{ed.checks.testing ? 'Testing…' : 'Test connection'}</button
                    >
                    <button
                      type="button"
                      class="btn btn-sm"
                      disabled={ed.probing}
                      data-testid="vlm-probe-saved"
                      onclick={() => void ed.probeSaved()}
                      >{ed.probing ? 'Probing…' : 'Probe saved'}</button
                    >
                  </div>
                  <p class="text-xs text-zinc-500">
                    Test connection probes the draft; Probe saved probes the saved
                    endpoint.
                  </p>
                  {#if ed.checks.testError}
                    <p class="text-xs text-red-300" data-testid="vlm-test-error">
                      {ed.checks.testError}
                    </p>
                  {/if}
                  {#if ed.probeError}
                    <p class="text-xs text-red-300" data-testid="vlm-probe-error">
                      {ed.probeError}
                    </p>
                  {/if}
                  {#if ed.checks.probe}
                    <VlmProbePanel probe={ed.checks.probe} />
                  {:else if lastProbe}
                    <VlmProbePanel probe={lastProbe} />
                  {/if}
                </div>
              {/if}
            {/snippet}
          </ConfigSavePanel>
          <ConfigRevisions {ed} />
        </aside>
      </div>
    {/if}
  </ConfigGate>
</div>

{#if activating && ed.doc}
  <ConfigActivateDialog
    {ed}
    target={ed.viewing ?? ed.doc}
    ack={{
      warning: (ed.viewing ?? ed.doc).warning,
      required: (ed.viewing ?? ed.doc).sends_images_externally,
    }}
    blurb="Runs that start after this use this endpoint for this project. Labels the VLM already wrote keep their provenance."
    onclose={() => (activating = false)}
    onactivated={(t) =>
      toastStore.success(
        `Activated ${t.name}${t.revision == null ? '' : ` r${t.revision}`}`,
      )}
  />
{/if}

{#if restoring}
  <ConfigRestoreDialog
    {ed}
    noun="endpoint"
    onclose={() => (restoring = false)}
    onrestored={(from, to) =>
      toastStore.success(`Restored revision ${from} as revision ${to}`)}
  />
{/if}

{#if cloning}
  <ConfigCloneDialog
    title="Clone {endpointName}"
    nameLabel="New endpoint name"
    withDescription
    busy={cloner.busy}
    error={cloner.cloneError}
    report={cloner.cloneReport}
    onconfirm={(name, description) => void doClone(name, description)}
    oncancel={() => (cloning = false)}
  />
{/if}
