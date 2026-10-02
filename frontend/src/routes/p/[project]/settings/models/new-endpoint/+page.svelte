<script lang="ts">
  /**
   * `/settings/models/new-endpoint` — register a VLM endpoint
   * (any_domain_plan.md §7.8.5; docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md
   * §3.3). The form is the served schema seeded from each row's default;
   * the name is checked with the draft so a taken or malformed one shows
   * the server's own issue before Create. No key input: a key is a host
   * secret, named by reference.
   */
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import VlmEndpointForm from '$components/vlm/VlmEndpointForm.svelte';
  import VlmProbePanel from '$components/vlm/VlmProbePanel.svelte';
  import { issuesForField } from '$lib/config/validationIssues';
  import { projectHref } from '$lib/projectPaths';
  import { vlmAvailability } from '$lib/vlm/vlmAvailability.svelte';
  import { createVlmEndpointCreator } from '$lib/vlm/vlmEndpointCreateController.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const creator = createVlmEndpointCreator();

  $effect(() => {
    if (vlmAvailability.available !== true) return;
    void creator.load();
    return () => creator.stop();
  });

  const counts = $derived({
    errors: creator.report?.errors.length ?? 0,
    warnings: creator.report?.warnings.length ?? 0,
  });

  async function doCreate(): Promise<void> {
    const doc = await creator.create();
    if (!doc) return;
    toastStore.success(`Created ${doc.name}`);
    await goto(
      resolve(projectHref(`/settings/models/vlm/${encodeURIComponent(doc.name)}`)),
    );
  }
</script>

<div class="mx-auto flex max-w-7xl flex-col gap-4 p-6">
  <header class="flex flex-wrap items-baseline gap-3">
    <h1 class="text-2xl font-semibold tracking-tight">New VLM endpoint</h1>
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
    {#if creator.loadError}
      <p class="text-sm text-red-300" data-testid="vlm-load-error">{creator.loadError}</p>
    {:else if !creator.schema}
      <p class="text-sm text-zinc-500">Loading…</p>
    {:else}
      <div class="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <section
          class="surface flex min-w-0 flex-col gap-4 p-4"
          aria-label="Endpoint fields"
        >
          <label class="flex flex-col gap-1 text-xs text-zinc-400">
            Name
            <input
              class="input input-sm max-w-md font-mono text-sm"
              value={creator.name}
              data-testid="vlm-new-name"
              oninput={(e) =>
                creator.setName((e.currentTarget as HTMLInputElement).value)}
            />
          </label>
          <ConfigIssueList issues={issuesForField(creator.report, 'name')} />
          <label class="flex flex-col gap-1 text-xs text-zinc-400">
            Description
            <input
              class="input input-sm text-sm"
              value={creator.description}
              oninput={(e) =>
                creator.setDescription((e.currentTarget as HTMLInputElement).value)}
            />
          </label>

          {#if creator.checks.facts?.sends_images_externally}
            <div
              class="rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200"
              data-testid="vlm-external-banner"
            >
              This endpoint sends crops outside this deployment.
            </div>
          {/if}

          <VlmEndpointForm
            schema={creator.schema}
            body={creator.draftBody}
            report={creator.report}
            list={creator.list}
            catalog={creator.catalog}
            onchange={(field, value) => creator.setField(field, value)}
          />
        </section>

        <aside
          class="order-first flex min-w-0 flex-col gap-4 lg:sticky lg:top-4 lg:order-none lg:self-start"
        >
          <section class="surface flex flex-col gap-2 p-4 text-sm" aria-label="Create">
            <div
              class="flex flex-wrap items-center gap-2 text-xs"
              data-testid="report-counts"
            >
              <span class={counts.errors > 0 ? 'text-red-300' : 'text-zinc-400'}
                >{counts.errors} error{counts.errors === 1 ? '' : 's'}</span
              >
              <span class={counts.warnings > 0 ? 'text-amber-300' : 'text-zinc-400'}
                >{counts.warnings} warning{counts.warnings === 1 ? '' : 's'}</span
              >
              {#if creator.validating}<span class="text-zinc-500">checking…</span>{/if}
            </div>
            {#if creator.validateError}
              <p class="text-xs text-red-300">
                Could not check the draft: {creator.validateError}
              </p>
            {/if}
            <div class="flex flex-wrap gap-2">
              <button
                type="button"
                class="btn btn-primary btn-sm"
                disabled={!creator.canCreate}
                data-testid="vlm-create"
                onclick={() => void doCreate()}
                >{creator.creating ? 'Creating…' : 'Create'}</button
              >
              <button
                type="button"
                class="btn btn-sm"
                disabled={creator.checks.testing}
                data-testid="vlm-test-connection"
                onclick={() => void creator.testConnection()}
                >{creator.checks.testing ? 'Testing…' : 'Test connection'}</button
              >
            </div>
            {#if creator.createError}
              <p class="text-xs text-red-300" data-testid="vlm-create-error">
                {creator.createError}
              </p>
            {/if}
            {#if creator.createReport}
              <ConfigIssueList
                issues={[
                  ...creator.createReport.errors,
                  ...creator.createReport.warnings,
                ]}
                showField
              />
            {/if}
            {#if creator.checks.testError}
              <p class="text-xs text-red-300" data-testid="vlm-test-error">
                {creator.checks.testError}
              </p>
            {/if}
            {#if creator.checks.probe}
              <VlmProbePanel probe={creator.checks.probe} />
            {/if}
          </section>
        </aside>
      </div>
    {/if}
  </ConfigGate>
</div>
