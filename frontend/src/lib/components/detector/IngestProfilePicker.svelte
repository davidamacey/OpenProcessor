<!--
  Lists the project's ingest profiles and activates one. Everything shown is
  served: the rows, the active flag and, on a refusal, the report's issue
  messages. "Activate anyway" appears only when the served report says
  `force_allowed`. Profiles are edited through the ingest-profile routes.
-->
<script lang="ts">
  import ConfigIssueList from '$lib/components/config/ConfigIssueList.svelte';
  import type { IngestProfilePicker } from '$lib/detector/ingestProfilePickerController.svelte';

  let { ctl }: { ctl: IngestProfilePicker } = $props();

  import { tick } from 'svelte';

  const refusal = $derived(ctl.refusal);

  let section: HTMLElement | undefined = $state();
  let heading: HTMLElement | undefined = $state();
  let refusalEl: HTMLElement | undefined = $state();

  // An outcome moves keyboard focus to where it is shown, so a screen-reader
  // or keyboard user is never left on a control that has just unmounted.
  $effect(() => {
    if (refusal && refusalEl) refusalEl.focus();
  });

  $effect(() => {
    const name = ctl.activatedName;
    if (name === null) return;
    void tick().then(() => {
      const row = section?.querySelector<HTMLElement>(
        `[data-testid="ingest-profile-row"][data-name="${CSS.escape(name)}"]`,
      );
      (row ?? heading)?.focus();
    });
  });

  function activate(name: string, force = false): void {
    // The "Activate anyway" button unmounts the moment the refusal clears.
    if (force && ctl.activating === null) heading?.focus();
    void ctl.activate(name, force);
  }
</script>

<section
  class="surface space-y-3 p-4"
  data-testid="ingest-profile-picker"
  bind:this={section}
>
  <h2 class="text-sm font-semibold" tabindex="-1" bind:this={heading}>Ingest profiles</h2>
  {#if ctl.loadError}
    <p class="text-xs text-red-300" role="alert" data-testid="ingest-profiles-error">
      {ctl.loadError}
    </p>
  {:else if ctl.list === null}
    <p class="text-xs text-zinc-400">Loading ingest profiles.</p>
  {:else if ctl.list.profiles.length === 0}
    <p class="text-xs text-zinc-400" data-testid="ingest-profiles-empty">
      No ingest profiles are stored for this project.
    </p>
  {:else}
    <ul
      class="divide-y divide-zinc-800"
      data-testid="ingest-profile-list"
      aria-busy={ctl.activating !== null || ctl.loading}
    >
      {#each ctl.list.profiles as p (p.name)}
        <li
          class="flex flex-wrap items-center gap-x-3 gap-y-1 py-2 text-xs"
          tabindex="-1"
          data-testid="ingest-profile-row"
          data-name={p.name}
          data-active={p.active}
        >
          <span class="font-mono text-zinc-100">{p.name}</span>
          {#if p.revision !== null}<span class="text-zinc-500">r{p.revision}</span>{/if}
          <span class="text-zinc-400">
            detector <span class="font-mono text-zinc-200">{p.primary_model}</span>
            {#if p.secondary_model}
              + <span class="font-mono text-zinc-200">{p.secondary_model}</span>{/if}
          </span>
          <span class="grow"></span>
          {#if p.active}
            <span
              class="rounded border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-emerald-200"
              data-testid="ingest-profile-active-chip">Active</span
            >
          {:else}
            <button
              type="button"
              class="btn btn-sm"
              data-testid="ingest-profile-activate"
              aria-label="{ctl.activating === p.name
                ? 'Activating'
                : 'Activate'} {p.name}"
              aria-disabled={ctl.activating !== null}
              onclick={() => activate(p.name)}
              >{ctl.activating === p.name ? 'Activating...' : 'Activate'}</button
            >
          {/if}
        </li>
      {/each}
    </ul>
  {/if}

  {#if refusal}
    <div
      class="space-y-2 rounded border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-200"
      data-testid="ingest-profile-refusal"
      role="alert"
      tabindex="-1"
      bind:this={refusalEl}
    >
      <p>
        <span class="font-mono">{refusal.name}</span> was not activated.
        {refusal.message}
      </p>
      {#if refusal.report}
        <ConfigIssueList
          issues={[...refusal.report.errors, ...refusal.report.warnings]}
          showField
        />
        {#if refusal.report.force_allowed}
          <button
            type="button"
            class="btn btn-sm"
            data-testid="ingest-profile-force"
            aria-disabled={ctl.activating !== null}
            onclick={() => activate(refusal.name, true)}
            >Activate anyway (the server allows overriding these errors)</button
          >
        {/if}
      {/if}
    </div>
  {/if}

  {#if ctl.followUpError}
    <p
      class="text-xs text-red-300"
      role="alert"
      data-testid="ingest-profile-followup-error"
    >
      <span class="font-mono">{ctl.activatedName}</span> was activated, but refreshing the
      page failed. {ctl.followUpError}
    </p>
  {/if}

  <div role="status" data-testid="ingest-profile-status">
    {#if !refusal && ctl.lastValidation && ctl.lastValidation.warnings.length > 0}
      <div class="space-y-1 text-xs" data-testid="ingest-profile-warnings">
        <p class="text-amber-200">Activated, with warnings.</p>
        <ConfigIssueList issues={ctl.lastValidation.warnings} showField />
      </div>
    {/if}
  </div>
</section>
