<script lang="ts">
  /**
   * Deployment defaults — admin page for the shared curation-strategy
   * defaults (`GET,PUT {API_PREFIX}/settings`). See
   * curation settings UI plan §5.
   *
   * A dedicated route, not a StrategyBar/AssistScopeBar chip (plan §2):
   * this write is deployment-wide, the categorical opposite of those
   * bars' per-session/reset-any-time contract — every save (and every
   * clear) is gated behind an explicit confirm dialog for that reason,
   * even though the backend's null-clear path (added after this plan
   * was written) closed the one case that used to be genuinely
   * irreversible (H-1: a pinned `sort` default). Which axes get a
   * control is the server's per-entry `settable` flag on `/methods`
   * (`settableAxes`), never a hardcoded axis id.
   */

  import { createDefaultsEditor } from '$lib/curationSettingsEditor.svelte';
  import { formatTimestamp } from '$lib/format/date';
  import {
    advisoryAxes,
    axisCopy,
    axisOptions,
    effectiveDefaultId,
    unofferedServedValue,
    isPinned,
    settableAxes,
    settingsOptionView,
  } from '$lib/curationSettings';
  import { curationSettingsStore } from '$stores/curationSettings.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
  import { hasFieldCoverage, type MethodInfoBase } from '$lib/strategies';
  import VlmScopePanel from '$lib/components/settings/VlmScopePanel.svelte';
  import { resolve } from '$app/paths';
  import PageHeader from '$components/ui/PageHeader.svelte';
  import { projectHref } from '$lib/projectPaths';

  // `axisOptions()` returns the shared `MethodInfoBase[]` (it serves every
  // axis, not just review_sorts), which doesn't itself declare
  // `field_coverage` — the real runtime entries do (`ReviewSortInfo` etc.,
  // same structural gap StrategyBar.svelte's local `hasFieldCoverage`
  // wrapper works around).
  function coverageOf(opt: MethodInfoBase): boolean {
    return hasFieldCoverage(opt as { field_coverage?: number | null });
  }
  import { keyboardStore } from '$stores/keyboard.svelte';
  import Modal from '$components/Modal.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  // Both never throw (getMethods() falls back; curationSettingsStore.init()
  // catches internally) and both init()s are idempotent — safe on every
  // mount, no guard needed. Installs no listener.
  $effect(() => {
    void curationSettingsStore.init();
    void strategiesStore.init();
  });

  const editor = createDefaultsEditor();
  const saveErrors = $derived(editor.saveErrors);
  const ackRefused = $derived(editor.ackRefused);
  const confirmSpec = $derived(editor.confirmSpec);
  const pendingId = $derived(editor.pendingId);
  const confirmMode = $derived(editor.confirmMode);
  const {
    currentSelection,
    onSelectChange,
    openConfirm,
    openConfirmClear,
    closeConfirm,
  } = editor;
  const { reload, confirmAction } = editor;

  const advisoryVisible = $derived(
    advisoryAxes(strategiesStore.methods).some(
      (a) => axisOptions(strategiesStore.methods, a).length > 0,
    ),
  );
</script>

<div class="flex flex-col gap-4">
  <PageHeader title="General">
    {#snippet actions()}
      <button type="button" class="btn" onclick={reload}>Reload</button>
    {/snippet}
  </PageHeader>

  <div
    class="rounded border border-line-strong bg-surface-1/60 px-4 py-3 text-sm text-fg-2"
  >
    These are deployment-wide. There is no per-user setting and no undo. Every operator on
    this Cropwright instance gets what you save here.
  </div>

  {#if curationSettingsStore.loading && !curationSettingsStore.loaded}
    <section class="surface p-6 text-sm text-fg-3">Loading settings…</section>
  {:else if curationSettingsStore.error}
    <section class="surface flex flex-col gap-3 p-6 text-sm">
      <p class="text-danger-text">{curationSettingsStore.error}</p>
      <button type="button" class="btn w-fit" onclick={reload}>Retry</button>
    </section>
  {:else}
    <section class="surface flex flex-col gap-5 p-5">
      <div class="flex items-center justify-between">
        <h2 class="text-base font-semibold">Active defaults</h2>
        {#if curationSettingsStore.settings.updated_at}
          <span
            class="text-xs text-fg-3"
            title={curationSettingsStore.settings.updated_at}
          >
            last changed {formatTimestamp(curationSettingsStore.settings.updated_at)}
          </span>
        {/if}
      </div>

      {#each settableAxes(strategiesStore.methods) as spec (spec.axis)}
        {@const options = axisOptions(strategiesStore.methods, spec)}
        {@const effective = effectiveDefaultId(
          curationSettingsStore.settings,
          strategiesStore.methods,
          spec,
        )}
        {@const selected = currentSelection(spec)}
        {@const pinned = isPinned(curationSettingsStore.settings, spec)}
        {@const copy = axisCopy(strategiesStore.methods, spec)}
        <div
          class="flex flex-col gap-1.5 border-t border-line pt-4 first:border-0 first:pt-0"
        >
          <div class="flex flex-wrap items-center gap-2">
            <span class="w-44 shrink-0 text-sm font-medium">{copy.label}</span>
            {#if options.length === 0}
              <span class="text-sm text-fg-3">no options advertised</span>
            {:else}
              <select
                class="select"
                aria-label={copy.label}
                value={selected ?? ''}
                onchange={(e) =>
                  onSelectChange(spec, (e.currentTarget as HTMLSelectElement).value)}
              >
                {#if selected == null}
                  <!-- Nothing pinned and no served default for this
                       axis (e.g. review sort: each tab keeps its own), so
                       say that instead of rendering a blank select. -->
                  <option value="" disabled
                    >not set: each view uses its own default</option
                  >
                {/if}
                {#if unofferedServedValue(selected, options)}
                  <option value={selected} disabled
                    >{selected} (served, not an offered choice)</option
                  >
                {/if}
                {#each options as opt (opt.id)}
                  {@const view = settingsOptionView(spec, opt)}
                  <option value={opt.id} disabled={view.disabled}>
                    {opt.label}{opt.status === 'experimental'
                      ? ' · beta'
                      : ''}{coverageOf(opt) ? '' : ' · no coverage yet'}{view.suffix}
                  </option>
                {/each}
              </select>
              {@const selectedOpt = options.find((o) => o.id === (selected ?? effective))}
              {@const selectedWarning = selectedOpt
                ? settingsOptionView(spec, selectedOpt).warning
                : null}
              {#if selectedWarning}
                <span
                  class="rounded border border-danger/40 bg-danger/10 px-1.5 py-0.5 text-caption text-danger-text"
                  data-testid="settings-option-warning">{selectedWarning}</span
                >
              {/if}
              {#if selectedOpt && !coverageOf(selectedOpt)}
                <span
                  class="rounded border border-warn/40 bg-warn/10 px-1.5 py-0.5 text-caption text-warn-text"
                >
                  0 coverage — pinning this sorts by tie-break only
                </span>
              {/if}
              <button
                type="button"
                class="btn"
                disabled={selected == null ||
                  selected === effective ||
                  curationSettingsStore.saving === spec.axis}
                onclick={() => openConfirm(spec)}
              >
                {curationSettingsStore.saving === spec.axis && confirmMode === 'set'
                  ? 'Saving…'
                  : 'Save'}
              </button>
              <button
                type="button"
                class="btn"
                disabled={!pinned || curationSettingsStore.saving === spec.axis}
                onclick={() => openConfirmClear(spec)}
              >
                {curationSettingsStore.saving === spec.axis && confirmMode === 'clear'
                  ? 'Clearing…'
                  : 'Clear'}
              </button>
            {/if}
          </div>
          <p class="text-xs text-fg-2">{copy.blurb}</p>
          <p class="text-xs text-fg-3">
            {pinned ? 'pinned' : "inherited from the backend's built-in default"}
          </p>
          {#if options.some((o) => settingsOptionView(spec, o).disabled)}
            <p class="text-xs text-fg-2" data-testid="settings-ack-hint">
              An endpoint that sends crops outside this deployment is acknowledged when it
              is activated.
              <a
                class="text-accent-text hover:underline"
                href={resolve(projectHref('/settings/models'))}>Settings → Models</a
              >
            </p>
          {/if}
          {#if saveErrors[spec.axis]}
            <p
              class="rounded border border-danger/40 bg-danger/10 px-2 py-1 text-xs text-danger-text"
              data-testid="settings-save-error"
            >
              {saveErrors[spec.axis]}
              {#if ackRefused[spec.axis]}
                <a
                  class="ml-1 text-accent-text hover:underline"
                  href={resolve(projectHref('/settings/models'))}>Settings → Models</a
                >
              {/if}
            </p>
          {/if}
        </div>
      {/each}
    </section>

    {#if advisoryVisible}
      <section class="surface flex flex-col gap-4 border-dashed p-5">
        <h2 class="text-base font-semibold text-fg-2">Not settable on this backend</h2>
        <p class="text-xs text-fg-2">
          The backend does not accept a shared default for these axes (it marks them not
          settable). Shown here so you can see what is active.
        </p>
        {#each advisoryAxes(strategiesStore.methods) as spec (spec.axis)}
          {@const options = axisOptions(strategiesStore.methods, spec)}
          {#if options.length > 0}
            <div class="flex flex-wrap items-center gap-2 text-sm">
              <span class="w-44 shrink-0 font-medium text-fg-2"
                >{axisCopy(strategiesStore.methods, spec).label}</span
              >
              {#each options as opt (opt.id)}
                <span class="text-fg-2"
                  >{opt.label}{(opt as { default?: boolean }).default
                    ? ' (active)'
                    : ''}</span
                >
              {/each}
            </div>
          {/if}
        {/each}
      </section>
    {/if}
  {/if}

  <VlmScopePanel />
</div>

{#if confirmSpec}
  <Modal label="Confirm shared default" onclose={closeConfirm}>
    <h3 class="mb-3 text-base font-semibold">
      {confirmMode === 'clear' ? 'Clear shared default' : 'Set shared default'}: {axisCopy(
        strategiesStore.methods,
        confirmSpec,
      ).label}
    </h3>
    <p class="mb-3 text-sm text-fg-2">
      {effectiveDefaultId(
        curationSettingsStore.settings,
        strategiesStore.methods,
        confirmSpec,
      ) ?? '(none)'} →
      <strong>{confirmMode === 'clear' ? "each caller's own default" : pendingId}</strong>
    </p>
    <p class="mb-3 text-xs text-fg-2">
      {axisCopy(strategiesStore.methods, confirmSpec).blurb}
    </p>
    {#if confirmMode === 'clear'}
      <p class="mb-3 text-xs text-fg-2">
        Removes this axis's pinned override entirely — every caller that reads it falls
        back to its own built-in default instead of the shared one.
      </p>
    {:else if confirmSpec.irreversibleWarning}
      <div
        class="mb-3 rounded border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn-text"
      >
        {confirmSpec.irreversibleWarning}
      </div>
    {/if}
    <div class="flex justify-end gap-2">
      <button type="button" class="btn" onclick={closeConfirm}>Cancel</button>
      <button type="button" class="btn btn-primary" onclick={() => void confirmAction()}>
        Confirm
      </button>
    </div>
  </Modal>
{/if}
