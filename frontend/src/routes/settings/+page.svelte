<script lang="ts">
  /**
   * Deployment defaults — admin page for the shared curation-strategy
   * defaults (`GET,PUT {API_PREFIX}/settings`). See
   * docs/design/curation-settings-ui-plan-2026-09-21.md §5.
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

  import { ApiError } from '$lib/api';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import { formatTimestamp } from '$lib/formatDate';
  import {
    advisoryAxes,
    axisOptions,
    effectiveDefaultId,
    isPinned,
    settableAxes,
    type SettingsAxisSpec,
  } from '$lib/curationSettings';
  import { curationSettingsStore } from '$stores/curationSettings.svelte';
  import { strategiesStore } from '$stores/strategies.svelte';
  import { hasFieldCoverage, type MethodInfoBase } from '$lib/strategies';
  import ScoresCard from '$lib/components/ScoresCard.svelte';

  // `axisOptions()` returns the shared `MethodInfoBase[]` (it serves every
  // axis, not just review_sorts), which doesn't itself declare
  // `field_coverage` — the real runtime entries do (`ReviewSortInfo` etc.,
  // same structural gap StrategyBar.svelte's local `hasFieldCoverage`
  // wrapper works around).
  function coverageOf(opt: MethodInfoBase): boolean {
    return hasFieldCoverage(opt as { field_coverage?: number | null });
  }
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

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

  /** Local, unsaved selection per settable axis id. Cleared back to
   *  "follow the effective id" after a successful save or a reload, so
   *  it never drifts from the server's own record. */
  let selections = $state<Record<string, string>>({});
  let saveErrors = $state<Record<string, string>>({});

  let confirmSpec = $state<SettingsAxisSpec | null>(null);
  let pendingId = $state<string | null>(null);
  /** Distinguishes the confirm dialog's two possible actions — 'clear'
   *  sends `{[axis]: null}` instead of `{[axis]: pendingId}`. */
  let confirmMode = $state<'set' | 'clear'>('set');

  function currentSelection(spec: SettingsAxisSpec): string | null {
    return (
      selections[spec.axis] ??
      effectiveDefaultId(curationSettingsStore.settings, strategiesStore.methods, spec)
    );
  }

  function onSelectChange(spec: SettingsAxisSpec, value: string): void {
    selections = { ...selections, [spec.axis]: value };
    if (saveErrors[spec.axis]) {
      const next = { ...saveErrors };
      delete next[spec.axis];
      saveErrors = next;
    }
  }

  function openConfirm(spec: SettingsAxisSpec): void {
    const value = selections[spec.axis];
    if (value == null) return;
    confirmSpec = spec;
    pendingId = value;
    confirmMode = 'set';
  }

  function openConfirmClear(spec: SettingsAxisSpec): void {
    confirmSpec = spec;
    pendingId = null;
    confirmMode = 'clear';
  }

  function closeConfirm(): void {
    confirmSpec = null;
    pendingId = null;
    confirmMode = 'set';
  }

  function reload(): void {
    void curationSettingsStore.refresh();
    strategiesStore.reset();
    void strategiesStore.init();
  }

  async function confirmAction(): Promise<void> {
    if (!confirmSpec) return;
    if (confirmMode === 'set' && pendingId == null) return;
    const spec = confirmSpec;
    const id = pendingId;
    const mode = confirmMode;
    try {
      if (mode === 'clear') {
        await curationSettingsStore.clearDefault(spec.axis);
      } else {
        await curationSettingsStore.saveDefault(spec.axis, id as string);
      }
      const nextSelections = { ...selections };
      delete nextSelections[spec.axis];
      selections = nextSelections;
      if (saveErrors[spec.axis]) {
        const nextErrors = { ...saveErrors };
        delete nextErrors[spec.axis];
        saveErrors = nextErrors;
      }
      // /methods' per-entry `default: true` flag is derived server-side
      // from this same record (plan §4.3) — invalidate the cache so the
      // next mount of any StrategyBar/AssistScopeBar consumer re-fetches.
      strategiesStore.reset();
      toastStore.success(
        mode === 'clear'
          ? `Shared ${spec.label} default cleared`
          : `Shared ${spec.label} default set to ${id}`,
      );
    } catch (e) {
      const message =
        (e as Error)?.message ??
        (mode === 'clear' ? 'failed to clear setting' : 'failed to save settings');
      saveErrors = { ...saveErrors, [spec.axis]: message };
      toastStore.error(message);
      // Drop the rejected local pick — the control must revert to
      // whatever is actually in effect, never keep showing the rejected
      // value as though it had been saved.
      const nextSelections = { ...selections };
      delete nextSelections[spec.axis];
      selections = nextSelections;
      // A 422 "not currently advertised" means /methods is stale (plan
      // §4.4) — re-sync both caches rather than leave a dead option
      // selectable.
      if (e instanceof ApiError && e.status === 422) {
        strategiesStore.reset();
        void curationSettingsStore.refresh();
      }
    } finally {
      confirmSpec = null;
      pendingId = null;
      confirmMode = 'set';
    }
  }

  const advisoryVisible = $derived(
    advisoryAxes(strategiesStore.methods).some(
      (a) => axisOptions(strategiesStore.methods, a).length > 0,
    ),
  );
</script>

<div class="mx-auto flex h-full max-w-7xl flex-col gap-4 p-6">
  <header class="flex flex-wrap items-center gap-3">
    <h1 class="text-2xl font-semibold tracking-tight">Deployment defaults</h1>
    <span class="grow"></span>
    <button type="button" class="btn" onclick={reload}>Reload</button>
  </header>

  <div
    class="rounded border border-zinc-700 bg-zinc-900/60 px-4 py-3 text-sm text-zinc-300"
  >
    These are deployment-wide. There is no per-user setting and no undo. Every operator on
    this Cropwright instance gets what you save here.
  </div>

  {#if curationSettingsStore.loading && !curationSettingsStore.loaded}
    <section class="surface p-6 text-sm text-zinc-500">Loading settings…</section>
  {:else if curationSettingsStore.supported === false}
    <section class="surface p-6 text-sm text-zinc-400">
      This backend does not support shared curation defaults. <code
        class="text-xs text-zinc-500">GET {'{API_PREFIX}'}/settings</code
      > returned 404 — the server predates the feature. Per-session defaults on /clusters and
      /review are unaffected.
    </section>
  {:else if curationSettingsStore.error}
    <section class="surface flex flex-col gap-3 p-6 text-sm">
      <p class="text-red-300">{curationSettingsStore.error}</p>
      <button type="button" class="btn w-fit" onclick={reload}>Retry</button>
    </section>
  {:else}
    <section class="surface flex flex-col gap-5 p-5">
      <div class="flex items-center justify-between">
        <h2 class="text-base font-semibold">Active defaults</h2>
        {#if curationSettingsStore.settings.updated_at}
          <span
            class="text-xs text-zinc-500"
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
        <div
          class="flex flex-col gap-1.5 border-t border-zinc-800 pt-4 first:border-0 first:pt-0"
        >
          <div class="flex flex-wrap items-center gap-2">
            <span class="w-44 shrink-0 text-sm font-medium">{spec.label}</span>
            {#if options.length === 0}
              <span class="text-sm text-zinc-500">no options advertised</span>
            {:else}
              <select
                class="select"
                value={selected ?? ''}
                onchange={(e) =>
                  onSelectChange(spec, (e.currentTarget as HTMLSelectElement).value)}
              >
                {#each options as opt (opt.id)}
                  <option value={opt.id}>
                    {opt.label}{opt.status === 'experimental'
                      ? ' · beta'
                      : ''}{coverageOf(opt) ? '' : ' · no coverage yet'}
                  </option>
                {/each}
              </select>
              {@const selectedOpt = options.find((o) => o.id === (selected ?? effective))}
              {#if selectedOpt && !coverageOf(selectedOpt)}
                <span
                  class="rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[11px] text-amber-200"
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
          <p class="text-xs text-zinc-400">{spec.blurb}</p>
          <p class="text-xs text-zinc-500">
            {pinned ? 'pinned' : "inherited from the backend's built-in default"}
          </p>
          {#if saveErrors[spec.axis]}
            <p
              class="rounded border border-red-500/40 bg-red-500/10 px-2 py-1 text-xs text-red-200"
            >
              {saveErrors[spec.axis]}
            </p>
          {/if}
        </div>
      {/each}
    </section>

    {#if advisoryVisible}
      <section class="surface flex flex-col gap-4 border-dashed p-5">
        <h2 class="text-base font-semibold text-zinc-300">
          Set by the backend's startup config
        </h2>
        <p class="text-xs text-zinc-400">
          These axes are chosen by the backend's startup config, not by a shared default —
          the settings API rejects one for them. Shown here so you can see what is active.
        </p>
        {#each advisoryAxes(strategiesStore.methods) as spec (spec.axis)}
          {@const options = axisOptions(strategiesStore.methods, spec)}
          {#if options.length > 0}
            <div class="flex flex-wrap items-center gap-2 text-sm">
              <span class="w-44 shrink-0 font-medium text-zinc-400">{spec.label}</span>
              {#each options as opt (opt.id)}
                <span class="text-zinc-300"
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

  <ScoresCard />
</div>

{#if confirmSpec}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Confirm shared default"
    tabindex="-1"
    use:focusOnMount
    use:trapFocus={{ onEscape: closeConfirm }}
    onclick={(e) => {
      if (e.target === e.currentTarget) closeConfirm();
    }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-3 text-base font-semibold">
        {confirmMode === 'clear' ? 'Clear shared default' : 'Set shared default'}: {confirmSpec.label}
      </h3>
      <p class="mb-3 text-sm text-zinc-300">
        {effectiveDefaultId(
          curationSettingsStore.settings,
          strategiesStore.methods,
          confirmSpec,
        ) ?? '(none)'} →
        <strong
          >{confirmMode === 'clear' ? "each caller's own default" : pendingId}</strong
        >
      </p>
      <p class="mb-3 text-xs text-zinc-400">{confirmSpec.blurb}</p>
      {#if confirmMode === 'clear'}
        <p class="mb-3 text-xs text-zinc-400">
          Removes this axis's pinned override entirely — every caller that reads it falls
          back to its own built-in default instead of the shared one.
        </p>
      {:else if confirmSpec.irreversibleWarning}
        <div
          class="mb-3 rounded border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200"
        >
          {confirmSpec.irreversibleWarning}
        </div>
      {/if}
      <div class="flex justify-end gap-2">
        <button type="button" class="btn" onclick={closeConfirm}>Cancel</button>
        <button
          type="button"
          class="btn btn-primary"
          onclick={() => void confirmAction()}
        >
          Confirm
        </button>
      </div>
    </div>
  </div>
{/if}
