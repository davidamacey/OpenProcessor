/*
 * Editor for the deployment-wide strategy defaults on /settings: the local
 * unsaved pick per axis, the confirm dialog state and the save / clear
 * action with its refusal handling. Every save and clear is gated behind
 * the confirm dialog because the write is deployment-wide.
 */
import { ApiError, apiErrorText } from '$lib/api_errors';
import { unknownStrategyDetail } from '$lib/api_autoLabel';
import { configErrorDetail } from '$lib/api_profiles';
import { effectiveDefaultId, type SettingsAxisSpec } from '$lib/curationSettings';
import { curationSettingsStore } from '$stores/curationSettings.svelte';
import { strategiesStore } from '$stores/strategies.svelte';
import { toastStore } from '$stores/toast.svelte';

export function createDefaultsEditor() {
  /** Local, unsaved selection per settable axis id. Cleared back to
   *  "follow the effective id" after a successful save or a reload, so
   *  it never drifts from the server's own record. */
  let selections = $state<Record<string, string>>({});
  let saveErrors = $state<Record<string, string>>({});
  /** Axes whose last save the server refused for a missing external
   *  acknowledgement: the error is followed by the Models link. */
  let ackRefused = $state<Record<string, boolean>>({});

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
    if (ackRefused[spec.axis]) ackRefused = { ...ackRefused, [spec.axis]: false };
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

  function withoutAxis<V>(rec: Record<string, V>, axis: string): Record<string, V> {
    const next = { ...rec };
    delete next[axis];
    return next;
  }

  function onSaveSuccess(
    spec: SettingsAxisSpec,
    mode: 'set' | 'clear',
    id: string | null,
  ) {
    selections = withoutAxis(selections, spec.axis);
    if (saveErrors[spec.axis]) saveErrors = withoutAxis(saveErrors, spec.axis);
    // /methods' per-entry `default: true` flag is derived server-side
    // from this same record (plan §4.3) — invalidate the cache so the
    // next mount of any StrategyBar/AssistScopeBar consumer re-fetches.
    strategiesStore.reset();
    toastStore.success(
      mode === 'clear'
        ? `Shared ${spec.label} default cleared`
        : `Shared ${spec.label} default set to ${id}`,
    );
  }

  /** The served words: a refused VLM pick names its endpoint and the
   *  valid ids; other structured refusals carry their own `message`. */
  function saveFailureMessage(e: unknown, mode: 'set' | 'clear'): string {
    const unknown = unknownStrategyDetail(e);
    if (unknown) {
      return `Unknown ${unknown.axis.replace('_', ' ')} "${unknown.requested}" — valid: ${unknown.valid_ids.join(', ') || 'none'}.`;
    }
    const refusal = configErrorDetail(e);
    if (refusal) return refusal.message;
    return (
      apiErrorText(e) ??
      (mode === 'clear' ? 'failed to clear setting' : 'failed to save settings')
    );
  }

  function onSaveFailure(spec: SettingsAxisSpec, mode: 'set' | 'clear', e: unknown) {
    const message = saveFailureMessage(e, mode);
    saveErrors = { ...saveErrors, [spec.axis]: message };
    // A refused save is shown under its control; the store's load-error
    // state would otherwise replace the whole page with a Retry.
    curationSettingsStore.error = null;
    if (configErrorDetail(e)?.error === 'vlm_external_not_acknowledged') {
      ackRefused = { ...ackRefused, [spec.axis]: true };
    }
    toastStore.error(message);
    // Drop the rejected local pick — the control must revert to
    // whatever is actually in effect, never keep showing the rejected
    // value as though it had been saved.
    selections = withoutAxis(selections, spec.axis);
    // A 422 "not currently advertised" means /methods is stale (plan
    // §4.4) — re-sync both caches rather than leave a dead option
    // selectable.
    if (e instanceof ApiError && e.status === 422) {
      strategiesStore.reset();
      void curationSettingsStore.refresh();
    }
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
      onSaveSuccess(spec, mode, id);
    } catch (e) {
      onSaveFailure(spec, mode, e);
    } finally {
      confirmSpec = null;
      pendingId = null;
      confirmMode = 'set';
    }
  }

  return {
    get saveErrors() {
      return saveErrors;
    },
    get ackRefused() {
      return ackRefused;
    },
    get confirmSpec() {
      return confirmSpec;
    },
    get pendingId() {
      return pendingId;
    },
    get confirmMode() {
      return confirmMode;
    },
    currentSelection,
    onSelectChange,
    openConfirm,
    openConfirmClear,
    closeConfirm,
    reload,
    confirmAction,
  };
}
