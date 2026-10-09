/**
 * State/logic for the dashboard's `<AssistScopeBar>` control — which
 * class (and, optionally, which prompt pack) an
 * assisted auto-label run is scoped to
 * (docs/design/vlm-scoped-labeling-assist-plan-2026-09-20.md §3).
 *
 * Same "logic extracted from the component so it's unit-testable"
 * pattern as `strategyBar.svelte.ts` / `pager.svelte.ts` /
 * `selection.svelte.ts`: a factory returning a plain object with
 * getters/setters over `$state`, not a class, so a component can
 * `bind:value={scope.classId}` directly.
 *
 * Deliberately dumb, exactly like `strategyBar.svelte.ts`: this module
 * knows nothing about `/methods`, entry status, or whether an axis is
 * even available. That is capability discovery, owned by
 * `strategiesStore` and rendered/gated by `AssistScopeBar.svelte`. This
 * module only knows "what is currently selected" and "how to serialize
 * it", so it stays trivially testable without mocking the store or the
 * network.
 *
 * `toStartParams()` is the ONE place a selection becomes wire params.
 */

// Type-only import: erased at compile time, so this module keeps its
// zero-runtime-dependency property (importing `$lib/api` for real would
// drag in `$env` and the fetch layer for no reason).
import type { AutoLabelStartParams } from '$lib/api';

export interface AssistScope {
  /** Class to scope the run to, or `null` = whole dataset (today's
   *  behavior, and the default). */
  classId: number | null;
  /** Selected `axis: 'prompt_pack'` entry id, or `null` = server default. */
  promptPack: string | null;
  /** Selected `axis: 'vlm'` entry id (a registered endpoint, or `off`), or
   *  `null` = the project's active endpoint (nothing is sent). */
  vlm: string | null;
  /** The operator's acknowledgement that this run sends crops outside the
   *  deployment. Sent only when `true`. */
  acknowledgeExternal: boolean;
  /** True when nothing is scoped — the run is exactly today's run. */
  readonly isDefault: boolean;
  /**
   * The subset of `AutoLabelStartParams` this selection contributes.
   * `{}` when `isDefault`, so spreading it into the existing
   * `startAutoLabel({...})` call produces a byte-identical URL to the
   * one this app sends today. Only keys with a real selection are
   * emitted — `qs()` would drop nulls anyway, but omitting them here
   * makes the byte-identity provable by `toEqual({})` in a unit test
   * rather than by reasoning about `qs()`.
   */
  toStartParams(): Partial<AutoLabelStartParams>;
  reset(): void;
}

export function createAssistScope(): AssistScope {
  let classId = $state<number | null>(null);
  let promptPack = $state<string | null>(null);
  let vlm = $state<string | null>(null);
  let acknowledgeExternal = $state(false);

  return {
    get classId() {
      return classId;
    },
    set classId(next: number | null) {
      classId = next;
    },
    get promptPack() {
      return promptPack;
    },
    set promptPack(next: string | null) {
      promptPack = next;
    },
    get vlm() {
      return vlm;
    },
    set vlm(next: string | null) {
      vlm = next;
    },
    get acknowledgeExternal() {
      return acknowledgeExternal;
    },
    set acknowledgeExternal(next: boolean) {
      acknowledgeExternal = next;
    },
    get isDefault() {
      return classId == null && promptPack == null && vlm == null && !acknowledgeExternal;
    },

    toStartParams(): Partial<AutoLabelStartParams> {
      const params: Partial<AutoLabelStartParams> = {};
      if (classId != null) params.class_id = classId;
      if (promptPack != null) params.prompt_pack = promptPack;
      if (vlm != null) params.vlm = vlm;
      if (acknowledgeExternal) params.acknowledge_external = true;
      return params;
    },

    reset(): void {
      classId = null;
      promptPack = null;
      vlm = null;
      acknowledgeExternal = false;
    },
  };
}
