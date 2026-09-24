<script lang="ts">
  /**
   * Scope selector for an assisted auto-label run
   * (docs/design/vlm-scoped-labeling-assist-plan-2026-09-20.md §4).
   * Rendered by `AutoLabelPanel` on `/dashboard`, and only when
   * `isScopedAssistAvailable` says the backend advertises the feature —
   * that gate lives in the parent, not here, so this component never
   * renders against a backend that would silently ignore its output.
   *
   * Interaction pattern is `StrategyBar.svelte`'s, deliberately: a
   * collapsed summary chip that expands into independent, stackable
   * controls; blue when non-default; a reset that only appears when
   * there is something to reset; pointer-first with zero global
   * keybindings (AssistScopeBar.test.ts asserts no global listener call).
   *
   * Three independent controls, each additive and each individually
   * optional:
   *   1. class scope   — always offered once the bar renders; the
   *                      `class_id` param, fuzzy-searched via
   *                      `searchClasses` ($lib/classPicker), the same
   *                      ranking /review's picker uses.
   *   2. detection profile — only when the `detection_profile` axis is
   *                      advertised.
   *   3. prompt pack   — only when the `prompt_pack` axis is advertised.
   * Leaving all three alone produces `{}` from `toStartParams()`, i.e.
   * exactly today's unscoped run.
   */

  import ChevronDownIcon from './ChevronDownIcon.svelte';
  import { searchClasses } from '$lib/classPicker';
  import {
    isDetectionProfileAvailable,
    isPromptPackAvailable,
    selectableAxisEntries,
  } from '$lib/strategies';
  import type { AssistScope } from '$lib/assistScope.svelte';
  import type { OpClass } from '$lib/types';
  import { strategiesStore } from '$stores/strategies.svelte';

  interface Props {
    scope: AssistScope;
    /** Full class registry. Passed in rather than read from
     *  `classesStore` here so this component stays as store-free as
     *  StrategyBar is about `classesStore` — the parent already has it. */
    classes: OpClass[];
    disabled?: boolean;
  }

  let { scope, classes, disabled = false }: Props = $props();

  // getMethods()/init() never throws (any failure degrades to
  // FALLBACK_METHODS) and init() is idempotent — safe to call on every
  // mount without a guard, exactly as StrategyBar does. Installs no
  // listener.
  $effect(() => {
    void strategiesStore.init();
  });

  const detectionProfiles = $derived(
    selectableAxisEntries(strategiesStore.methods.detection_profiles),
  );
  const promptPacks = $derived(
    selectableAxisEntries(strategiesStore.methods.prompt_packs),
  );
  const detectionProfileAvailable = $derived(
    isDetectionProfileAvailable(strategiesStore.methods.detection_profiles),
  );
  const promptPackAvailable = $derived(
    isPromptPackAvailable(strategiesStore.methods.prompt_packs),
  );

  // `/methods` resolves after mount, so an axis can disappear (a
  // redeploy, a flag flipped off) while its value is selected. Force the
  // stale selection back to "server default" rather than sending an id
  // the backend no longer advertises. Same shape as /train's
  // datasetExportAvailable reset and /clusters' showEmbeddingViz reset —
  // the `!= null` guard is what makes the effect converge instead of
  // looping on state it also reads. Defense in depth: nothing can select
  // a value before the axis is available, so this should never fire.
  $effect(() => {
    if (!detectionProfileAvailable && scope.detectionProfile != null) {
      scope.detectionProfile = null;
    }
  });
  $effect(() => {
    if (!promptPackAvailable && scope.promptPack != null) {
      scope.promptPack = null;
    }
  });

  const selectedClass = $derived(
    scope.classId == null ? null : (classes.find((c) => c.id === scope.classId) ?? null),
  );
  const summary = $derived(selectedClass ? selectedClass.name : 'whole dataset');

  let expanded = $state(false);
  let query = $state('');
  // 50 matches the /review picker's cap — enough to reach the long tail
  // without rendering an 84-row list by default.
  const results = $derived(searchClasses(classes, query, 50));

  function pick(id: number | null): void {
    scope.classId = id;
    query = '';
  }
</script>

<div class="inline-flex flex-wrap items-center gap-1.5 text-xs">
  {#if !expanded}
    <button
      type="button"
      class="btn-sm {scope.isDefault
        ? 'border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800'
        : 'border-blue-500/60 bg-blue-500/15 text-blue-100 hover:bg-blue-500/25'}"
      onclick={() => (expanded = true)}
      title="Limit this run's VLM labeling to one class (clustering still covers the whole pool), and pick a detection profile / prompt pack"
      {disabled}
    >
      <span class="text-zinc-500">assist:</span>
      <span>{summary}</span>
      <span class="text-zinc-500"><ChevronDownIcon size={15} /></span>
    </button>
  {:else}
    <label class="flex items-center gap-1.5">
      <span class="text-zinc-500">class</span>
      <input
        type="search"
        bind:value={query}
        placeholder={summary}
        class="input-sm w-40"
        {disabled}
      />
    </label>
    <div
      class="max-h-40 w-44 overflow-y-auto rounded border border-zinc-700 bg-zinc-950 py-0.5"
    >
      <button
        type="button"
        class="block w-full px-2 py-0.5 text-left {scope.classId == null
          ? 'bg-blue-500/20 text-white'
          : 'text-zinc-300 hover:bg-zinc-800'}"
        onclick={() => pick(null)}
        {disabled}
      >
        whole dataset
      </button>
      {#each results as cls (cls.id)}
        <button
          type="button"
          class="flex w-full items-center gap-2 px-2 py-0.5 text-left {scope.classId ===
          cls.id
            ? 'bg-blue-500/20 text-white'
            : 'text-zinc-300 hover:bg-zinc-800'}"
          onclick={() => pick(cls.id)}
          {disabled}
        >
          <span class="grow truncate">{cls.name}</span>
          <span class="shrink-0 text-[10px] text-zinc-500">{cls.validated_count}</span>
        </button>
      {/each}
      {#if results.length === 0}
        <p class="px-2 py-1 text-zinc-500">No matching class.</p>
      {/if}
    </div>

    <!-- Optional axis: absent entirely (not disabled) when /methods
         doesn't advertise a detection profile. -->
    {#if detectionProfileAvailable}
      <label class="flex items-center gap-1.5">
        <span class="text-zinc-500">detector</span>
        <select
          value={scope.detectionProfile ?? ''}
          onchange={(e) => {
            const v = (e.currentTarget as HTMLSelectElement).value;
            scope.detectionProfile = v === '' ? null : v;
          }}
          class="select-sm"
          {disabled}
        >
          <option value="">Server default</option>
          {#each detectionProfiles as p (p.id)}
            <option value={p.id}>
              {p.label}{p.status === 'experimental' ? ' · beta' : ''}
            </option>
          {/each}
        </select>
      </label>
    {/if}

    {#if promptPackAvailable}
      <label class="flex items-center gap-1.5">
        <span class="text-zinc-500">prompts</span>
        <select
          value={scope.promptPack ?? ''}
          onchange={(e) => {
            const v = (e.currentTarget as HTMLSelectElement).value;
            scope.promptPack = v === '' ? null : v;
          }}
          class="select-sm"
          {disabled}
        >
          <option value="">Server default</option>
          {#each promptPacks as p (p.id)}
            <option value={p.id}>
              {p.label}{p.status === 'experimental' ? ' · beta' : ''}
            </option>
          {/each}
        </select>
      </label>
    {/if}

    {#if !scope.isDefault}
      <button
        type="button"
        class="btn-sm bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
        onclick={() => {
          scope.reset();
          query = '';
        }}
        {disabled}
      >
        reset
      </button>
    {/if}

    <button
      type="button"
      class="btn-sm btn-icon border-zinc-700 bg-zinc-900 text-zinc-400 hover:bg-zinc-800"
      onclick={() => (expanded = false)}
      title="Collapse"
    >
      ×
    </button>
  {/if}
</div>
