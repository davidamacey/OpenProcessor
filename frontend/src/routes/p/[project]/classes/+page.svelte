<script lang="ts">
  import { apiErrorText } from '$lib/api';
  import {
    ApiError,
    classStillReferencedDetail,
    deprecateClass,
    getNewClassProposalsSummary,
    getTestHoldoutStats,
    getThumbUrl,
    mergeClasses,
    previewClassMerge,
    renameClass,
    resolveNewClassProposal,
    restoreClass,
    classMergedDetail,
    classMergedRestoreText,
    syncClassesToOpensearch,
    type NewClassProposalsSummary,
    type NewClassProposalTerm,
  } from '$lib/api';
  import AddClassModal from '$components/AddClassModal.svelte';
  import SeedFromDetectorPanel from '$lib/components/detector/SeedFromDetectorPanel.svelte';
  import { adequacyChipClass, adequacyTooltip } from '$lib/adequacy';
  import { proposalRows, termRulesText } from '$lib/classes/proposalRows';
  import { formatDateOnly } from '$lib/formatDate';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import { reservedHotkeyLetters, setClassHotkey } from '$lib/classHotkey';
  import type {
    ClassMergeDryRun,
    RegistryClass,
    ResolveNewClassRequest,
    ResolveNewClassResponse,
  } from '$lib/types';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { undoStore } from '$stores/undo.svelte';
  import { onMount } from 'svelte';

  $effect(() => {
    keyboardStore.setScope('classes');
  });

  // ---- table state -------------------------------------------------------

  let query = $state<string>('');
  let showDeprecated = $state<boolean>(false);

  // Inline rename state — only one row may be in edit mode at a time.
  let editingId = $state<number | null>(null);
  let editName = $state<string>('');

  // Modal state
  let addOpen = $state<boolean>(false);
  let mergeOpen = $state<boolean>(false);
  let busy = $state<boolean>(false);

  // Merge form
  let mergeSourceId = $state<number | null>(null);
  let mergeTargetId = $state<number | null>(null);
  let mergePreview = $state<ClassMergeDryRun | null>(null);
  let mergePreviewError = $state<string | null>(null);
  let mergePreviewBusy = $state<boolean>(false);

  // ---- reserved-hotkey conflicts ------------------------------------------
  // A class's bound hotkey can predate a later reservation (e.g. a class
  // bound to `b` before the backend reserved `b` for a slot's keymap). The backend keeps the existing
  // binding rather than silently clearing it, so surface it here instead of
  // hiding the conflict.
  const reservedConflicts = $derived.by(() => {
    const reserved = reservedHotkeyLetters();
    return classesStore.classes.filter(
      (c) =>
        !c.deprecated && c.hotkey_letter && reserved.has(c.hotkey_letter.toLowerCase()),
    );
  });

  // ---- derived data ------------------------------------------------------

  const allClasses = $derived(classesStore.classes);

  const groups = $derived.by(() => {
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local set built and consumed within this computation, never stored in reactive state (only the resulting sorted array is)
    const set = new Set<string>();
    for (const c of allClasses) {
      const g = c.group ?? '';
      if (g) set.add(g);
    }
    return [...set].sort();
  });

  const filtered = $derived.by(() => {
    const q = query.trim().toLowerCase();
    let list = allClasses;
    if (q) {
      list = list.filter(
        (c) =>
          c.name.toLowerCase().includes(q) ||
          (c.group ?? '').toLowerCase().includes(q) ||
          String(c.id).includes(q),
      );
    }
    return list;
  });

  const activeRows = $derived(filtered.filter((c) => !c.deprecated));
  const deprecatedRows = $derived(filtered.filter((c) => c.deprecated));

  // ---- mutations ---------------------------------------------------------

  function startEdit(cls: RegistryClass): void {
    editingId = cls.id;
    editName = cls.name;
  }

  function cancelEdit(): void {
    editingId = null;
    editName = '';
  }

  async function commitRename(cls: RegistryClass): Promise<void> {
    const next = editName.trim();
    if (!next) {
      toastStore.error('Class name is required.');
      return;
    }
    if (next === cls.name) {
      cancelEdit();
      return;
    }
    busy = true;
    try {
      // No client-side slug check — `PUT {API_PREFIX}/classes/{id}` 422s on a
      // bad name (`^[a-z0-9_]+$`) with the pattern in its detail; that
      // message is what the toast shows.
      await renameClass(cls.id, { name: next });
      toastStore.success(`Renamed ${cls.name} → ${next}`);
      cancelEdit();
      await classesStore.clearAndRefetch();
    } catch (e) {
      toastStore.error(`Rename failed: ${apiErrorText(e)}`);
    } finally {
      busy = false;
    }
  }

  async function changeGroup(cls: RegistryClass, newG: string): Promise<void> {
    if ((cls.group ?? '') === newG) return;
    busy = true;
    try {
      await renameClass(cls.id, { group: newG });
      toastStore.success(`${cls.name} → group ${newG}`);
      await classesStore.clearAndRefetch();
    } catch (e) {
      toastStore.error(`Group change failed: ${apiErrorText(e)}`);
    } finally {
      busy = false;
    }
  }

  async function setHotkey(cls: RegistryClass, raw: string): Promise<void> {
    busy = true;
    try {
      await setClassHotkey(cls, raw);
    } finally {
      busy = false;
    }
  }

  // -- Deprecate / Restore (was disabled — no backend support; now real,
  //    POST {API_PREFIX}/classes/{id}/deprecate + /restore). ------------------

  async function deprecateClassAction(cls: RegistryClass): Promise<void> {
    const ok = window.confirm(
      `Deprecate "${cls.name}"? It drops out of pickers, exports, VLM prompts and ` +
        'hotkeys, and can be restored later (unless another class later claims its name).',
    );
    if (!ok) return;
    busy = true;
    try {
      await deprecateClass(cls.id);
      toastStore.success(`Deprecated ${cls.name}.`);
      await classesStore.clearAndRefetch();
    } catch (e) {
      // 409 while the class still has data — the backend names the exact
      // counts; merge is the only way to retire a class in that state, so
      // offer the existing merge flow with this class preselected as the
      // source rather than just failing.
      const ref = classStillReferencedDetail(e);
      if (ref) {
        const goMerge = window.confirm(
          `${ref.message} (${ref.item_count} item(s), ${ref.confirmed_label_count} confirmed ` +
            `label(s)). Merge "${cls.name}" into another class instead?`,
        );
        if (goMerge) openMergeWithSource(cls.id);
      } else {
        toastStore.error(`Deprecate failed: ${apiErrorText(e)}`);
      }
    } finally {
      busy = false;
    }
  }

  async function restoreClassAction(cls: RegistryClass): Promise<void> {
    const ok = window.confirm(`Restore "${cls.name}"? It becomes assignable again.`);
    if (!ok) return;
    busy = true;
    try {
      await restoreClass(cls.id);
      toastStore.success(`Restored ${cls.name}.`);
      await classesStore.clearAndRefetch();
    } catch (e) {
      // F-56: a merged class 409s with a structured `class_merged` detail
      // (its crops already live on the merge target); say where they went
      // and that un-merge isn't supported, using the served text.
      const merged = classMergedDetail(e);
      // Any other 409 is a PLAIN STRING detail (a live class already uses
      // this name) — show it verbatim.
      if (merged) {
        toastStore.error(classMergedRestoreText(merged));
      } else if (e instanceof ApiError && e.status === 409 && e.detail) {
        toastStore.error(e.detail);
      } else {
        toastStore.error(`Restore failed: ${apiErrorText(e)}`);
      }
    } finally {
      busy = false;
    }
  }

  function openAdd(): void {
    addOpen = true;
  }

  function openMerge(): void {
    mergeSourceId = null;
    mergeTargetId = null;
    mergePreview = null;
    mergePreviewError = null;
    mergeOpen = true;
  }

  function openMergeWithSource(sourceId: number): void {
    openMerge();
    mergeSourceId = sourceId;
  }

  function closeMerge(): void {
    if (busy) return;
    mergeOpen = false;
  }

  const mergeSource = $derived(
    mergeSourceId != null
      ? (allClasses.find((c) => c.id === mergeSourceId) ?? null)
      : null,
  );
  const mergeTarget = $derived(
    mergeTargetId != null
      ? (allClasses.find((c) => c.id === mergeTargetId) ?? null)
      : null,
  );

  // Dry-run preview: every time the pair changes, ask the server what a
  // real merge would do (would_relabel / validations_carried_over / holdout_blocking
  // / blocked) before it's possible to confirm. Never guessed client-side —
  // the backend already knows about holdout-blocking rows the frontend has
  // no visibility into.
  $effect(() => {
    const sourceId = mergeSourceId;
    const targetId = mergeTargetId;
    mergePreview = null;
    mergePreviewError = null;
    if (!mergeOpen || sourceId == null || targetId == null || sourceId === targetId)
      return;
    const ctrl = new AbortController();
    mergePreviewBusy = true;
    void previewClassMerge({ source_id: sourceId, target_id: targetId }, ctrl.signal)
      .then((res) => {
        mergePreview = res;
      })
      .catch((e: unknown) => {
        if ((e as Error)?.name === 'AbortError') return;
        mergePreviewError = apiErrorText(e);
      })
      .finally(() => {
        mergePreviewBusy = false;
      });
    return () => ctrl.abort();
  });

  async function submitMerge(): Promise<void> {
    if (mergeSourceId == null || mergeTargetId == null) {
      toastStore.error('Pick both source and target classes.');
      return;
    }
    if (mergeSourceId === mergeTargetId) {
      toastStore.error('Source and target must differ.');
      return;
    }
    if (mergePreview == null) return;
    if (mergePreview?.blocked) {
      toastStore.error('Merge is blocked — resolve the holdout conflict first.');
      return;
    }
    const ok = window.confirm(
      `Merge "${mergeSource?.name}" into "${mergeTarget?.name}"? This relabels ` +
        `${mergePreview.would_relabel} crops and marks the ` +
        'source deprecated. The action is recorded in the registry but cannot be undone from the UI.',
    );
    if (!ok) return;
    busy = true;
    try {
      const res = await mergeClasses({
        source_id: mergeSourceId,
        target_id: mergeTargetId,
      });
      toastStore.success(`Merged '${res.source_name}' into '${res.target_name}'.`);
      mergeOpen = false;
      await classesStore.clearAndRefetch();
    } catch (e) {
      toastStore.error(`Merge failed: ${apiErrorText(e)}`);
    } finally {
      busy = false;
    }
  }

  async function syncToOpensearch(): Promise<void> {
    busy = true;
    try {
      const res = await syncClassesToOpensearch();
      const upserted = res.upserted ?? 0;
      toastStore.success(`Synced ${upserted} classes to OpenSearch.`);
    } catch (e) {
      toastStore.error(`Sync failed: ${apiErrorText(e)}`);
    } finally {
      busy = false;
    }
  }

  // -- New-class proposals (2026-09-24 logic-moves W5; bulk resolve added
  //    2026-09-24 for OpenProcessor 2f5cda2) --------------------------------
  //
  // Aggregate view of the same cohort the `/review` "New Class Proposals"
  // tab pages through one crop at a time — top VLM-proposed-but-unmatched
  // terms with counts and a handful of sample crop ids each (GET
  // {API_PREFIX}/review/new_class_proposals/summary, rendered as thumbnails
  // only). The two actions per term — "Create class & assign" / "Map to
  // existing" — now go through `POST {API_PREFIX}/review/new_class_proposals/
  // resolve`, which the backend resolves against **every** pending item
  // proposing the term, not just the summary's capped `sample_crop_ids`
  // preview. Each action dry-runs first (`?dry_run=true`) to show the
  // operator the real served `matched` count in a confirm dialog before
  // writing anything.
  let proposalsSummary = $state<NewClassProposalsSummary | null>(null);
  let proposalsError = $state<string | null>(null);
  let proposalsLoading = $state<boolean>(false);
  // Per-term inline form state, keyed by term label.
  let newClassNameByTerm = $state<Record<string, string>>({});
  let mapTargetByTerm = $state<Record<string, number | null>>({});
  let proposalBusyTerm = $state<string | null>(null);

  async function loadProposals(): Promise<void> {
    proposalsLoading = true;
    proposalsError = null;
    try {
      proposalsSummary = await getNewClassProposalsSummary();
    } catch (e) {
      // Observed live: this aggregate can 500 on an opensearch outage even
      // while the rest of /classes works fine — degrade to an inline error
      // rather than breaking the page.
      proposalsError = apiErrorText(e);
    } finally {
      proposalsLoading = false;
    }
  }

  onMount(() => void loadProposals());

  // L5 (visual audit 2026-09-24): `validated_count` includes the frozen
  // test holdout (vw 35 here vs 30 trainable on /train). The served
  // per-class holdout count is shown next to it rather than subtracted
  // client-side. A failed read => no suffix.
  let testHeldOutByClass = $state<Map<number, number>>(new Map());
  onMount(() => {
    const ctrl = new AbortController();
    getTestHoldoutStats(ctrl.signal)
      .then((res) => {
        testHeldOutByClass = new Map(res.by_class.map((b) => [b.key, b.doc_count]));
      })
      .catch(() => {
        testHeldOutByClass = new Map();
      });
    return () => ctrl.abort();
  });

  function dismissProposalTerm(label: string): void {
    if (!proposalsSummary) return;
    proposalsSummary = {
      ...proposalsSummary,
      top_terms: proposalsSummary.top_terms.filter((t) => t.label !== label),
      flagged_terms: proposalsSummary.flagged_terms.filter((t) => t.label !== label),
    };
  }

  /**
   * Record the resolve's `updated_ids` for undo (same ring buffer + `Z`
   * behavior `bulkLabel`/`moveCropsToCluster` already use elsewhere in
   * the app — ONE entry for the whole resolve, so a single `Z` reverts
   * every crop it touched, batched through `undo_batch` when there's
   * more than one id), and toast the served counts. Never a
   * client-computed count: `matched` only ever comes from the dry-run/
   * real response, `updated`/`conflicts`/`skipped` only from the real
   * resolve's response.
   */
  function reportResolve(res: ResolveNewClassResponse, verb: string): void {
    if (res.updated_ids.length > 0) undoStore.recordWrites(res.updated_ids);
    const parts = [`${res.updated} updated`];
    if (res.conflicts.length > 0) parts.push(`${res.conflicts.length} conflict(s)`);
    if (res.skipped.length > 0) parts.push(`${res.skipped.length} skipped`);
    toastStore.success(`${verb} — ${parts.join(', ')}.`);
  }

  async function createClassAndAssign(term: { label: string }): Promise<void> {
    const name = (newClassNameByTerm[term.label] ?? term.label).trim();
    if (!name) {
      toastStore.error('Class name is required.');
      return;
    }
    proposalBusyTerm = term.label;
    try {
      // No client-side slug check — the resolve call 422s on a bad
      // `create.class_name` with the pattern in its detail; that message
      // is what the toast shows on failure.
      const body: ResolveNewClassRequest = {
        label: term.label,
        create: { class_name: name, group: '' },
      };
      const preview = await resolveNewClassProposal(body, { dryRun: true });
      const ok = window.confirm(
        `Create class "${name}" and assign ${preview.matched} crop(s) proposing "${term.label}"?`,
      );
      if (!ok) return;
      const res = await resolveNewClassProposal(body);
      reportResolve(res, `Created "${res.class_name}"`);
      dismissProposalTerm(term.label);
      await Promise.all([loadProposals(), classesStore.clearAndRefetch()]);
    } catch (e) {
      toastStore.error(`Create & assign failed: ${apiErrorText(e)}`);
    } finally {
      proposalBusyTerm = null;
    }
  }

  async function mapToExisting(term: { label: string }): Promise<void> {
    const targetId = mapTargetByTerm[term.label];
    if (targetId == null) {
      toastStore.error('Pick an existing class first.');
      return;
    }
    proposalBusyTerm = term.label;
    try {
      const body: ResolveNewClassRequest = { label: term.label, class_id: targetId };
      const preview = await resolveNewClassProposal(body, { dryRun: true });
      const cls = classesStore.byId(targetId);
      const ok = window.confirm(
        `Assign ${preview.matched} crop(s) proposing "${term.label}" to "${cls?.name ?? targetId}"?`,
      );
      if (!ok) return;
      const res = await resolveNewClassProposal(body);
      reportResolve(res, `Assigned to "${cls?.name ?? res.class_name}"`);
      dismissProposalTerm(term.label);
      await loadProposals();
    } catch (e) {
      toastStore.error(`Assign failed: ${apiErrorText(e)}`);
    } finally {
      proposalBusyTerm = null;
    }
  }

  // DQ-M11 (dq-queues cutover, 2026-09-24): flagged_terms — super-category
  // ('generic_parent'), junk ('non_object') and already-registered
  // ('existing_class') proposed terms. No "create class" action is
  // offered for any of these (that was the original DQ-M11 bug — a
  // one-click create over 89 crops proposing a generic-parent term would
  // have made a super-class). `existing_class` still gets a one-click map action,
  // using the server's own `class_id`, not an operator-picked select.
  function flagReason(term: NewClassProposalTerm): string {
    if (term.flag === 'generic_parent') return 'generic parent';
    if (term.flag === 'non_object') return 'not an object';
    if (term.flag === 'existing_class') {
      const cls = term.class_id != null ? classesStore.byId(term.class_id) : null;
      return `existing class → map to ${cls?.name ?? term.class_id}`;
    }
    return '';
  }

  async function mapFlaggedTermToClass(term: NewClassProposalTerm): Promise<void> {
    if (term.class_id == null) return;
    proposalBusyTerm = term.label;
    try {
      const body: ResolveNewClassRequest = { label: term.label, class_id: term.class_id };
      const preview = await resolveNewClassProposal(body, { dryRun: true });
      const cls = classesStore.byId(term.class_id);
      const ok = window.confirm(
        `Assign ${preview.matched} crop(s) proposing "${term.label}" to "${cls?.name ?? term.class_id}"?`,
      );
      if (!ok) return;
      const res = await resolveNewClassProposal(body);
      reportResolve(res, `Assigned to "${cls?.name ?? res.class_name}"`);
      await loadProposals();
    } catch (e) {
      toastStore.error(`Assign failed: ${apiErrorText(e)}`);
    } finally {
      proposalBusyTerm = null;
    }
  }
</script>

<div class="mx-auto max-w-7xl p-6">
  <!-- Header -->
  <header class="mb-4 flex flex-wrap items-center gap-3">
    <h1 class="text-2xl font-semibold tracking-tight">Class management</h1>
    <span class="grow"></span>
    <input
      type="search"
      bind:value={query}
      placeholder="Filter by name, group, id…"
      class="input w-64 placeholder:text-zinc-500"
    />
    <button class="btn" type="button" onclick={openMerge} disabled={busy}
      >Merge classes</button
    >
    <button
      class="btn"
      type="button"
      onclick={() => void syncToOpensearch()}
      disabled={busy}
    >
      Sync to OpenSearch
    </button>
    <button class="btn btn-primary" type="button" onclick={openAdd} disabled={busy}>
      + Add Class
    </button>
  </header>

  {#if reservedConflicts.length > 0}
    <div
      class="mb-4 rounded-md border border-orange-500/40 bg-orange-500/10 px-4 py-3 text-xs text-orange-200"
    >
      <strong>Reserved-hotkey conflict:</strong>
      {#each reservedConflicts as c, i (c.id)}{i > 0 ? ', ' : ''}<strong>{c.name}</strong>
        → '{c.hotkey_letter}'{/each} — bound before this key became reserved for a labeling
      action. The binding is kept; rebind to a free letter when convenient.
    </div>
  {/if}

  <!-- Active classes. F-53: the registry is the page's primary content,
       so it renders first and the proposals list sits below it. F-50: no
       inner scroll pane; the page scrolls and the sticky header sticks to
       the main scroll container. -->
  <section class="surface overflow-x-auto">
    {#if classesStore.loading && allClasses.length === 0}
      <div class="p-6 text-sm text-zinc-500">Loading classes…</div>
    {:else if classesStore.error}
      <div class="p-6 text-sm text-red-300">API unavailable: {classesStore.error}</div>
    {:else if activeRows.length === 0}
      <div class="p-6 text-sm text-zinc-500">No classes match this filter.</div>
    {:else}
      <table class="w-full text-sm">
        <thead
          class="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-950 text-left text-xs uppercase text-zinc-400"
        >
          <tr>
            <!-- L5: ID and Added hide below lg (1024px) so the table fits 800px. -->
            <th class="hidden px-2 py-2 lg:px-3 font-medium lg:table-cell">ID</th>
            <th class="px-2 py-2 lg:px-3 font-medium">Name</th>
            <th class="px-2 py-2 lg:px-3 font-medium">Group</th>
            <th class="px-2 py-2 lg:px-3 text-center font-medium">Hotkey</th>
            <th
              class="px-2 py-2 lg:px-3 text-right font-medium"
              title="Human-validated crops, including any frozen as test holdout (shown as incl. N test)"
            >
              Validated
            </th>
            <!-- DQ-m9 (docs/design/data-quality-pass-2026-09-24.md): this
                 "Total" is `cluster_size` from GET /classes — the
                 class-cluster bucket size (what /clusters/{id} shows as
                 "in cluster"), NOT the same number as /export's "Total"
                 column (GET /stats/classes, every crop with that
                 class_id). They can legitimately disagree a lot — a
                 region-bound class can count region boxes (sub-boxes
                 counted in the cluster bucket) here but 0 on /export (no
                 crop's own class_id is the region class). -->
            <th
              class="px-2 py-2 lg:px-3 text-right font-medium"
              title="Class-cluster size (cluster_size) — the same number the cluster page and sidebar show as &quot;in cluster&quot;. Not the same as /export's Total column."
            >
              Total (in cluster)
            </th>
            <th class="hidden px-2 py-2 lg:px-3 font-medium lg:table-cell">Added</th>
            <th class="px-2 py-2 lg:px-3"></th>
          </tr>
        </thead>
        <tbody>
          {#each activeRows as cls (cls.id)}
            {@const heldOut = testHeldOutByClass.get(cls.id) ?? 0}
            <tr
              class="border-b border-zinc-900 hover:bg-zinc-900/40"
              data-testid="class-row-{cls.id}"
            >
              <td
                class="hidden px-2 py-1.5 lg:px-3 font-mono text-xs text-zinc-400 lg:table-cell"
                >{cls.id}</td
              >
              <td class="px-2 py-1.5 lg:px-3">
                {#if editingId === cls.id}
                  <input
                    type="text"
                    bind:value={editName}
                    class="w-full rounded border border-blue-500/60 bg-zinc-900 px-1.5 py-0.5 text-sm focus:outline-none"
                    onkeydown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        void commitRename(cls);
                      } else if (e.key === 'Escape') {
                        e.preventDefault();
                        cancelEdit();
                      }
                    }}
                    use:focusOnMount={{ select: true }}
                  />
                {:else}
                  <button
                    type="button"
                    class="rounded px-1 py-0.5 text-left hover:bg-zinc-800"
                    onclick={() => startEdit(cls)}
                    title="Click to rename"
                  >
                    {cls.name}
                  </button>
                {/if}
              </td>
              <td class="px-2 py-1.5 lg:px-3">
                <select
                  class="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-xs focus:border-blue-500 focus:outline-none"
                  value={cls.group ?? ''}
                  onchange={(e) =>
                    void changeGroup(cls, (e.currentTarget as HTMLSelectElement).value)}
                  disabled={busy}
                >
                  {#each groups as g (g)}
                    <option value={g}>{g}</option>
                  {/each}
                  {#if cls.group && !groups.includes(cls.group)}
                    <option value={cls.group}>{cls.group}</option>
                  {/if}
                </select>
              </td>
              <td class="px-2 py-1.5 lg:px-3 text-center">
                <input
                  type="text"
                  maxlength="1"
                  value={cls.hotkey_letter ?? ''}
                  placeholder="—"
                  title="Single char keyboard shortcut for this class. Empty to clear."
                  class="w-10 rounded border border-zinc-700 bg-zinc-900 px-1 py-0.5 text-center font-mono text-xs uppercase focus:border-blue-500 focus:outline-none"
                  onchange={(e) =>
                    void setHotkey(cls, (e.currentTarget as HTMLInputElement).value)}
                  disabled={busy}
                />
              </td>
              <td class="px-2 py-1.5 lg:px-3 text-right font-mono">
                <span
                  class="rounded-md border px-1.5 py-0.5 text-xs {adequacyChipClass(
                    cls.adequacy,
                  )}"
                  title={adequacyTooltip(cls.adequacy, cls.validated_count ?? 0)}
                >
                  {cls.validated_count ?? 0}
                </span>
                {#if heldOut > 0}
                  <span
                    class="ml-1 whitespace-nowrap text-[10px] text-zinc-500"
                    data-testid="validated-test-suffix">incl. {heldOut} test</span
                  >
                {/if}
              </td>
              <!-- L1 (visual audit 2026-09-24): this showed sample_count
                   (271 for one live class) under a header promising the cluster page's
                   "in cluster" number (270) — it now shows cluster_size. -->
              <td
                class="px-2 py-1.5 lg:px-3 text-right font-mono text-zinc-400"
                data-testid="in-cluster">{cls.cluster_size ?? 0}</td
              >
              <td class="hidden px-2 py-1.5 lg:px-3 text-xs text-zinc-500 lg:table-cell">
                {formatDateOnly(cls.added_at)}
              </td>
              <td class="px-2 py-1.5 lg:px-3 text-right">
                {#if editingId === cls.id}
                  <button
                    type="button"
                    class="btn"
                    onclick={() => void commitRename(cls)}
                    disabled={busy}
                  >
                    Save
                  </button>
                  <button type="button" class="btn" onclick={cancelEdit} disabled={busy}>
                    Cancel
                  </button>
                {:else}
                  <button
                    type="button"
                    class="btn"
                    onclick={() => startEdit(cls)}
                    disabled={busy}
                  >
                    Rename
                  </button>
                  <button
                    type="button"
                    class="btn"
                    data-testid="deprecate-{cls.id}"
                    onclick={() => void deprecateClassAction(cls)}
                    disabled={busy}
                  >
                    Deprecate
                  </button>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </section>

  <!-- Deprecated section -->
  {#if deprecatedRows.length > 0}
    <section class="mt-4">
      <button
        type="button"
        class="flex items-center gap-2 text-xs text-zinc-400 hover:text-zinc-200"
        onclick={() => (showDeprecated = !showDeprecated)}
      >
        <span aria-hidden="true">{showDeprecated ? '▼' : '▶'}</span>
        Deprecated classes ({deprecatedRows.length})
      </button>
      {#if showDeprecated}
        <div class="surface mt-2 overflow-auto">
          <table class="w-full text-sm">
            <thead
              class="border-b border-zinc-800 text-left text-xs uppercase text-zinc-500"
            >
              <tr>
                <th class="px-3 py-2 font-medium">ID</th>
                <th class="px-3 py-2 font-medium">Name</th>
                <th class="px-3 py-2 font-medium">Group</th>
                <th class="px-3 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {#each deprecatedRows as cls (cls.id)}
                <tr class="border-b border-zinc-900 text-zinc-500">
                  <td class="px-3 py-1.5 font-mono text-xs">{cls.id}</td>
                  <td class="px-3 py-1.5 line-through">{cls.name}</td>
                  <td class="px-3 py-1.5">{cls.group ?? '—'}</td>
                  <td class="px-3 py-1.5 text-right">
                    {#if cls.merged_into != null}
                      <!-- 51b05d7: a merged class can't be restored (its
                           crops live on the target); say where they went. -->
                      <span
                        class="text-xs text-zinc-500"
                        data-testid="merged-into-{cls.id}"
                      >
                        merged into {classesStore.byId(cls.merged_into)?.name ??
                          `#${cls.merged_into}`}
                      </span>
                    {:else}
                      <button
                        type="button"
                        class="btn"
                        data-testid="restore-{cls.id}"
                        onclick={() => void restoreClassAction(cls)}
                        disabled={busy}
                      >
                        Restore
                      </button>
                    {/if}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </section>
  {/if}

  <SeedFromDetectorPanel />

  <!-- New-class proposals (2026-09-24 logic-moves W5) — aggregate view of
       the same cohort /review's "New Class Proposals" tab pages one crop
       at a time. Absent (not shown as an error banner) while nothing has
       loaded yet or the pool is empty; shown as an inline error when the
       backend genuinely failed (e.g. the opensearch aggregation 500 seen
       live), never a page-breaking crash. -->
  {#if proposalsLoading}
    <div class="surface mt-4 p-4 text-xs text-zinc-500">Loading proposals…</div>
  {:else if proposalsError}
    <div class="surface mt-4 flex items-center gap-3 p-4 text-xs text-red-300">
      <span>Proposals unavailable: {proposalsError}</span>
      <button type="button" class="btn-sm" onclick={() => void loadProposals()}>
        retry
      </button>
    </div>
  {:else if proposalsSummary && (proposalsSummary.top_terms.length > 0 || proposalsSummary.flagged_terms.length > 0)}
    <details class="surface mt-4 p-4" data-testid="proposals-section">
      <summary class="cursor-pointer text-sm font-semibold text-zinc-200">
        New class proposals
        <span class="ml-1 font-normal text-zinc-500"
          >({proposalsSummary.total_pending} pending{#if proposalsSummary.without_term > 0},
            {proposalsSummary.without_term} with no proposed term{/if})</span
        >
      </summary>
      <p class="mb-3 mt-2 text-xs text-zinc-500">
        Crops the VLM flagged as needing a class the registry doesn't have yet. Each row
        is a proposed term with a few sample thumbnails — create a class or map onto an
        existing one to resolve <strong>every</strong> pending crop proposing that term
        (not just the samples shown), after a confirm step showing the real count. Crops
        can still be triaged one at a time on <code>/review</code>'s "New Class Proposals"
        tab instead.
        {#if proposalsSummary.without_term > 0}
          {proposalsSummary.without_term} pending item(s) have no proposed term (a human "needs
          new class" flag with no name) and aren't listed below — triage those on
          <code>/review</code> instead.
        {/if}
      </p>
      {#if proposalsSummary.term_rules}
        {@const rules = termRulesText(proposalsSummary.term_rules)}
        {#if rules}
          <p class="mb-3 text-[11px] text-zinc-600" data-testid="proposal-term-rules">
            {rules}
          </p>
        {/if}
      {/if}
      <!-- L2 (visual audit 2026-09-24): one list, biggest term first.
           Flagged terms (generic parent / not an object / existing class)
           used to sit in a collapsed section with no action at all while
           holding most pending crops; they now get "map to existing"
           too. Only "Create class" stays limited to un-flagged terms. -->
      <ul class="flex flex-col gap-3" data-testid="proposal-rows">
        {#each proposalRows(proposalsSummary) as row (row.term.label)}
          {@const term = row.term}
          <li
            class="flex flex-wrap items-center gap-3 rounded border border-zinc-800 p-2"
            data-testid="proposal-row"
          >
            <div class="flex shrink-0 items-center gap-1">
              {#each term.sample_crop_ids.slice(0, 4) as cropId (cropId)}
                <img
                  src={getThumbUrl(cropId, 64)}
                  alt=""
                  loading="lazy"
                  class="h-10 w-10 rounded object-cover"
                />
              {/each}
            </div>
            <div class="min-w-0 shrink-0">
              <div class="text-sm text-zinc-100">{term.label}</div>
              <div class="text-[11px] text-zinc-500">
                {term.count} crop(s)
                {#if term.flag}
                  · <span class="text-amber-300">{flagReason(term)}</span>
                {/if}
              </div>
            </div>
            <span class="grow"></span>
            {#if row.canCreate}
              <div class="flex shrink-0 items-center gap-1.5">
                <input
                  type="text"
                  placeholder={term.label}
                  value={newClassNameByTerm[term.label] ?? ''}
                  oninput={(e) => {
                    newClassNameByTerm = {
                      ...newClassNameByTerm,
                      [term.label]: (e.currentTarget as HTMLInputElement).value,
                    };
                  }}
                  class="input-sm w-32"
                  disabled={proposalBusyTerm === term.label}
                />
                <button
                  type="button"
                  class="btn-sm btn-primary"
                  disabled={proposalBusyTerm === term.label}
                  onclick={() => void createClassAndAssign(term)}
                >
                  Create class & assign
                </button>
              </div>
            {/if}
            {#if row.fixedMapClassId != null}
              <button
                type="button"
                class="btn-sm"
                disabled={proposalBusyTerm === term.label}
                onclick={() => void mapFlaggedTermToClass(term)}
              >
                Map to {classesStore.byId(row.fixedMapClassId)?.name ??
                  row.fixedMapClassId}
              </button>
            {/if}
            {#if row.canPickMapTarget}
              <div class="flex shrink-0 items-center gap-1.5">
                <select
                  class="select-sm"
                  aria-label="Map {term.label} to an existing class"
                  value={mapTargetByTerm[term.label] ?? ''}
                  onchange={(e) => {
                    const v = (e.currentTarget as HTMLSelectElement).value;
                    mapTargetByTerm = {
                      ...mapTargetByTerm,
                      [term.label]: v === '' ? null : Number(v),
                    };
                  }}
                  disabled={proposalBusyTerm === term.label}
                >
                  <option value="">map to existing…</option>
                  {#each allClasses.filter((c) => !c.deprecated) as cls (cls.id)}
                    <option value={cls.id}>{cls.name}</option>
                  {/each}
                </select>
                <button
                  type="button"
                  class="btn-sm"
                  disabled={proposalBusyTerm === term.label ||
                    mapTargetByTerm[term.label] == null}
                  onclick={() => void mapToExisting(term)}
                >
                  Assign
                </button>
              </div>
            {/if}
            <button
              type="button"
              class="btn-sm btn-icon"
              title="Hide this term from the list until the page reloads. Not saved: the crops stay pending."
              aria-label="Hide {term.label} until reload (not saved)"
              onclick={() => dismissProposalTerm(term.label)}
            >
              Hide
            </button>
          </li>
        {/each}
      </ul>
    </details>
  {/if}
</div>

<AddClassModal open={addOpen} onclose={() => (addOpen = false)} />

<!-- Merge modal -->
{#if mergeOpen}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Merge classes"
    tabindex="-1"
    use:focusOnMount
    use:trapFocus={{ onEscape: closeMerge }}
    onclick={(e) => {
      if (e.target === e.currentTarget) closeMerge();
    }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-3 text-base font-semibold">Merge classes</h3>
      <p class="mb-3 text-xs text-zinc-400">
        Source class is marked deprecated; all crops + label rows are relabeled to the
        target.
      </p>

      <label class="mb-3 block text-sm">
        <span class="mb-1 block text-zinc-400">Source (will be merged away)</span>
        <select
          bind:value={mergeSourceId}
          class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
        >
          <option value={null}>— pick source —</option>
          {#each allClasses.filter((c) => !c.deprecated) as c (c.id)}
            <option value={c.id}>{c.name} ({c.validated_count ?? 0})</option>
          {/each}
        </select>
      </label>

      <label class="mb-3 block text-sm">
        <span class="mb-1 block text-zinc-400">Target (kept)</span>
        <select
          bind:value={mergeTargetId}
          class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
        >
          <option value={null}>— pick target —</option>
          {#each allClasses.filter((c) => !c.deprecated && c.id !== mergeSourceId) as c (c.id)}
            <option value={c.id}>{c.name} ({c.validated_count ?? 0})</option>
          {/each}
        </select>
      </label>

      {#if mergeSource && mergeTarget}
        {#if mergePreviewBusy}
          <div class="mb-3 text-xs text-zinc-500">Checking impact…</div>
        {:else if mergePreviewError}
          <div
            class="mb-3 rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200"
          >
            Preview failed: {mergePreviewError}
          </div>
        {:else if mergePreview}
          <div
            class="mb-3 rounded border px-3 py-2 text-xs {mergePreview.blocked
              ? 'border-red-500/40 bg-red-500/10 text-red-200'
              : 'border-orange-500/40 bg-orange-500/10 text-orange-200'}"
          >
            Will relabel <strong>{mergePreview.would_relabel}</strong> crops from
            <strong>{mergeSource.name}</strong> to <strong>{mergeTarget.name}</strong>
            {#if mergePreview.validations_carried_over > 0}
              <!-- 51b05d7: a merge keeps human validations. -->
              &middot; <strong>{mergePreview.validations_carried_over}</strong> human
              validation{mergePreview.validations_carried_over === 1 ? '' : 's'} will carry
              over
            {/if}
            {#if mergePreview.holdout_blocking > 0}
              &middot; <strong>{mergePreview.holdout_blocking}</strong> test-holdout crops block
              this merge
            {/if}
            {#if mergePreview.blocked}
              <div class="mt-1 font-semibold">
                Blocked — a real merge will 409 until the holdout conflict is resolved.
              </div>
            {/if}
          </div>
        {/if}
      {/if}

      <div class="flex justify-end gap-2">
        <button type="button" class="btn" onclick={closeMerge} disabled={busy}>
          Cancel
        </button>
        <button
          type="button"
          class="btn btn-primary"
          onclick={() => void submitMerge()}
          disabled={busy ||
            mergeSourceId == null ||
            mergeTargetId == null ||
            mergePreviewBusy ||
            mergePreview == null ||
            mergePreview?.blocked}
        >
          {busy ? 'Merging…' : 'Merge'}
        </button>
      </div>
    </div>
  </div>
{/if}
