<script lang="ts">
  import {
    getNewClassProposalsSummary,
    getThumbUrl,
    mergeClasses,
    previewClassMerge,
    renameClass,
    resolveNewClassProposal,
    syncClassesToOpensearch,
    type NewClassProposalsSummary,
    type NewClassProposalTerm,
  } from '$lib/api';
  import AddClassModal from '$components/AddClassModal.svelte';
  import { adequacyChipClass, adequacyTooltip } from '$lib/adequacy';
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
      toastStore.error(`Rename failed: ${(e as Error).message}`);
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
      toastStore.error(`Group change failed: ${(e as Error).message}`);
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
  // real merge would do (would_relabel / would_unvalidate / holdout_blocking
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
        mergePreviewError = (e as Error).message;
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
    if (mergePreview?.blocked) {
      toastStore.error('Merge is blocked — resolve the holdout conflict first.');
      return;
    }
    const ok = window.confirm(
      `Merge "${mergeSource?.name}" into "${mergeTarget?.name}"? This relabels ` +
        `${mergePreview?.would_relabel ?? mergeSource?.validated_count ?? 0} crops and marks the ` +
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
      toastStore.error(`Merge failed: ${(e as Error).message}`);
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
      toastStore.error(`Sync failed: ${(e as Error).message}`);
    } finally {
      busy = false;
    }
  }

  // -- New-class proposals (2026-09-24 logic-moves W5; bulk resolve added
  //    2026-09-24 for OpenProcessor af3a580) --------------------------------
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
      proposalsError = (e as Error).message;
    } finally {
      proposalsLoading = false;
    }
  }

  onMount(() => void loadProposals());

  function dismissProposalTerm(label: string): void {
    if (!proposalsSummary) return;
    proposalsSummary = {
      ...proposalsSummary,
      top_terms: proposalsSummary.top_terms.filter((t) => t.label !== label),
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
      toastStore.error(`Create & assign failed: ${(e as Error).message}`);
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
      toastStore.error(`Assign failed: ${(e as Error).message}`);
    } finally {
      proposalBusyTerm = null;
    }
  }

  // DQ-M11 (dq-queues cutover, 2026-09-24): flagged_terms — super-category
  // ('generic_parent'), junk ('non_object') and already-registered
  // ('existing_class') proposed terms. No "create class" action is
  // offered for any of these (that was the original DQ-M11 bug — a
  // one-click create over 89 "motorcycle" crops would have made a
  // super-class). `existing_class` still gets a one-click map action,
  // using the server's own `class_id`, not an operator-picked select.
  let flaggedSectionOpen = $state<boolean>(false);

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
      toastStore.error(`Assign failed: ${(e as Error).message}`);
    } finally {
      proposalBusyTerm = null;
    }
  }
</script>

<div class="mx-auto flex h-full max-w-7xl flex-col p-6">
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

  <!-- New-class proposals (2026-09-24 logic-moves W5) — aggregate view of
       the same cohort /review's "New Class Proposals" tab pages one crop
       at a time. Absent (not shown as an error banner) while nothing has
       loaded yet or the pool is empty; shown as an inline error when the
       backend genuinely failed (e.g. the opensearch aggregation 500 seen
       live), never a page-breaking crash. -->
  {#if proposalsLoading}
    <div class="surface mb-4 p-4 text-xs text-zinc-500">Loading proposals…</div>
  {:else if proposalsError}
    <div class="surface mb-4 flex items-center gap-3 p-4 text-xs text-red-300">
      <span>Proposals unavailable: {proposalsError}</span>
      <button type="button" class="btn-sm" onclick={() => void loadProposals()}>
        retry
      </button>
    </div>
  {:else if proposalsSummary && (proposalsSummary.top_terms.length > 0 || proposalsSummary.flagged_terms.length > 0)}
    <section class="surface mb-4 p-4">
      <h2 class="mb-1 text-sm font-semibold text-zinc-200">
        New class proposals
        <span class="ml-1 font-normal text-zinc-500"
          >({proposalsSummary.total_pending} pending{#if proposalsSummary.without_term > 0},
            {proposalsSummary.without_term} with no proposed term{/if})</span
        >
      </h2>
      <p class="mb-3 text-xs text-zinc-500">
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
        <p class="mb-3 text-[11px] text-zinc-600">
          Terms are auto-flagged, not offered a one-click create, when they match this
          deployment's generic-parent list ({proposalsSummary.term_rules.generic_terms.join(
            ', ',
          )}{proposalsSummary.term_rules.registry_groups_are_generic
            ? ', plus any existing class-registry group name'
            : ''}) or non-object list ({proposalsSummary.term_rules.non_object_terms.join(
            ', ',
          )}), or when the term already names a registered class.
        </p>
      {/if}
      {#if proposalsSummary.top_terms.length === 0}
        <p class="mb-3 text-xs text-zinc-500">
          No actionable terms right now — see "flagged terms" below.
        </p>
      {/if}
      <ul class="flex flex-col gap-3">
        {#each proposalsSummary.top_terms as term (term.label)}
          <li
            class="flex flex-wrap items-center gap-3 rounded border border-zinc-800 p-2"
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
              <div class="text-[11px] text-zinc-500">{term.count} crop(s)</div>
            </div>
            <span class="grow"></span>
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
            <div class="flex shrink-0 items-center gap-1.5">
              <select
                class="select-sm"
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
            <button
              type="button"
              class="btn-sm btn-icon"
              title="Dismiss this term from the list (doesn't touch the crops)"
              onclick={() => dismissProposalTerm(term.label)}
            >
              ×
            </button>
          </li>
        {/each}
      </ul>

      {#if proposalsSummary.flagged_terms.length > 0}
        <details
          class="mt-3 rounded border border-zinc-800 p-2"
          bind:open={flaggedSectionOpen}
        >
          <summary class="cursor-pointer text-xs text-zinc-400">
            Flagged terms ({proposalsSummary.flagged_terms.length}) — not offered a
            one-click create
          </summary>
          <ul class="mt-2 flex flex-col gap-2">
            {#each proposalsSummary.flagged_terms as term (term.label)}
              <li
                class="flex flex-wrap items-center gap-3 rounded border border-zinc-800/60 p-2"
              >
                <div class="min-w-0 shrink-0">
                  <div class="text-sm text-zinc-200">{term.label}</div>
                  <div class="text-[11px] text-zinc-500">
                    {term.count} crop(s) ·
                    <span class="text-amber-300">{flagReason(term)}</span>
                  </div>
                </div>
                <span class="grow"></span>
                {#if term.flag === 'existing_class' && term.class_id != null}
                  <button
                    type="button"
                    class="btn-sm"
                    disabled={proposalBusyTerm === term.label}
                    onclick={() => void mapFlaggedTermToClass(term)}
                  >
                    Map to {classesStore.byId(term.class_id)?.name ?? term.class_id}
                  </button>
                {/if}
              </li>
            {/each}
          </ul>
        </details>
      {/if}
    </section>
  {/if}

  <!-- Active classes -->
  <section class="surface flex-1 overflow-auto">
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
            <th class="px-3 py-2 font-medium">ID</th>
            <th class="px-3 py-2 font-medium">Name</th>
            <th class="px-3 py-2 font-medium">Group</th>
            <th class="px-3 py-2 text-center font-medium">Hotkey</th>
            <th class="px-3 py-2 text-right font-medium">Validated</th>
            <!-- DQ-m9 (docs/design/data-quality-pass-2026-09-24.md): this
                 "Total" is `sample_count` from GET /classes — the
                 class-cluster bucket size (what /clusters/{id} shows as
                 "in cluster"), NOT the same number as /export's "Total"
                 column (GET /stats/classes, every crop with that
                 class_id). They can legitimately disagree a lot — a
                 region-bound class can count region boxes (sub-boxes
                 counted in the cluster bucket) here but 0 on /export (no
                 crop's own class_id is the region class). -->
            <th
              class="px-3 py-2 text-right font-medium"
              title="Class-cluster bucket size (sample_count) — matches the per-cluster page's &quot;in cluster&quot; count. Not the same as /export's Total column."
            >
              Total (in cluster)
            </th>
            <th class="px-3 py-2 font-medium">Added</th>
            <th class="px-3 py-2"></th>
          </tr>
        </thead>
        <tbody>
          {#each activeRows as cls (cls.id)}
            <tr class="border-b border-zinc-900 hover:bg-zinc-900/40">
              <td class="px-3 py-1.5 font-mono text-xs text-zinc-400">{cls.id}</td>
              <td class="px-3 py-1.5">
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
              <td class="px-3 py-1.5">
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
              <td class="px-3 py-1.5 text-center">
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
              <td class="px-3 py-1.5 text-right font-mono">
                <span
                  class="rounded-md border px-1.5 py-0.5 text-xs {adequacyChipClass(
                    cls.adequacy,
                  )}"
                  title={adequacyTooltip(cls.adequacy, cls.validated_count ?? 0)}
                >
                  {cls.validated_count ?? 0}
                </span>
              </td>
              <td class="px-3 py-1.5 text-right font-mono text-zinc-400"
                >{cls.count ?? 0}</td
              >
              <td class="px-3 py-1.5 text-xs text-zinc-500">
                {formatDateOnly(cls.added_at)}
              </td>
              <td class="px-3 py-1.5 text-right">
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
                    <button
                      type="button"
                      class="btn"
                      disabled
                      title="Restore not yet implemented"
                    >
                      Restore
                    </button>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </section>
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
            {#if mergePreview.would_unvalidate > 0}
              &middot; <strong>{mergePreview.would_unvalidate}</strong> lose validation
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
            mergePreview?.blocked}
        >
          {busy ? 'Merging…' : 'Merge'}
        </button>
      </div>
    </div>
  </div>
{/if}
