<script lang="ts">
  import { apiErrorText, ApiError } from '$lib/api_errors';
  import {
    classStillReferencedDetail,
    deprecateClass,
    renameClass,
    restoreClass,
    classMergedDetail,
    classMergedRestoreText,
    syncClassesToOpensearch,
  } from '$lib/api_classes';
  import { getTestHoldoutStats } from '$lib/api_export';
  import AddClassModal from '$components/AddClassModal.svelte';
  import SeedFromDetectorPanel from '$lib/components/detector/SeedFromDetectorPanel.svelte';
  import ClassRegistryTable from '$lib/components/classes/ClassRegistryTable.svelte';
  import DeprecatedClassesTable from '$lib/components/classes/DeprecatedClassesTable.svelte';
  import MergeClassesModal from '$lib/components/classes/MergeClassesModal.svelte';
  import NewClassProposalsPanel from '$lib/components/classes/NewClassProposalsPanel.svelte';
  import { createMergeClassesController } from '$lib/classes/mergeClassesController.svelte';
  import { createNewClassProposalsController } from '$lib/classes/newClassProposalsController.svelte';
  import { reservedHotkeyLetters, setClassHotkey } from '$lib/classHotkey';
  import type { RegistryClass } from '$lib/types_classes';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';
  import { onMount } from 'svelte';

  $effect(() => {
    keyboardStore.setScope('classes');
  });

  // ---- table state -------------------------------------------------------

  let query = $state<string>('');

  // Inline rename state — only one row may be in edit mode at a time.
  let editingId = $state<number | null>(null);
  let editName = $state<string>('');

  // Modal state
  let addOpen = $state<boolean>(false);
  let busy = $state<boolean>(false);

  const merge = createMergeClassesController({
    get busy() {
      return busy;
    },
    set busy(v: boolean) {
      busy = v;
    },
  });
  const proposals = createNewClassProposalsController();

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
        if (goMerge) merge.openMergeWithSource(cls.id);
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
      // a merged class 409s with a structured `class_merged` detail
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

  onMount(() => void proposals.loadProposals());

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
    <button class="btn" type="button" onclick={merge.openMerge} disabled={busy}
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

  <!-- Active classes. The registry is the page's primary content,
       so it renders first and the proposals list sits below it. No
       inner scroll pane; the page scrolls and the sticky header sticks to
       the main scroll container. -->
  <ClassRegistryTable
    {allClasses}
    {activeRows}
    {groups}
    {busy}
    {testHeldOutByClass}
    {editingId}
    bind:editName
    {startEdit}
    {cancelEdit}
    {commitRename}
    {changeGroup}
    {setHotkey}
    {deprecateClassAction}
  />

  <DeprecatedClassesTable {deprecatedRows} {busy} {restoreClassAction} />

  <SeedFromDetectorPanel />

  <NewClassProposalsPanel p={proposals} {allClasses} />
</div>

<AddClassModal open={addOpen} onclose={() => (addOpen = false)} />

<MergeClassesModal m={merge} {allClasses} {busy} />
