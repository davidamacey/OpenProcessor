<script lang="ts">
  import { mergeClasses, renameClass, syncClassesToOpensearch } from '$lib/api';
  import AddClassModal from '$components/AddClassModal.svelte';
  import { adequacyChipClass, adequacyTooltip } from '$lib/adequacy';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { setClassHotkey } from '$lib/classHotkey';
  import type { OpClass } from '$lib/types';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

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

  const SLUG_RE = /^[a-z0-9_]+$/;

  function isSlug(v: string): boolean {
    return v.length > 0 && SLUG_RE.test(v);
  }

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

  function startEdit(cls: OpClass): void {
    editingId = cls.id;
    editName = cls.name;
  }

  function cancelEdit(): void {
    editingId = null;
    editName = '';
  }

  async function commitRename(cls: OpClass): Promise<void> {
    const next = editName.trim();
    if (!isSlug(next)) {
      toastStore.error('Class name must be lowercase a-z, 0-9, _ only.');
      return;
    }
    if (next === cls.name) {
      cancelEdit();
      return;
    }
    busy = true;
    try {
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

  async function changeGroup(cls: OpClass, newG: string): Promise<void> {
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

  async function setHotkey(cls: OpClass, raw: string): Promise<void> {
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
    mergeOpen = true;
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

  async function submitMerge(): Promise<void> {
    if (mergeSourceId == null || mergeTargetId == null) {
      toastStore.error('Pick both source and target classes.');
      return;
    }
    if (mergeSourceId === mergeTargetId) {
      toastStore.error('Source and target must differ.');
      return;
    }
    const ok = window.confirm(
      `Merge "${mergeSource?.name}" into "${mergeTarget?.name}"? This relabels ` +
        `${mergeSource?.validated_count ?? 0} validated crops and marks the source deprecated. ` +
        'The action is recorded in the registry but cannot be undone from the UI.',
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
      const created = res.created ?? 0;
      const updated = res.updated ?? 0;
      toastStore.success(
        `Synced to OpenSearch (${created} created, ${updated} updated).`,
      );
    } catch (e) {
      toastStore.error(`Sync failed: ${(e as Error).message}`);
    } finally {
      busy = false;
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
            <th class="px-3 py-2 text-right font-medium">Total</th>
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
                    cls.validated_count ?? 0,
                  )}"
                  title={adequacyTooltip(cls.validated_count ?? 0)}
                >
                  {cls.validated_count ?? 0}
                </span>
              </td>
              <td class="px-3 py-1.5 text-right font-mono text-zinc-400"
                >{cls.count ?? 0}</td
              >
              <td class="px-3 py-1.5 text-xs text-zinc-500">
                {cls.added_at ? new Date(cls.added_at).toLocaleDateString() : '—'}
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
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Merge classes"
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
        <div
          class="mb-3 rounded border border-orange-500/40 bg-orange-500/10 px-3 py-2 text-xs text-orange-200"
        >
          Will relabel <strong>{mergeSource.validated_count ?? 0}</strong> validated crops
          from
          <strong>{mergeSource.name}</strong> to <strong>{mergeTarget.name}</strong>.
        </div>
      {/if}

      <div class="flex justify-end gap-2">
        <button
          type="button"
          class="btn"
          onclick={() => (mergeOpen = false)}
          disabled={busy}
        >
          Cancel
        </button>
        <button
          type="button"
          class="btn btn-primary"
          onclick={() => void submitMerge()}
          disabled={busy || mergeSourceId == null || mergeTargetId == null}
        >
          {busy ? 'Merging…' : 'Merge'}
        </button>
      </div>
    </div>
  </div>
{/if}
