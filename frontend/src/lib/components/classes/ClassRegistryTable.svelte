<script lang="ts">
  import { adequacyChipClass, adequacyTooltip } from '$lib/adequacy';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { formatDateOnly } from '$lib/format/date';
  import type { RegistryClass } from '$lib/types_classes';
  import { classesStore } from '$stores/classes.svelte';

  // The active-class registry table on /classes. Row actions are the page's
  // own handlers; the rename draft is bound back to the page.
  let {
    allClasses,
    activeRows,
    groups,
    busy,
    testHeldOutByClass,
    editingId,
    editName = $bindable(),
    startEdit,
    cancelEdit,
    commitRename,
    changeGroup,
    setHotkey,
    deprecateClassAction,
  }: {
    allClasses: RegistryClass[];
    activeRows: RegistryClass[];
    groups: string[];
    busy: boolean;
    testHeldOutByClass: Map<number, number>;
    editingId: number | null;
    editName: string;
    startEdit: (cls: RegistryClass) => void;
    cancelEdit: () => void;
    commitRename: (cls: RegistryClass) => Promise<void>;
    changeGroup: (cls: RegistryClass, newG: string) => Promise<void>;
    setHotkey: (cls: RegistryClass, raw: string) => Promise<void>;
    deprecateClassAction: (cls: RegistryClass) => Promise<void>;
  } = $props();
</script>

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
