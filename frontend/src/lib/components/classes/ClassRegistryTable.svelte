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
    <div class="p-6 text-body text-fg-3">Loading classes…</div>
  {:else if classesStore.error}
    <div class="p-6 text-body text-danger-text">
      API unavailable: {classesStore.error}
    </div>
  {:else if allClasses.length === 0}
    <div class="p-6 text-body text-fg-3">
      No classes yet. Add one, or create them from the detector below.
    </div>
  {:else if activeRows.length === 0}
    <div class="p-6 text-body text-fg-3">No classes match this filter.</div>
  {:else}
    <table class="w-full text-body">
      <thead
        class="sticky top-0 z-10 border-b border-line bg-surface-0 text-left text-meta uppercase text-fg-2"
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
          <th
            class="px-2 py-2 lg:px-3 text-right font-medium"
            title="Items carrying this class (GET /classes sample_count). Not the same as /export's Total column."
          >
            Items
          </th>
          <th class="hidden px-2 py-2 lg:px-3 font-medium lg:table-cell">Added</th>
          <th class="px-2 py-2 lg:px-3"><span class="sr-only">Actions</span></th>
        </tr>
      </thead>
      <tbody>
        {#each activeRows as cls (cls.id)}
          {@const heldOut = testHeldOutByClass.get(cls.id) ?? 0}
          <tr
            class="border-b border-line hover:bg-surface-1/40"
            data-testid="class-row-{cls.id}"
          >
            <td
              class="hidden px-2 py-1.5 lg:px-3 font-mono text-meta text-fg-2 lg:table-cell"
              >{cls.id}</td
            >
            <td class="px-2 py-1.5 lg:px-3">
              {#if editingId === cls.id}
                <input
                  type="text"
                  bind:value={editName}
                  class="w-full rounded border border-accent/60 bg-surface-1 px-1.5 py-0.5 text-body focus:outline-none"
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
                  class="rounded px-1 py-0.5 text-left hover:bg-surface-2"
                  onclick={() => startEdit(cls)}
                  title="Click to rename"
                >
                  {cls.name}
                </button>
              {/if}
            </td>
            <td class="px-2 py-1.5 lg:px-3">
              <select
                class="rounded border border-line-strong bg-surface-1 px-1.5 py-0.5 text-meta focus:border-accent focus:outline-none"
                aria-label="Group for {cls.name}"
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
                aria-label="Hotkey for {cls.name}"
                title="Single char keyboard shortcut for this class. Empty to clear."
                class="w-10 rounded border border-line-strong bg-surface-1 px-1 py-0.5 text-center font-mono text-meta uppercase focus:border-accent focus:outline-none"
                onchange={(e) =>
                  void setHotkey(cls, (e.currentTarget as HTMLInputElement).value)}
                disabled={busy}
              />
            </td>
            <td class="px-2 py-1.5 lg:px-3 text-right font-mono">
              <span
                class="rounded-md border px-1.5 py-0.5 text-meta {adequacyChipClass(
                  cls.adequacy,
                )}"
                title={adequacyTooltip(cls.adequacy, cls.validated_count ?? 0)}
              >
                {cls.validated_count ?? 0}
              </span>
              {#if heldOut > 0}
                <span
                  class="ml-1 whitespace-nowrap text-caption text-fg-3"
                  data-testid="validated-test-suffix">incl. {heldOut} test</span
                >
              {/if}
            </td>
            <td
              class="px-2 py-1.5 lg:px-3 text-right font-mono text-fg-2"
              data-testid="item-count">{cls.count}</td
            >
            <td class="hidden px-2 py-1.5 lg:px-3 text-meta text-fg-3 lg:table-cell">
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
