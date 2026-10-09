<script lang="ts">
  import { apiErrorText } from '$lib/api';
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import { addClass } from '$lib/api';
  import { classesStore } from '$stores/classes.svelte';
  import { toastStore } from '$stores/toast.svelte';

  interface Props {
    open: boolean;
    onclose: () => void;
    /** Optional callback fired after a successful add (e.g., navigate). */
    oncreated?: (created: { name: string; group: string }) => void;
  }

  let { open, onclose, oncreated }: Props = $props();

  let name = $state<string>('');
  let groupExisting = $state<string>('');
  let groupNew = $state<string>('');
  let createNewGroup = $state<boolean>(false);
  let notes = $state<string>('');
  let busy = $state<boolean>(false);
  let errorText = $state<string | null>(null);

  const groups = $derived.by(() => {
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local set built and consumed within this computation, never stored in reactive state (only the resulting sorted array is)
    const set = new Set<string>();
    for (const c of classesStore.classes) {
      const g = c.group ?? '';
      if (g) set.add(g);
    }
    return [...set].sort();
  });

  // When the modal opens, default the group selector sensibly: pick the first
  // existing group if any exist; otherwise force the new-group flow on.
  $effect(() => {
    if (!open) return;
    name = '';
    notes = '';
    groupNew = '';
    errorText = null;
    if (groups.length === 0) {
      createNewGroup = true;
      groupExisting = '';
    } else {
      createNewGroup = false;
      groupExisting = groups[0] ?? '';
    }
  });

  async function submit(): Promise<void> {
    const slug = name.trim().toLowerCase();
    if (!slug) {
      errorText = 'Name is required.';
      return;
    }
    const group = (createNewGroup ? groupNew : groupExisting).trim();
    if (!group) {
      errorText = 'Group is required.';
      return;
    }
    errorText = null;
    busy = true;
    try {
      // No client-side slug check — `POST {API_PREFIX}/classes` 422s on a
      // bad name (`^[a-z0-9_]+$`) with the pattern in its detail; that
      // message is what renders below the field.
      await addClass({ name: slug, group, notes: notes.trim() || undefined });
      toastStore.success(`Created class "${slug}".`);
      await classesStore.clearAndRefetch();
      oncreated?.({ name: slug, group });
      onclose();
    } catch (e) {
      errorText = apiErrorText(e);
    } finally {
      busy = false;
    }
  }
</script>

{#if open}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Add class"
    use:focusOnMount
    use:trapFocus={{ onEscape: onclose }}
    tabindex="-1"
    onclick={(e) => {
      // Backdrop only: a click that bubbled up from the panel is not a
      // dismiss gesture.
      if (e.target === e.currentTarget) onclose();
    }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-3 text-base font-semibold">Add Class</h3>
      <form
        onsubmit={async (e) => {
          e.preventDefault();
          await submit();
        }}
      >
        <label class="mb-3 block text-sm">
          <span class="mb-1 block text-xs text-zinc-400">Class name (lowercase slug)</span
          >
          <input
            type="text"
            bind:value={name}
            required
            placeholder="e.g. delivery_van"
            class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
          <span class="mt-1 block text-[11px] text-zinc-500">
            Lowercase a–z, 0–9, _ only. No spaces — the server rejects anything else.
          </span>
        </label>

        <label class="mb-1 block text-sm">
          <span class="mb-1 block text-xs text-zinc-400">Group</span>
          {#if createNewGroup}
            <input
              type="text"
              bind:value={groupNew}
              placeholder="e.g. animals / tools / furniture"
              class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
            />
          {:else}
            <select
              bind:value={groupExisting}
              class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
            >
              {#each groups as g (g)}
                <option value={g}>{g}</option>
              {/each}
            </select>
          {/if}
        </label>
        <button
          type="button"
          class="mb-3 text-[11px] text-blue-400 hover:underline"
          onclick={() => (createNewGroup = !createNewGroup)}
          disabled={groups.length === 0}
        >
          {createNewGroup ? 'pick existing group' : 'create new group'}
        </button>

        <label class="mb-3 block text-sm">
          <span class="mb-1 block text-xs text-zinc-400">Notes (optional)</span>
          <textarea
            bind:value={notes}
            rows="2"
            class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          ></textarea>
        </label>

        {#if errorText}
          <p class="mb-3 text-xs text-red-300">{errorText}</p>
        {/if}

        <div class="flex items-center justify-end gap-2">
          <button type="button" class="btn" onclick={onclose} disabled={busy}>
            Cancel
          </button>
          <button
            type="submit"
            class="btn btn-primary"
            disabled={busy || name.trim() === ''}
          >
            {busy ? 'Creating…' : 'Create'}
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}
