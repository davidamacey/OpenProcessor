<script lang="ts">
  import { classesStore } from '$stores/classes.svelte';
  import type { OpClass } from '$lib/types';

  interface Props {
    selectedId: number | null;
    onselect: (cls: OpClass | null) => void;
  }

  let { selectedId = null, onselect }: Props = $props();

  let query = $state<string>('');
  let modalOpen = $state<boolean>(false);

  const filtered = $derived.by(() => {
    const q = query.trim().toLowerCase();
    const list = q
      ? classesStore.classes.filter(
          (c) =>
            c.name.toLowerCase().includes(q) ||
            (c.group ?? '').toLowerCase().includes(q),
        )
      : classesStore.classes;
    return [...list].sort((a, b) => (b.validated_count ?? 0) - (a.validated_count ?? 0));
  });

  const badgeColor = (n: number): string => {
    if (n >= 500) return 'bg-green-500/20 text-green-300 border-green-500/30';
    if (n >= 100) return 'bg-orange-500/20 text-orange-200 border-orange-500/30';
    return 'bg-red-500/20 text-red-200 border-red-500/30';
  };
</script>

<aside class="flex h-full w-64 flex-col border-r border-zinc-800 bg-zinc-950">
  <div class="border-b border-zinc-800 p-3">
    <h2 class="mb-2 text-xs font-semibold tracking-wide text-zinc-400 uppercase">Classes</h2>
    <input
      type="search"
      bind:value={query}
      placeholder="Filter..."
      class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-blue-500 focus:outline-none"
    />
  </div>

  <div class="flex-1 overflow-y-auto">
    {#if classesStore.loading && classesStore.classes.length === 0}
      <div class="p-4 text-sm text-zinc-500">Loading classes...</div>
    {:else if classesStore.error}
      <div class="p-4 text-sm text-red-300">API unavailable</div>
    {:else if filtered.length === 0}
      <div class="p-4 text-sm text-zinc-500">No classes</div>
    {:else}
      <ul class="py-1">
        <li>
          <button
            type="button"
            class="flex w-full items-center justify-between px-3 py-1.5 text-left text-sm hover:bg-zinc-900 {selectedId ===
            null
              ? 'bg-zinc-800 text-white'
              : 'text-zinc-300'}"
            onclick={() => onselect(null)}
          >
            <span>All classes</span>
          </button>
        </li>
        {#each filtered as cls (cls.id)}
          <li>
            <button
              type="button"
              class="flex w-full items-center justify-between gap-2 px-3 py-1.5 text-left text-sm hover:bg-zinc-900 {selectedId ===
              cls.id
                ? 'bg-zinc-800 text-white'
                : 'text-zinc-300'}"
              onclick={() => onselect(cls)}
              title={cls.group ? `${cls.group} / ${cls.name}` : cls.name}
            >
              <span class="truncate">{cls.name}</span>
              <span
                class="rounded-md border px-1.5 py-0.5 font-mono text-xs {badgeColor(
                  cls.validated_count ?? 0,
                )}"
              >
                {cls.validated_count ?? 0}
              </span>
            </button>
          </li>
        {/each}
      </ul>
    {/if}
  </div>

  <div class="border-t border-zinc-800 p-2">
    <button type="button" class="btn w-full justify-center" onclick={() => (modalOpen = true)}>
      + Add Class
    </button>
  </div>
</aside>

{#if modalOpen}
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Add class"
  >
    <div class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl">
      <h3 class="mb-2 text-base font-semibold">Add Class</h3>
      <p class="mb-4 text-sm text-zinc-400">
        Class management arrives in <strong>v1.1</strong>. For now, add classes via
        <code class="font-mono text-xs text-zinc-200">POST /curation/classes</code>.
      </p>
      <div class="flex justify-end">
        <button type="button" class="btn" onclick={() => (modalOpen = false)}>Close</button>
      </div>
    </div>
  </div>
{/if}
