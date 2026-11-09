<script lang="ts">
  /**
   * Which embedding space a search ranks in. The list is the project's ready
   * spaces for one target; with a single ready space there is nothing to
   * choose and nothing renders. `fuse` is every other ready space, merged by
   * reciprocal rank fusion server-side.
   */
  import { onMount } from 'svelte';
  import { getEmbeddingSpaces } from '$lib/api_embeddingSpaces';
  import type { EmbeddingSpaceView, EmbeddingTarget } from '$lib/types_embeddingSpaces';

  interface Props {
    target?: EmbeddingTarget;
    /** `space` is null for the target's primary space. */
    onChange: (choice: { space: string | null; fuse: string[] }) => void;
  }

  let { target = 'item', onChange }: Props = $props();

  let space = $state<string | null>(null);
  let spaces = $state<EmbeddingSpaceView[]>([]);
  let fuseOthers = $state(false);

  const ready = $derived(spaces.filter((s) => s.status === 'ready'));
  const primary = $derived(ready.find((s) => s.role === 'primary') ?? null);

  onMount(() => {
    const ctl = new AbortController();
    getEmbeddingSpaces(target, ctl.signal).then(
      (list) => {
        spaces = list;
      },
      () => {
        // Without the list the search simply runs in the primary space.
      },
    );
    return () => ctl.abort();
  });

  function label(s: EmbeddingSpaceView): string {
    return `${s.model_key} v${s.model_version}${s.role === 'primary' ? ' (primary)' : ''}`;
  }

  function apply(): void {
    const chosen = space ?? primary?.space_key ?? null;
    const fuse = fuseOthers
      ? ready.filter((s) => s.space_key !== chosen).map((s) => s.space_key)
      : [];
    onChange({ space, fuse });
  }
</script>

{#if ready.length > 1}
  <div class="inline-flex items-center gap-2" data-testid="search-space-picker">
    <label class="flex items-center gap-1 text-zinc-400">
      Space
      <select
        class="input-sm"
        data-testid="search-space-select"
        value={space ?? primary?.space_key ?? ''}
        onchange={(e) => {
          space = (e.currentTarget as HTMLSelectElement).value || null;
          apply();
        }}
      >
        {#each ready as s (s.space_key)}
          <option value={s.space_key}>{label(s)}</option>
        {/each}
      </select>
    </label>
    <label class="flex items-center gap-1 text-zinc-400">
      <input
        type="checkbox"
        data-testid="search-space-fuse"
        bind:checked={fuseOthers}
        onchange={apply}
      />
      Fuse all spaces
    </label>
  </div>
{/if}
