<script lang="ts">
  /**
   * What is active for a preset kind, as served: one line per axis (a cluster profile has two),
   * or the one deployment-wide serving profile. Reads only; the write is the activate dialog.
   */
  import { formatTimestamp } from '$lib/format/date';
  import type { PresetActive } from '$lib/presets/presetActive.svelte';

  interface Props {
    active: PresetActive;
  }

  let { active }: Props = $props();

  const label = (axis: string | null): string =>
    active.kind.axes.find((a) => a.id === axis)?.label ?? 'Deployment';
</script>

<section class="surface flex flex-col gap-2 p-4 text-sm" aria-label="Active">
  <h2 class="text-base font-semibold">Active</h2>
  {#if active.loadError}
    <p class="text-red-300" data-testid="preset-active-error">{active.loadError}</p>
  {:else if active.entries.length === 0}
    <p class="text-zinc-500">Loading…</p>
  {:else}
    <ul class="flex flex-col gap-1" data-testid="preset-active">
      {#each active.entries as e (e.axis ?? 'deployment')}
        <li class="flex flex-wrap items-baseline gap-2" data-testid="preset-active-row">
          <span class="w-36 shrink-0 text-xs text-zinc-400">{label(e.axis)}</span>
          {#if e.state === 'active' && e.name}
            <span class="font-mono">{e.name}</span>
            <span class="font-mono text-xs text-zinc-400">r{e.revision}</span>
            {#if e.activated_at}
              <span class="text-xs text-zinc-500"
                >since {formatTimestamp(e.activated_at)}</span
              >
            {/if}
          {:else}
            <span class="text-zinc-400"
              >{e.state === 'off' ? 'turned off' : 'nothing active'}</span
            >
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</section>
