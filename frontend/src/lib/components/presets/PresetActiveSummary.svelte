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

  const label = (axis: string | null): string => active.axisLabel(axis) || 'Deployment';
</script>

<section class="surface flex flex-col gap-2 p-4 text-body" aria-label="Active">
  <h2 class="text-title font-semibold">Active</h2>
  {#if active.loadError}
    <p class="text-danger-text" data-testid="preset-active-error">{active.loadError}</p>
  {:else if active.entries.length === 0}
    <p class="text-fg-3">Loading…</p>
  {:else}
    <ul class="flex flex-col gap-1" data-testid="preset-active">
      {#each active.entries as e (e.axis ?? 'deployment')}
        <li class="flex flex-wrap items-baseline gap-2" data-testid="preset-active-row">
          <span class="w-36 shrink-0 text-meta text-fg-2">{label(e.axis)}</span>
          {#if e.state === 'active' && e.name}
            <code>{e.name}</code>
            <span class="text-meta text-fg-2 tabular-nums">r{e.revision}</span>
            {#if e.activated_at}
              <span class="text-meta text-fg-3"
                >since {formatTimestamp(e.activated_at)}</span
              >
            {/if}
          {:else}
            <span class="text-fg-2"
              >{e.state === 'off' ? 'turned off' : 'nothing active'}</span
            >
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</section>
