<script lang="ts">
  // A repeatable value picker: a select adds a value as a chip, each chip's
  // button removes it. The options are served values; nothing is invented.
  let {
    label,
    options,
    selected,
    onchange,
  }: {
    label: string;
    options: readonly string[];
    selected: readonly string[];
    onchange: (next: string[]) => void;
  } = $props();

  const available = $derived(options.filter((o) => !selected.includes(o)));
</script>

<div class="flex flex-wrap items-center gap-1.5" data-testid="chip-picker-{label}">
  {#if available.length > 0}
    <select
      aria-label="Add {label}"
      class="select-sm"
      value=""
      onchange={(e) => {
        const v = e.currentTarget.value;
        e.currentTarget.value = '';
        if (v) onchange([...selected, v]);
      }}
    >
      <option value="">{label}</option>
      {#each available as o (o)}
        <option value={o}>{o}</option>
      {/each}
    </select>
  {/if}
  {#each selected as v (v)}
    <button
      type="button"
      class="btn-sm border border-blue-500/50 bg-blue-500/15 text-blue-100 hover:bg-blue-500/25"
      aria-label="Remove {label} {v}"
      onclick={() => onchange(selected.filter((x) => x !== v))}
    >
      {v}
      <span aria-hidden="true" class="ml-1 text-blue-300">×</span>
    </button>
  {/each}
</div>
