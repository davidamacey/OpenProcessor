<!--
  A multi-select of class names: the detector's served labels as checkboxes
  plus free text for a name the detector does not list (sent by name; the
  server reports a name it does not know as `unknown_names`). Names are
  never validated here.
-->
<script lang="ts">
  let {
    value,
    options,
    label,
    onchange,
  }: {
    value: string[];
    options: string[];
    label: string;
    onchange: (next: string[]) => void;
  } = $props();

  let extra = $state('');

  const extras = $derived(value.filter((v) => !options.includes(v)));

  function toggle(name: string, on: boolean): void {
    onchange(on ? [...value, name] : value.filter((v) => v !== name));
  }

  function addExtra(): void {
    const names = extra
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s !== '' && !value.includes(s));
    if (names.length > 0) onchange([...value, ...names]);
    extra = '';
  }
</script>

<div class="space-y-1" data-testid="class-name-picker">
  <p class="text-xs text-zinc-400">{label}</p>
  {#if options.length > 0}
    <div class="flex max-h-32 flex-wrap gap-x-3 gap-y-1 overflow-y-auto text-xs">
      {#each options as name (name)}
        <label class="flex items-center gap-1">
          <input
            type="checkbox"
            checked={value.includes(name)}
            onchange={(e) => toggle(name, e.currentTarget.checked)}
          />
          <span class="text-zinc-200">{name}</span>
        </label>
      {/each}
    </div>
  {/if}
  {#if extras.length > 0}
    <div class="flex flex-wrap gap-1">
      {#each extras as name (name)}
        <button
          type="button"
          class="chip"
          title="Remove"
          onclick={() => toggle(name, false)}>{name} x</button
        >
      {/each}
    </div>
  {/if}
  <div class="flex gap-1">
    <input
      class="input w-48 text-xs"
      placeholder="Another class name"
      bind:value={extra}
      onkeydown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          addExtra();
        }
      }}
    />
    <button type="button" class="btn btn-sm" onclick={addExtra}>Add</button>
  </div>
</div>
