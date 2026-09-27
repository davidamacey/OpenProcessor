<script lang="ts">
  /**
   * One region-profile field, rendered from its served schema row (§7.3):
   * label, help, the served default and range, the control its served
   * `type` calls for, the served choices its `choices_from` names (with the
   * served `empty_choice`), and the served issues on it. A row whose
   * `applies_when` is off in the saved revision is dimmed, never disabled
   * (W4-Q2). No client rule checks the value; HTML `min`/`max` only hint.
   */
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import { numberFromInput, selectOptions, valueText } from '$lib/profiles/profileFields';
  import type { ValidationIssue } from '$lib/types_config';
  import type {
    Choice,
    ProfileFieldValue,
    ProfileSchemaField,
  } from '$lib/types_profiles';

  interface Props {
    field: ProfileSchemaField;
    value: ProfileFieldValue | undefined;
    issues: ValidationIssue[];
    /** The served list `choices_from` names; null when the row has none. */
    choices: Choice[] | null;
    /** `applies_when` in the saved revision (null = no condition / unknown). */
    applies: boolean | null;
    readonly?: boolean;
    onchange: (value: ProfileFieldValue) => void;
  }

  let {
    field,
    value,
    issues,
    choices,
    applies,
    readonly = false,
    onchange,
  }: Props = $props();

  const id = $derived(`profile-field-${field.field}`);
  const hasError = $derived(issues.some((i) => i.severity === 'error'));
  const list = $derived(Array.isArray(value) ? (value as unknown[]) : []);
  const fixedArity = $derived(
    field.type === 'float_pair' ? 2 : field.type === 'rgb' ? 3 : 0,
  );
  const known = [
    'string',
    'int',
    'float',
    'bool',
    'enum',
    'string_list',
    'int_list',
    'float_pair',
    'rgb',
  ];

  let adding = $state('');

  function addItem(): void {
    const raw = adding.trim();
    if (raw === '') return;
    const item = field.type === 'int_list' ? numberFromInput(raw) : raw;
    if (item == null) return;
    onchange([...list, item] as ProfileFieldValue);
    adding = '';
  }

  function removeItem(idx: number): void {
    onchange(list.filter((_, i) => i !== idx) as ProfileFieldValue);
  }

  function setTupleItem(idx: number, raw: string): void {
    const next = Array.from({ length: fixedArity }, (_, i) =>
      i === idx ? numberFromInput(raw) : ((list[i] as number | null | undefined) ?? null),
    );
    onchange(next as ProfileFieldValue);
  }

  function setJson(raw: string): void {
    try {
      onchange(JSON.parse(raw) as ProfileFieldValue);
    } catch {
      onchange(raw);
    }
  }

  const rangeText = $derived(
    field.min != null && field.max != null
      ? `${field.min} to ${field.max}`
      : field.min != null
        ? `at least ${field.min}`
        : field.max != null
          ? field.type === 'string'
            ? `up to ${field.max} characters`
            : `at most ${field.max}`
          : null,
  );
</script>

<div
  class="flex flex-col gap-1.5 border-t border-zinc-800 pt-3 first:border-0 first:pt-0"
  class:opacity-60={applies === false}
  data-testid="profile-field"
  data-field={field.field}
  data-type={field.type}
>
  <div class="flex flex-wrap items-baseline gap-2">
    <label class="text-sm font-medium text-zinc-100" for={id}>{field.label}</label>
    <code class="font-mono text-[11px] text-zinc-500">{field.field}</code>
    {#if field.advanced}
      <span class="rounded border border-zinc-700 px-1 text-[10px] text-zinc-400"
        >advanced</span
      >
    {/if}
  </div>
  {#if field.help}<p class="text-xs text-zinc-400">{field.help}</p>{/if}
  <p class="text-[11px] text-zinc-500" data-testid="field-facts">
    default: <span class="font-mono">{valueText(field.default)}</span>
    {#if rangeText}· {rangeText}{/if}
  </p>
  {#if applies === false}
    <p class="text-[11px] text-amber-300/80" data-testid="field-not-applied">
      Not used by the saved revision ({field.applies_when} is off).
    </p>
  {/if}

  {#if field.type === 'bool'}
    <label class="flex items-center gap-2 text-sm">
      <input
        {id}
        type="checkbox"
        checked={value === true}
        disabled={readonly}
        onchange={(e) => onchange((e.currentTarget as HTMLInputElement).checked)}
      />
      <span class="text-zinc-300">{value === true ? 'on' : 'off'}</span>
    </label>
  {:else if field.type === 'enum' && field.enum}
    <select
      {id}
      class="select select-sm max-w-md {hasError ? 'border-red-500/60' : ''}"
      value={typeof value === 'string' ? value : ''}
      disabled={readonly}
      onchange={(e) => onchange((e.currentTarget as HTMLSelectElement).value)}
    >
      {#each selectOptions(field, field.enum, value) as o (o.id)}
        <option value={o.id}>{o.label}</option>
      {/each}
    </select>
  {:else if field.type === 'string' && choices}
    <select
      {id}
      class="select select-sm max-w-md font-mono {hasError ? 'border-red-500/60' : ''}"
      value={typeof value === 'string' ? value : ''}
      disabled={readonly}
      data-testid="field-choice"
      onchange={(e) => onchange((e.currentTarget as HTMLSelectElement).value)}
    >
      {#each selectOptions(field, choices, value) as o (o.id)}
        <option value={o.id}>{o.label}</option>
      {/each}
    </select>
  {:else if field.type === 'string'}
    <input
      {id}
      class="input input-sm max-w-md font-mono {hasError ? 'border-red-500/60' : ''}"
      value={typeof value === 'string' ? value : ''}
      {readonly}
      oninput={(e) => onchange((e.currentTarget as HTMLInputElement).value)}
    />
  {:else if field.type === 'int' || field.type === 'float'}
    <input
      {id}
      type="number"
      class="input input-sm w-40 font-mono {hasError ? 'border-red-500/60' : ''}"
      step={field.type === 'int' ? 1 : 'any'}
      min={field.min ?? undefined}
      max={field.max ?? undefined}
      value={typeof value === 'number' ? value : ''}
      {readonly}
      oninput={(e) =>
        onchange(numberFromInput((e.currentTarget as HTMLInputElement).value))}
    />
  {:else if field.type === 'string_list' || field.type === 'int_list'}
    <div class="flex flex-wrap items-center gap-1" data-testid="field-list">
      {#each list as item, idx (idx)}
        <span
          class="flex items-center gap-1 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 font-mono text-xs"
          data-testid="list-item"
          >{choices?.find((c) => c.id === item)?.label ?? String(item)}
          {#if !readonly}
            <button
              type="button"
              class="text-zinc-500 hover:text-zinc-200"
              aria-label="Remove {String(item)}"
              onclick={() => removeItem(idx)}>×</button
            >
          {/if}
        </span>
      {/each}
      {#if list.length === 0}<span class="text-xs text-zinc-500">none</span>{/if}
    </div>
    {#if !readonly}
      <div class="flex flex-wrap items-center gap-2">
        <input
          {id}
          class="input input-sm w-56 font-mono"
          type={field.type === 'int_list' ? 'number' : 'text'}
          list={choices ? `${id}-choices` : undefined}
          placeholder={choices ? 'Pick or type a name' : 'Add a value'}
          bind:value={adding}
          data-testid="list-add-input"
          onkeydown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              addItem();
            }
          }}
        />
        {#if choices}
          <datalist id="{id}-choices">
            {#each choices.filter((c) => !list.includes(c.id)) as c (c.id)}
              <option value={c.id}>{c.label}</option>
            {/each}
          </datalist>
        {/if}
        <button
          type="button"
          class="btn btn-sm"
          data-testid="list-add"
          disabled={adding.trim() === ''}
          onclick={addItem}>Add</button
        >
      </div>
    {/if}
  {:else if fixedArity > 0}
    <div class="flex flex-wrap items-center gap-2" data-testid="field-tuple">
      {#each Array.from({ length: fixedArity }, (_, i) => i) as i (i)}
        <input
          id={i === 0 ? id : `${id}-${i}`}
          type="number"
          class="input input-sm w-24 font-mono"
          step={field.type === 'rgb' ? 1 : 'any'}
          min={field.min ?? undefined}
          max={field.max ?? undefined}
          aria-label="{field.label} {i + 1}"
          value={typeof list[i] === 'number' ? (list[i] as number) : ''}
          {readonly}
          oninput={(e) => setTupleItem(i, (e.currentTarget as HTMLInputElement).value)}
        />
      {/each}
    </div>
  {:else if !known.includes(field.type)}
    <textarea
      {id}
      class="input min-h-16 w-full resize-y font-mono text-xs"
      value={JSON.stringify(value ?? null)}
      {readonly}
      data-testid="field-json"
      oninput={(e) => setJson((e.currentTarget as HTMLTextAreaElement).value)}></textarea>
  {:else}
    <input
      {id}
      class="input input-sm max-w-md font-mono"
      value={valueText(value)}
      readonly
    />
  {/if}

  <ConfigIssueList {issues} showField={issues.some((i) => i.field !== field.field)} />
</div>
