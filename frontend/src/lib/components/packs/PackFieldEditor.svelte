<script lang="ts">
  /**
   * One prompt-pack field, rendered from its served schema row (§3.4):
   * label, help, placeholder and reply-key chips, the pipeline steps that
   * use it, and the served issues on it. `kind: "text"` is a textarea,
   * `kind: "map"` a key/value list. No client rule checks the value.
   */
  import type { ValidationIssue } from '$lib/types_config';
  import type { PackFieldValue, PackSchemaField } from '$lib/types_packs';
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';

  interface Props {
    field: PackSchemaField;
    value: PackFieldValue | undefined;
    issues: ValidationIssue[];
    readonly?: boolean;
    onchange: (value: PackFieldValue) => void;
  }

  let { field, value, issues, readonly = false, onchange }: Props = $props();

  const isMap = $derived(field.kind === 'map');
  const text = $derived(typeof value === 'string' ? value : '');
  const entries = $derived(
    value && typeof value === 'object'
      ? Object.entries(value)
      : ([] as [string, string][]),
  );
  const optionalPlaceholders = $derived(
    field.allowed_placeholders.filter((p) => !field.required_placeholders.includes(p)),
  );
  const hasError = $derived(issues.some((i) => i.severity === 'error'));

  function setEntries(next: [string, string][]): void {
    onchange(Object.fromEntries(next));
  }
  function setKey(idx: number, key: string): void {
    setEntries(entries.map((e, i) => (i === idx ? [key, e[1]] : e)));
  }
  function setValue(idx: number, v: string): void {
    setEntries(entries.map((e, i) => (i === idx ? [e[0], v] : e)));
  }
  function removeEntry(idx: number): void {
    setEntries(entries.filter((_, i) => i !== idx));
  }
  function addEntry(): void {
    setEntries([...entries, ['', '']]);
  }
</script>

<div
  class="flex flex-col gap-1.5 border-t border-zinc-800 pt-3 first:border-0 first:pt-0"
  data-testid="pack-field"
  data-field={field.field}
>
  <div class="flex flex-wrap items-baseline gap-2">
    <label class="text-sm font-medium text-zinc-100" for="pack-field-{field.field}"
      >{field.label}</label
    >
    <code class="font-mono text-[11px] text-zinc-500">{field.field}</code>
  </div>
  {#if field.help}<p class="text-xs text-zinc-400">{field.help}</p>{/if}

  {#if field.allowed_placeholders.length > 0 || field.expected_reply_keys.length > 0 || field.optional_reply_keys.length > 0}
    <div class="flex flex-wrap items-center gap-1 text-[11px]">
      {#each field.required_placeholders as p (p)}
        <span
          class="rounded border border-sky-500/50 bg-sky-500/10 px-1.5 py-0.5 font-mono text-sky-200"
          data-testid="placeholder-chip"
          data-required="true"
          title="required placeholder">{`{${p}}`} required</span
        >
      {/each}
      {#each optionalPlaceholders as p (p)}
        <span
          class="rounded border border-sky-500/30 px-1.5 py-0.5 font-mono text-sky-300"
          data-testid="placeholder-chip"
          data-required="false"
          title="allowed placeholder">{`{${p}}`}</span
        >
      {/each}
      {#if field.expected_reply_keys.length > 0 || field.optional_reply_keys.length > 0}
        <span class="ml-1 text-zinc-500">reply keys:</span>
      {/if}
      {#each field.expected_reply_keys as k (k)}
        <span
          class="rounded border border-zinc-600 px-1.5 py-0.5 font-mono text-zinc-200"
          data-testid="reply-key-chip">{k}</span
        >
      {/each}
      {#each field.optional_reply_keys as k (k)}
        <span
          class="rounded border border-dashed border-zinc-700 px-1.5 py-0.5 font-mono text-zinc-400"
          data-testid="reply-key-chip"
          data-optional="true">{k} optional</span
        >
      {/each}
    </div>
  {/if}

  {#if isMap}
    <div class="space-y-1" data-testid="pack-map">
      {#each entries as [k, v], idx (idx)}
        <div class="flex items-center gap-1">
          <input
            class="input input-sm w-40 font-mono"
            aria-label="{field.label}: key {idx + 1}"
            value={k}
            {readonly}
            oninput={(e) => setKey(idx, (e.currentTarget as HTMLInputElement).value)}
          />
          <span class="text-zinc-500">→</span>
          <input
            class="input input-sm min-w-0 flex-1 font-mono"
            aria-label="{field.label}: value {idx + 1}"
            value={v}
            {readonly}
            oninput={(e) => setValue(idx, (e.currentTarget as HTMLInputElement).value)}
          />
          {#if !readonly}
            <button
              type="button"
              class="btn btn-sm"
              aria-label="Remove {k || 'entry'}"
              onclick={() => removeEntry(idx)}>Remove</button
            >
          {/if}
        </div>
      {:else}
        <p class="text-xs text-zinc-500">No entries.</p>
      {/each}
      {#if !readonly}
        <button
          type="button"
          class="btn btn-sm"
          disabled={entries.some(([k]) => k === '')}
          onclick={addEntry}>Add entry</button
        >
      {/if}
    </div>
  {:else}
    <textarea
      id="pack-field-{field.field}"
      class="input min-h-24 w-full resize-y font-mono text-xs {hasError
        ? 'border-red-500/60'
        : ''}"
      rows={Math.min(14, Math.max(3, text.split('\n').length + 1))}
      value={text}
      {readonly}
      spellcheck="false"
      oninput={(e) => onchange((e.currentTarget as HTMLTextAreaElement).value)}
    ></textarea>
  {/if}

  {#if field.used_by.length > 0}
    <p class="text-[11px] text-zinc-500">used by: {field.used_by.join(', ')}</p>
  {/if}
  <ConfigIssueList {issues} showField={isMap} />
</div>
