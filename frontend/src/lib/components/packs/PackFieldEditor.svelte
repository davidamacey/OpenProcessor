<script lang="ts">
  /**
   * One prompt-pack field, rendered from its served schema row (§3.4):
   * label, help, placeholder and reply-key chips, the pipeline steps that
   * use it, and the served issues on it. `kind: "text"` is a textarea,
   * `kind: "map"` a key/value list and `kind: "list"` a list of strings
   * (always sent as a clean `string[]`: trimmed, no blanks or duplicates,
   * max 500 entries of 200 chars). Any other kind is shown read-only so a
   * future kind never crashes the editor or has its value rewritten.
   */
  import type { ValidationIssue } from '$lib/types_config';
  import type { PackFieldValue, PackSchemaField } from '$lib/types_packs';
  import { untrack } from 'svelte';
  import {
    LIST_MAX_ENTRIES,
    LIST_MAX_ENTRY_CHARS,
    duplicateRows,
    listValue,
    normalizeList,
  } from '$lib/packs/packFieldValue';
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';

  interface Props {
    field: PackSchemaField;
    value: PackFieldValue | undefined;
    issues: ValidationIssue[];
    readonly?: boolean;
    onchange: (value: PackFieldValue) => void;
  }

  let { field, value, issues, readonly = false, onchange }: Props = $props();

  const isList = $derived(field.kind === 'list');
  const isMap = $derived(field.kind === 'map');
  const isText = $derived(field.kind === 'text');
  const unknownText = $derived.by(() => {
    try {
      return JSON.stringify(value ?? null, null, 2);
    } catch {
      return String(value);
    }
  });
  const text = $derived(typeof value === 'string' ? value : '');
  // Draft rows keep a just-added blank row on screen; only the cleaned list
  // is emitted.
  let rows = $state<string[]>([]);
  $effect(() => {
    const v = listValue(value);
    untrack(() => {
      if (JSON.stringify(normalizeList(rows)) !== JSON.stringify(v)) rows = [...v];
    });
  });
  const dups = $derived(duplicateRows(rows));
  const entries = $derived(
    value && typeof value === 'object' && !Array.isArray(value)
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
  function emitRows(next: string[]): void {
    rows = next;
    onchange(normalizeList(next));
  }
  function setPattern(idx: number, v: string): void {
    emitRows(rows.map((p, i) => (i === idx ? v : p)));
  }
  function removePattern(idx: number): void {
    emitRows(rows.filter((_, i) => i !== idx));
  }
  function addPattern(): void {
    rows = [...rows, ''];
  }
</script>

<div
  class="flex flex-col gap-1.5 border-t border-line pt-3 first:border-0 first:pt-0"
  data-testid="pack-field"
  data-field={field.field}
>
  <div class="flex flex-wrap items-baseline gap-2">
    <label class="text-sm font-medium text-fg-1" for="pack-field-{field.field}"
      >{field.label}</label
    >
    <code class="font-mono text-caption text-fg-3">{field.field}</code>
  </div>
  {#if field.help}<p class="text-xs text-fg-2">{field.help}</p>{/if}

  {#if field.allowed_placeholders.length > 0 || field.expected_reply_keys.length > 0 || field.optional_reply_keys.length > 0}
    <div class="flex flex-wrap items-center gap-1 text-caption">
      {#each field.required_placeholders as p (p)}
        <span
          class="rounded border border-info/50 bg-info/10 px-1.5 py-0.5 font-mono text-info-text"
          data-testid="placeholder-chip"
          data-required="true"
          title="required placeholder">{`{${p}}`} required</span
        >
      {/each}
      {#each optionalPlaceholders as p (p)}
        <span
          class="rounded border border-info/30 px-1.5 py-0.5 font-mono text-info-text"
          data-testid="placeholder-chip"
          data-required="false"
          title="allowed placeholder">{`{${p}}`}</span
        >
      {/each}
      {#if field.expected_reply_keys.length > 0 || field.optional_reply_keys.length > 0}
        <span class="ml-1 text-fg-3">reply keys:</span>
      {/if}
      {#each field.expected_reply_keys as k (k)}
        <span
          class="rounded border border-line-strong px-1.5 py-0.5 font-mono text-fg-1"
          data-testid="reply-key-chip">{k}</span
        >
      {/each}
      {#each field.optional_reply_keys as k (k)}
        <span
          class="rounded border border-dashed border-line-strong px-1.5 py-0.5 font-mono text-fg-2"
          data-testid="reply-key-chip"
          data-optional="true">{k} optional</span
        >
      {/each}
    </div>
  {/if}

  {#if isList}
    <div class="space-y-1" data-testid="pack-list">
      {#each rows as p, idx (idx)}
        <div class="flex items-center gap-1">
          <input
            class="input input-sm min-w-0 flex-1 font-mono"
            aria-label="{field.label}: pattern {idx + 1}"
            value={p}
            {readonly}
            maxlength={LIST_MAX_ENTRY_CHARS}
            spellcheck="false"
            oninput={(e) => setPattern(idx, (e.currentTarget as HTMLInputElement).value)}
          />
          {#if !readonly}
            <button
              type="button"
              class="btn btn-sm"
              aria-label="Remove {p || 'pattern'}"
              onclick={() => removePattern(idx)}>Remove</button
            >
          {/if}
          {#if dups.has(idx)}
            <span class="text-caption text-warn-text">Duplicate, ignored</span>
          {/if}
        </div>
      {:else}
        <p class="text-xs text-fg-3">No patterns.</p>
      {/each}
      {#if !readonly}
        <button
          type="button"
          class="btn btn-sm"
          disabled={rows.length >= LIST_MAX_ENTRIES || rows.some((p) => p.trim() === '')}
          onclick={addPattern}>Add pattern</button
        >
      {/if}
      <p class="text-caption text-fg-3">
        Case-insensitive globs (<code>blurry_*</code>, <code>*_scene</code>). A proposed
        new class matching one is dropped and never reaches the new-class queue.
      </p>
    </div>
  {:else if isMap}
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
          <span class="text-fg-3">→</span>
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
        <p class="text-xs text-fg-3">No entries.</p>
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
  {:else if isText}
    <textarea
      id="pack-field-{field.field}"
      class="input min-h-24 w-full resize-y font-mono text-xs {hasError
        ? 'border-danger/60'
        : ''}"
      rows={Math.min(14, Math.max(3, text.split('\n').length + 1))}
      value={text}
      {readonly}
      spellcheck="false"
      oninput={(e) => onchange((e.currentTarget as HTMLTextAreaElement).value)}
    ></textarea>
  {:else}
    <div data-testid="pack-unknown" class="space-y-1">
      <p class="text-xs text-warn-text">
        Unsupported field kind “{field.kind}”: shown read-only and saved unchanged.
      </p>
      <pre
        class="max-h-48 overflow-auto rounded border border-line p-2 font-mono text-caption text-fg-2">{unknownText}</pre>
    </div>
  {/if}

  {#if field.used_by.length > 0}
    <p class="text-caption text-fg-3">used by: {field.used_by.join(', ')}</p>
  {/if}
  <ConfigIssueList {issues} showField={isMap || isList} />
</div>
