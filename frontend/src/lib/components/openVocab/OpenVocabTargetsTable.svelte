<script lang="ts">
  /**
   * The targets of one open-vocabulary set: one row per target, a cell per
   * served `target` schema row (advanced ones behind a per-row expander),
   * each through `ProfileFieldEditor`. The class-name cell autocompletes
   * from the project's classes but accepts any text; an empty class name is
   * discovery mode. Every label, default, range and help text is served; the
   * issues under a cell are the served ones whose `field` path names it
   * (`targets[2].prompt`). Nothing here blocks enabling more targets than
   * the served limit: the validator answers that.
   */
  import ProfileFieldEditor from '$components/profiles/ProfileFieldEditor.svelte';
  import {
    issuePath,
    issuesForTargetField,
    openVocabFieldAsProfileField,
  } from '$lib/openVocab/openVocabFields';
  import type { ValidationReport } from '$lib/types_config';
  import type { OpenVocabFieldSchema, OpenVocabTargetBody } from '$lib/types_openVocab';
  import type { ProfileFieldValue } from '$lib/types_profiles';

  interface Props {
    targets: OpenVocabTargetBody[];
    /** The served `target`-scope schema rows, in served order. */
    rows: OpenVocabFieldSchema[];
    report: ValidationReport | null;
    readonly?: boolean;
    /** Non-deprecated class names for the class-name autocomplete. */
    classNames?: string[];
    maxEnabledTargets?: number | null;
    maxEnabledTargetsCeiling?: number | null;
    onadd: () => void;
    onremove: (index: number) => void;
    onmove: (index: number, dir: -1 | 1) => void;
    onchange: (index: number, field: string, value: unknown) => void;
  }

  let {
    targets,
    rows,
    report,
    readonly = false,
    classNames = [],
    maxEnabledTargets = null,
    maxEnabledTargetsCeiling = null,
    onadd,
    onremove,
    onmove,
    onchange,
  }: Props = $props();

  const basic = $derived(rows.filter((r) => !r.advanced));
  const advanced = $derived(rows.filter((r) => r.advanced));
  const enabledCount = $derived(targets.filter((t) => t.enabled !== false).length);
  let open = $state<Record<number, boolean>>({});

  const facts = $derived(
    [
      `${enabledCount} enabled of ${targets.length}`,
      maxEnabledTargets != null ? `up to ${maxEnabledTargets} enabled` : null,
      maxEnabledTargetsCeiling != null ? `ceiling ${maxEnabledTargetsCeiling}` : null,
    ]
      .filter((p) => p != null)
      .join(' · '),
  );

  /** A per-row field: the served row, with its served issue path as its
   *  name so each cell in the list has its own label target. */
  const cell = (row: OpenVocabFieldSchema, i: number) => ({
    ...openVocabFieldAsProfileField(row),
    field: issuePath('target', row.field, i),
  });
</script>

<section class="flex flex-col gap-3" data-testid="open-vocab-targets">
  <div class="flex flex-wrap items-baseline gap-3">
    <h2 class="text-sm font-semibold text-zinc-200">Targets</h2>
    <span class="text-xs text-zinc-400" data-testid="targets-facts">{facts}</span>
    <span class="grow"></span>
    {#if !readonly}
      <button type="button" class="btn btn-sm" data-testid="target-add" onclick={onadd}
        >Add target</button
      >
    {/if}
  </div>

  {#if targets.length === 0}
    <p class="text-sm text-zinc-400" data-testid="targets-empty">
      No targets yet. Add one to say what to look for.
    </p>
  {/if}

  <datalist id="open-vocab-class-names">
    {#each classNames as n (n)}<option value={n}></option>{/each}
  </datalist>

  <ol class="flex flex-col gap-3">
    {#each targets as t, i (i)}
      <li
        class="rounded border border-zinc-800 p-3"
        data-testid="target-row"
        data-index={i}
      >
        <div class="mb-2 flex flex-wrap items-center gap-2">
          <span class="font-mono text-xs text-zinc-500">#{i}</span>
          <span class="truncate text-sm text-zinc-200"
            >{t.prompt || 'untitled target'}</span
          >
          <span class="grow"></span>
          {#if !readonly}
            <button
              type="button"
              class="btn btn-sm"
              aria-label="Move target {i} up"
              disabled={i === 0}
              data-testid="target-up"
              onclick={() => onmove(i, -1)}>Up</button
            >
            <button
              type="button"
              class="btn btn-sm"
              aria-label="Move target {i} down"
              disabled={i === targets.length - 1}
              data-testid="target-down"
              onclick={() => onmove(i, 1)}>Down</button
            >
            <button
              type="button"
              class="btn btn-sm"
              data-testid="target-remove"
              onclick={() => onremove(i)}>Remove</button
            >
          {/if}
        </div>

        <div class="grid gap-3 md:grid-cols-2">
          {#each basic as r (r.field)}
            {#if r.field === 'class_name'}
              <div class="flex flex-col gap-1.5" data-testid="target-class-cell">
                <label
                  class="text-sm font-medium text-zinc-100"
                  for="target-{i}-class-name">{r.label}</label
                >
                {#if r.help}<p class="text-xs text-zinc-400">{r.help}</p>{/if}
                <input
                  id="target-{i}-class-name"
                  class="input input-sm max-w-md font-mono"
                  list="open-vocab-class-names"
                  value={t.class_name ?? ''}
                  {readonly}
                  oninput={(e) =>
                    onchange(
                      i,
                      'class_name',
                      (e.currentTarget as HTMLInputElement).value,
                    )}
                />
                {#if (t.class_name ?? '') === ''}
                  <p class="text-[11px] text-zinc-500" data-testid="discovery-hint">
                    Empty: discovery mode, hits are stored as unlabeled proposals named by
                    the prompt.
                  </p>
                {/if}
                {#each issuesForTargetField(report, i, r.field) as iss (iss.id)}
                  <p
                    class="text-xs {iss.severity === 'error'
                      ? 'text-red-300'
                      : 'text-amber-300'}"
                    data-testid="cell-issue"
                  >
                    {iss.message}
                  </p>
                {/each}
              </div>
            {:else}
              <ProfileFieldEditor
                field={cell(r, i)}
                value={t[r.field as keyof OpenVocabTargetBody] as
                  ProfileFieldValue | undefined}
                issues={issuesForTargetField(report, i, r.field)}
                choices={null}
                applies={null}
                {readonly}
                onchange={(v) => onchange(i, r.field, v)}
              />
            {/if}
          {/each}
        </div>

        {#if advanced.length > 0}
          <button
            type="button"
            class="mt-2 text-xs text-blue-300 hover:underline"
            data-testid="target-advanced-toggle"
            onclick={() => (open[i] = !open[i])}
            >{open[i] ? 'Hide' : 'Show'} advanced ({advanced.length})</button
          >
          {#if open[i]}
            <div class="mt-2 grid gap-3 md:grid-cols-2" data-testid="target-advanced">
              {#each advanced as r (r.field)}
                <ProfileFieldEditor
                  field={cell(r, i)}
                  value={t[r.field as keyof OpenVocabTargetBody] as
                    ProfileFieldValue | undefined}
                  issues={issuesForTargetField(report, i, r.field)}
                  choices={null}
                  applies={null}
                  {readonly}
                  onchange={(v) => onchange(i, r.field, v)}
                />
              {/each}
            </div>
          {/if}
        {/if}
      </li>
    {/each}
  </ol>
</section>
