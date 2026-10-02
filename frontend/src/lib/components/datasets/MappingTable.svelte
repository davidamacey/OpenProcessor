<script lang="ts">
  /**
   * The class-mapping step (any_domain_plan.md §7.12 item 1, W10.5): one
   * row per served `DatasetPreview.classes[]` entry. Mapping is by NAME:
   * the dataset's own class id and an OP export's source registry id are
   * shown as plain labels and never compared with anything. What the
   * request currently maps a row to is the preview's served `resolved`;
   * a row with boxes and no `resolved` target is highlighted until
   * chosen. Merging two dataset classes is picking the same class on both
   * rows.
   */
  import type { ImportWizard } from '$lib/datasets/importWizardController.svelte';
  import type { DatasetClassRow, DatasetFormatsResponse } from '$lib/types_import';
  import type { RegistryClass } from '$lib/types';

  interface Props {
    wizard: ImportWizard;
    formats: DatasetFormatsResponse;
    /** Non-deprecated registry classes, for the `map` picker. */
    classes: RegistryClass[];
  }

  let { wizard, formats, classes }: Props = $props();

  const rows = $derived(wizard.preview?.classes ?? []);
  const unmapped = $derived(new Set(wizard.unmappedRows.map((r) => r.dataset_class)));
  const actionLabel = $derived(
    new Map(formats.mapping_actions.map((a) => [a.id, a.label])),
  );
  const matchLabel = $derived(new Map(formats.match_kinds.map((m) => [m.id, m.label])));
  // The served info issue says the old index path would have mislabeled
  // something; each row's own served `index_would_have_mapped_to` is the
  // hint (plan §8 question 17: which rows it covers isn't served).
  const indexMismatch = $derived(
    (wizard.preview?.issues ?? []).some((i) => i.code === 'class_index_name_mismatch'),
  );

  function resolvedText(row: DatasetClassRow): string | null {
    const r = row.resolved;
    if (!r) return null;
    if (r.kind === 'item') return r.class_name ?? String(r.class_id);
    const kind = actionLabel.get(r.kind) ?? r.kind;
    return r.class_name ? `${kind}: ${r.class_name}` : kind;
  }
</script>

<div class="overflow-x-auto">
  <table class="w-full text-left text-xs" data-testid="mapping-table">
    <thead class="text-zinc-500">
      <tr>
        <th class="px-2 py-1">Dataset class</th>
        <th class="px-2 py-1 text-right">Boxes</th>
        <th class="px-2 py-1 text-right">Images</th>
        <th class="px-2 py-1">Suggestion</th>
        <th class="px-2 py-1">Action</th>
        <th class="px-2 py-1">Maps to</th>
      </tr>
    </thead>
    <tbody>
      {#each rows as row (row.dataset_class)}
        {@const choice = wizard.choiceFor(row.dataset_class)}
        <tr
          class="border-t border-zinc-800 align-top {unmapped.has(row.dataset_class)
            ? 'bg-amber-950/30'
            : ''}"
          data-testid="mapping-row"
          data-dataset-class={row.dataset_class}
          data-unmapped={unmapped.has(row.dataset_class) ? 'true' : undefined}
        >
          <td class="px-2 py-1.5">
            <div class="font-medium text-zinc-100">{row.dataset_class}</div>
            <div class="text-[11px] text-zinc-500">
              {#if row.dataset_id != null}dataset id {row.dataset_id}{/if}
              {#if row.source_class_id != null}
                · source registry id {row.source_class_id}{/if}
            </div>
            {#if row.index_would_have_mapped_to && indexMismatch}
              <div class="text-[11px] text-zinc-500" data-testid="index-hint">
                by index this would have been {row.index_would_have_mapped_to.class_name}
              </div>
            {/if}
          </td>
          <td class="px-2 py-1.5 text-right font-mono">{row.boxes.toLocaleString()}</td>
          <td class="px-2 py-1.5 text-right font-mono">{row.images.toLocaleString()}</td>
          <td class="px-2 py-1.5">
            {#if row.suggestion}
              <span
                class="inline-block rounded border border-zinc-700 px-1.5 py-0.5 text-[11px] whitespace-nowrap text-zinc-300"
                data-testid="suggestion-chip"
              >
                {matchLabel.get(row.suggestion.match) ?? row.suggestion.match}
              </span>
              <div class="mt-0.5 text-[11px] text-zinc-400">
                {actionLabel.get(row.suggestion.action) ?? row.suggestion.action}{row
                  .suggestion.class_name
                  ? `: ${row.suggestion.class_name}`
                  : ''}
              </div>
              <button
                type="button"
                class="mt-0.5 text-[11px] text-blue-300 hover:underline"
                onclick={() => wizard.useSuggestion(row)}>Use suggestion</button
              >
            {:else}
              <span class="text-zinc-600">—</span>
            {/if}
          </td>
          <td class="px-2 py-1.5">
            <select
              class="select select-sm"
              aria-label="Action for {row.dataset_class}"
              value={choice.action}
              onchange={(e) =>
                wizard.setChoice(row.dataset_class, {
                  action: (e.currentTarget as HTMLSelectElement).value,
                })}
            >
              <option value="">Not chosen</option>
              {#each formats.mapping_actions as a (a.id)}
                <option value={a.id}>{a.label}</option>
              {/each}
            </select>
            {#if choice.action === 'map'}
              <select
                class="select select-sm mt-1"
                aria-label="Class for {row.dataset_class}"
                value={choice.class_id == null ? '' : String(choice.class_id)}
                onchange={(e) => {
                  const v = (e.currentTarget as HTMLSelectElement).value;
                  wizard.setChoice(row.dataset_class, {
                    class_id: v === '' ? null : Number(v),
                  });
                }}
              >
                <option value="">Pick a class…</option>
                {#each classes as c (c.id)}
                  <option value={String(c.id)}>{c.name}</option>
                {/each}
              </select>
            {:else if choice.action === 'create'}
              <input
                class="input input-sm mt-1"
                aria-label="New class name for {row.dataset_class}"
                value={choice.new_class_name}
                oninput={(e) =>
                  wizard.setChoice(row.dataset_class, {
                    new_class_name: (e.currentTarget as HTMLInputElement).value,
                  })}
              />
            {/if}
          </td>
          <td class="px-2 py-1.5" data-testid="resolved">
            {#if resolvedText(row)}
              <span class="text-emerald-300">{resolvedText(row)}</span>
            {:else if unmapped.has(row.dataset_class)}
              <span class="text-amber-300">not mapped yet</span>
            {:else}
              <span class="text-zinc-600">—</span>
            {/if}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>
