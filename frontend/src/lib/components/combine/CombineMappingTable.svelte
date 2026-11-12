<script lang="ts">
  /**
   * Step 2 — one source's class mapping. Rows are the preview's served
   * `sources[].classes` (name, count, the served `mapped_to`). `create`
   * defines a target class by name; `map` picks one of the names the
   * form's own `create` rows define (across every source); `skip` and
   * `region` need nothing more. This is deliberately not the dataset-import
   * `MappingTable`: combine maps into classes the request itself creates,
   * by name, with no registry, `resolved` or match chips.
   */
  import {
    COMBINE_MAPPING_ACTIONS,
    type CombineWizard,
  } from '$lib/combine/combineWizardController.svelte';
  import CombineIssueList from '$components/combine/CombineIssueList.svelte';
  import type { CombinePreviewSource } from '$lib/types_combine';

  interface Props {
    wizard: CombineWizard;
    source: CombinePreviewSource;
  }
  let { wizard, source }: Props = $props();

  const rows = $derived(source.classes ?? []);
  const issues = $derived(wizard.issuesFor(source.project));
</script>

<div class="space-y-1" data-testid="combine-mapping-{source.project}">
  <h3 class="text-xs font-semibold text-fg-2">
    {source.project}
    <span class="font-normal text-fg-3">({rows.length} classes)</span>
  </h3>
  {#if rows.length === 0}
    <p class="text-xs text-fg-3">No labeled classes in this source.</p>
  {:else}
    <div class="overflow-x-auto rounded border border-line">
      <table class="w-full text-xs">
        <thead class="bg-surface-1 text-left text-fg-2">
          <tr>
            <th class="px-2 py-1 font-normal">Source class</th>
            <th class="px-2 py-1 text-right font-normal">Items</th>
            <th class="px-2 py-1 font-normal">Served mapping</th>
            <th class="px-2 py-1 font-normal">Action</th>
            <th class="px-2 py-1 font-normal">Target class</th>
          </tr>
        </thead>
        <tbody>
          {#each rows as r (r.name)}
            {@const c = wizard.choiceFor(source.project, r.name)}
            <tr
              class="border-t border-line"
              data-testid="combine-map-row"
              data-class={r.name}
              data-touched={c.touched}
            >
              <td class="px-2 py-1 text-fg-1">{r.name}</td>
              <td class="px-2 py-1 text-right tabular-nums">{r.count}</td>
              <td class="px-2 py-1 text-fg-2" data-testid="combine-map-served">
                {r.mapped_to ?? 'unmapped'}
              </td>
              <td class="px-2 py-1">
                <select
                  class="rounded border border-line-strong bg-surface-1 px-1 py-0.5 text-fg-1"
                  aria-label="Action for {r.name}"
                  data-testid="combine-map-action"
                  value={c.action}
                  title={wizard.actionDescription(c.action) ?? undefined}
                  onchange={(e) =>
                    wizard.setChoice(source.project, r.name, {
                      action: (e.currentTarget as HTMLSelectElement).value,
                    })}
                >
                  <option value="">Choose…</option>
                  {#each COMBINE_MAPPING_ACTIONS as a (a)}
                    <option value={a}>{wizard.actionLabel(a)}</option>
                  {/each}
                </select>
              </td>
              <td class="px-2 py-1">
                {#if c.action === 'create'}
                  <input
                    type="text"
                    class="w-40 rounded border border-line-strong bg-surface-1 px-1 py-0.5 text-fg-1"
                    aria-label="New class name for {r.name}"
                    data-testid="combine-map-name"
                    value={c.new_class_name}
                    oninput={(e) =>
                      wizard.setChoice(source.project, r.name, {
                        new_class_name: (e.currentTarget as HTMLInputElement).value,
                      })}
                  />
                {:else if c.action === 'map'}
                  <select
                    class="rounded border border-line-strong bg-surface-1 px-1 py-0.5 text-fg-1"
                    aria-label="Map {r.name} onto"
                    data-testid="combine-map-target"
                    value={c.new_class_name}
                    onchange={(e) =>
                      wizard.setChoice(source.project, r.name, {
                        new_class_name: (e.currentTarget as HTMLSelectElement).value,
                      })}
                  >
                    <option value="">Choose…</option>
                    {#each wizard.createdNames as n (n)}
                      <option value={n}>{n}</option>
                    {/each}
                  </select>
                {:else}
                  <span class="text-fg-3">—</span>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
  <CombineIssueList {issues} testid="combine-mapping-issues-{source.project}" />
</div>
