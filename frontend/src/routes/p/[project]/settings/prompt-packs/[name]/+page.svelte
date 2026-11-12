<script lang="ts">
  /**
   * `/settings/prompt-packs/[name]` — one prompt pack's editor
   * (any_domain_plan.md §3, §7.2, §7.6 items 1, 3 and 4; docs/design/
   * w3-pack-editor-ui-plan-2026-09-27.md §3). The fields, their help and
   * chips, every validation issue, the revisions and the active pack are
   * served; `PackEditor` holds the draft and follows the server.
   */
  import { page } from '$app/state';
  import ConfigDocHeader from '$components/config/ConfigDocHeader.svelte';
  import ConfigEditorDialogs from '$components/config/ConfigEditorDialogs.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import ConfigRevisions from '$components/config/ConfigRevisions.svelte';
  import ConfigSavePanel from '$components/config/ConfigSavePanel.svelte';
  import ConfigEditorLead from '$components/config/ConfigEditorLead.svelte';
  import ConfigDescriptionField from '$components/config/ConfigDescriptionField.svelte';
  import PackActivePanel from '$components/packs/PackActivePanel.svelte';
  import PackFieldEditor from '$components/packs/PackFieldEditor.svelte';
  import PackTestPanel from '$components/packs/PackTestPanel.svelte';
  import { issuesForField, unplacedIssues } from '$lib/config/validationIssues';
  import { createPackEditor } from '$lib/packs/packEditorController.svelte';
  import { createPackList } from '$lib/packs/packListController.svelte';
  import { packsAvailability } from '$lib/packs/packsAvailability.svelte';
  import { createConfigEditorFlow } from '$lib/config/editorFlow.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { PackSchemaField } from '$lib/types_packs';
  import { keyboardStore } from '$stores/keyboard.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const packName = $derived(page.params.name ?? '');
  const ed = $derived(createPackEditor(packName));
  // Clone-to-edit reuses the list controller's clone call.
  const cloner = createPackList();

  $effect(() => {
    if (packsAvailability.available !== true) return;
    const e = ed;
    e.start();
    return () => e.stop();
  });

  /** What the fields show: a revision being viewed, else the draft. */
  const shown = $derived(ed.viewing ? ed.viewing.body : ed.draftBody);
  const shownReport = $derived(ed.viewing ? ed.viewing.validation : ed.report);
  const fieldIds = $derived(ed.schema?.fields.map((f) => f.field) ?? []);

  /** Schema rows grouped by the served `group`, in first-appearance order;
   *  the heading is the served label of the call with that id. */
  const groups = $derived.by(() => {
    const schema = ed.schema;
    if (!schema)
      return [] as Array<{ id: string; label: string; fields: PackSchemaField[] }>;
    const out: Array<{ id: string; label: string; fields: PackSchemaField[] }> = [];
    for (const f of schema.fields) {
      let g = out.find((x) => x.id === f.group);
      if (!g) {
        g = {
          id: f.group,
          label: schema.calls.find((c) => c.id === f.group)?.label ?? f.group,
          fields: [],
        };
        out.push(g);
      }
      g.fields.push(f);
    }
    return out;
  });

  const flow = createConfigEditorFlow({
    ed: () => ed,
    cloner,
    name: () => packName,
    editorPath: (n) => projectHref(`/settings/prompt-packs/${encodeURIComponent(n)}`),
  });
</script>

<div class="mx-auto flex max-w-7xl flex-col gap-4 p-6">
  <ConfigDocHeader
    name={packName}
    doc={ed.doc}
    listPath="/settings/prompt-packs"
    listLabel="All prompt packs"
  />

  <ConfigGate
    store={packsAvailability}
    what="prompt packs"
    unavailableText="Prompt-pack editing is not available on this backend."
    testid="packs-unavailable"
  >
    {#if ed.loadError && !ed.doc}
      <p class="text-sm text-danger-text" data-testid="pack-load-error">{ed.loadError}</p>
    {:else if !ed.doc || !ed.schema}
      <p class="text-sm text-fg-3">Loading…</p>
    {:else}
      {@const d = ed.doc}
      <PackActivePanel ctl={ed.active} onrollback={() => ed.active.rollback()} />

      <div class="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <section class="surface flex min-w-0 flex-col gap-4 p-4" aria-label="Pack fields">
          <ConfigEditorLead {ed} {flow} noun="pack" />

          <ConfigDescriptionField {ed} />

          {#if ed.schema.placeholders.length > 0}
            <details class="text-xs">
              <summary class="cursor-pointer text-fg-2">Placeholders</summary>
              <dl class="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                {#each ed.schema.placeholders as p (p.name)}
                  <dt class="font-mono text-info-text">{`{${p.name}}`}</dt>
                  <dd class="text-fg-2">
                    {p.meaning}
                    {#if p.example}<span class="block font-mono text-fg-3"
                        >e.g. {p.example}</span
                      >{/if}
                  </dd>
                {/each}
              </dl>
            </details>
          {/if}

          <ConfigIssueList issues={unplacedIssues(shownReport, fieldIds)} showField />

          {#each groups as g (g.id)}
            <section
              class="flex flex-col gap-3"
              data-testid="pack-group"
              data-group={g.id}
            >
              <h2 class="text-sm font-semibold text-fg-1">{g.label}</h2>
              {#each g.fields as f (f.field)}
                <PackFieldEditor
                  field={f}
                  value={shown[f.field]}
                  issues={issuesForField(shownReport, f.field)}
                  readonly={!ed.editable}
                  onchange={(v) => ed.setField(f.field, v)}
                />
              {/each}
            </section>
          {/each}
        </section>

        <aside
          class="order-first flex min-w-0 flex-col gap-4 lg:sticky lg:top-4 lg:order-none lg:self-start"
        >
          <ConfigSavePanel
            {ed}
            report={shownReport}
            noun="pack"
            onsave={() => void flow.doSave()}
            onactivate={() => flow.openActivate(d)}
          />
          <ConfigRevisions {ed} />
        </aside>
      </div>

      {#key packName}
        <PackTestPanel
          calls={ed.schema.calls}
          name={d.name}
          revision={ed.viewing ? ed.viewing.revision : d.revision}
          savedOnly={!ed.editable}
          draft={ed.draftBody}
        />
      {/key}
    {/if}
  </ConfigGate>
</div>

<ConfigEditorDialogs
  {flow}
  {ed}
  {cloner}
  name={packName}
  noun="pack"
  blurb="Every VLM step that doesn't pick its own pack uses the active pack."
/>
