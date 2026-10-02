<script lang="ts">
  /**
   * `/settings/prompt-packs/[name]` — one prompt pack's editor
   * (any_domain_plan.md §3, §7.2, §7.6 items 1, 3 and 4; docs/design/
   * w3-pack-editor-ui-plan-2026-09-27.md §3). The fields, their help and
   * chips, every validation issue, the revisions and the active pack are
   * served; `PackEditor` holds the draft and follows the server.
   */
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import ConfigActivateDialog from '$components/config/ConfigActivateDialog.svelte';
  import ConfigCloneDialog from '$components/config/ConfigCloneDialog.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import ConfigRestoreDialog from '$components/config/ConfigRestoreDialog.svelte';
  import ConfigRevisions from '$components/config/ConfigRevisions.svelte';
  import ConfigSavePanel from '$components/config/ConfigSavePanel.svelte';
  import ConfigViewingBanner from '$components/config/ConfigViewingBanner.svelte';
  import PackActivePanel from '$components/packs/PackActivePanel.svelte';
  import PackFieldEditor from '$components/packs/PackFieldEditor.svelte';
  import PackTestPanel from '$components/packs/PackTestPanel.svelte';
  import { issuesForField, unplacedIssues } from '$lib/config/validationIssues';
  import { createPackEditor } from '$lib/packs/packEditorController.svelte';
  import { createPackList } from '$lib/packs/packListController.svelte';
  import { packsAvailability } from '$lib/packs/packsAvailability.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { ConfigDocBase } from '$lib/types_config';
  import type { PackSchemaField } from '$lib/types_packs';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

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
   *  the heading is the served label of the call with that id (W3-Q3). */
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

  let activating = $state<ConfigDocBase<unknown> | null>(null);
  let restoring = $state(false);
  let cloning = $state(false);

  function openActivate(target: ConfigDocBase<unknown>): void {
    ed.active.clearAction();
    activating = target;
  }

  async function doSave(): Promise<void> {
    if (await ed.save()) toastStore.success(`Saved revision ${ed.doc?.revision}`);
  }

  async function doClone(name: string): Promise<void> {
    const doc = await cloner.clone({ name: packName, source: null }, name, '');
    if (!doc) return;
    cloning = false;
    toastStore.success(`Created ${doc.name}`);
    await goto(
      resolve(projectHref(`/settings/prompt-packs/${encodeURIComponent(doc.name)}`)),
    );
  }
</script>

<div class="mx-auto flex max-w-7xl flex-col gap-4 p-6">
  <header class="flex flex-wrap items-baseline gap-3">
    <h1 class="font-mono text-2xl font-semibold tracking-tight">{packName}</h1>
    {#if ed.doc}
      {@const d = ed.doc}
      <span class="text-sm text-zinc-400" data-testid="config-meta">
        {d.source}{d.revision != null ? ` · revision ${d.revision}` : ''}{d.read_only
          ? ' · read-only'
          : ''}
      </span>
      {#if d.active}
        <span
          class="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 text-xs text-emerald-300"
          >active{d.active_revision != null ? ` r${d.active_revision}` : ''}</span
        >
      {/if}
    {/if}
    <span class="grow"></span>
    <a
      class="text-xs text-blue-300 hover:underline"
      href={resolve(projectHref('/settings/prompt-packs'))}>All prompt packs</a
    >
  </header>

  <ConfigGate
    store={packsAvailability}
    what="prompt packs"
    unavailableText="Prompt-pack editing is not available on this backend."
    testid="packs-unavailable"
  >
    {#if ed.loadError && !ed.doc}
      <p class="text-sm text-red-300" data-testid="pack-load-error">{ed.loadError}</p>
    {:else if !ed.doc || !ed.schema}
      <p class="text-sm text-zinc-500">Loading…</p>
    {:else}
      {@const d = ed.doc}
      <PackActivePanel ctl={ed.active} onrollback={() => ed.active.rollback()} />

      <div class="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <section class="surface flex min-w-0 flex-col gap-4 p-4" aria-label="Pack fields">
          <ConfigViewingBanner
            {ed}
            onrestore={() => (restoring = true)}
            onactivate={() => ed.viewing && openActivate(ed.viewing)}
          />

          {#if d.read_only}
            <div class="flex flex-wrap items-center gap-2 text-sm text-zinc-400">
              <span>This pack is read-only. Clone it to make an editable copy.</span>
              <button
                type="button"
                class="btn btn-sm"
                onclick={() => {
                  cloner.clearClone();
                  cloning = true;
                }}>Clone to edit</button
              >
            </div>
          {/if}

          <label class="flex flex-col gap-1 text-xs text-zinc-400">
            Description
            <input
              class="input input-sm text-sm"
              value={ed.viewing ? (ed.viewing.description ?? '') : ed.draftDescription}
              readonly={!ed.editable}
              oninput={(e) =>
                ed.setDescription((e.currentTarget as HTMLInputElement).value)}
            />
          </label>

          {#if ed.schema.placeholders.length > 0}
            <details class="text-xs">
              <summary class="cursor-pointer text-zinc-300">Placeholders</summary>
              <dl class="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                {#each ed.schema.placeholders as p (p.name)}
                  <dt class="font-mono text-sky-300">{`{${p.name}}`}</dt>
                  <dd class="text-zinc-400">
                    {p.meaning}
                    {#if p.example}<span class="block font-mono text-zinc-500"
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
              <h2 class="text-sm font-semibold text-zinc-200">{g.label}</h2>
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
            onsave={() => void doSave()}
            onactivate={() => openActivate(d)}
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

{#if activating}
  <ConfigActivateDialog
    {ed}
    target={activating}
    blurb="Every VLM step that doesn't pick its own pack uses the active pack."
    onclose={() => (activating = null)}
    onactivated={(t) =>
      toastStore.success(
        `Activated ${t.name}${t.revision == null ? '' : ` r${t.revision}`}`,
      )}
  />
{/if}

{#if restoring}
  <ConfigRestoreDialog
    {ed}
    noun="pack"
    onclose={() => (restoring = false)}
    onrestored={(from, to) =>
      toastStore.success(`Restored revision ${from} as revision ${to}`)}
  />
{/if}

{#if cloning}
  <ConfigCloneDialog
    title="Clone {packName}"
    nameLabel="New pack name"
    busy={cloner.busy}
    error={cloner.cloneError}
    report={cloner.cloneReport}
    onconfirm={(name) => void doClone(name)}
    oncancel={() => (cloning = false)}
  />
{/if}
