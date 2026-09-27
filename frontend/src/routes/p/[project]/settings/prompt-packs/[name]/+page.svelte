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
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import PackActivePanel from '$components/packs/PackActivePanel.svelte';
  import PackFieldEditor from '$components/packs/PackFieldEditor.svelte';
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import PackTestPanel from '$components/packs/PackTestPanel.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import { formatTimestamp } from '$lib/formatDate';
  import { issuesForField, unplacedIssues } from '$lib/config/validationIssues';
  import { createPackEditor } from '$lib/packs/packEditorController.svelte';
  import { createPackList } from '$lib/packs/packListController.svelte';
  import { packsAvailability } from '$lib/packs/packsAvailability.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { PackSchemaField, PromptPackDoc } from '$lib/types_packs';
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

  const counts = $derived({
    errors: shownReport?.errors.length ?? 0,
    warnings: (shownReport?.warnings ?? []).filter((i) => i.severity === 'warning')
      .length,
    info: (shownReport?.warnings ?? []).filter((i) => i.severity === 'info').length,
  });

  let activating = $state<PromptPackDoc | null>(null);
  let force = $state(false);
  let restoring = $state(false);
  let cloning = $state(false);
  let cloneName = $state('');

  function openActivate(target: PromptPackDoc): void {
    ed.active.clearAction();
    force = false;
    activating = target;
  }

  async function doActivate(): Promise<void> {
    const target = activating;
    if (!target) return;
    const ok = await ed.active.activate(target.name, target.revision, force);
    if (ok) {
      activating = null;
      toastStore.success(
        `Activated ${target.name}${target.revision == null ? '' : ` r${target.revision}`}`,
      );
    } else if (!ed.active.activateReport?.force_allowed) {
      force = false;
    }
  }

  async function doSave(): Promise<void> {
    if (await ed.save()) toastStore.success(`Saved revision ${ed.doc?.revision}`);
  }

  async function doRestore(): Promise<void> {
    const rev = ed.viewing?.revision;
    if (await ed.restoreViewed()) {
      restoring = false;
      toastStore.success(`Restored revision ${rev} as revision ${ed.doc?.revision}`);
    }
  }

  async function doClone(): Promise<void> {
    const doc = await cloner.clone({ name: packName, source: null }, cloneName, '');
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
      <span class="text-sm text-zinc-400" data-testid="pack-meta">
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
          {#if ed.viewing}
            {@const v = ed.viewing}
            <div
              class="flex flex-wrap items-center gap-2 rounded border border-sky-500/40 bg-sky-500/10 px-3 py-2 text-sm text-sky-100"
              data-testid="viewing-banner"
            >
              <span>Viewing revision {v.revision} (read-only)</span>
              <span class="grow"></span>
              <button type="button" class="btn btn-sm" onclick={() => ed.closeRevision()}
                >Back to latest</button
              >
              {#if !d.read_only}
                <button
                  type="button"
                  class="btn btn-sm"
                  onclick={() => (restoring = true)}
                  data-testid="restore-revision">Restore as new revision</button
                >
              {/if}
              <button
                type="button"
                class="btn btn-sm"
                onclick={() => openActivate(v)}
                data-testid="activate-viewed">Activate revision {v.revision}</button
              >
            </div>
          {/if}

          {#if d.read_only}
            <div class="flex flex-wrap items-center gap-2 text-sm text-zinc-400">
              <span>This pack is read-only. Clone it to make an editable copy.</span>
              <button
                type="button"
                class="btn btn-sm"
                onclick={() => {
                  cloner.clearClone();
                  cloneName = '';
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
          <section class="surface flex flex-col gap-2 p-4 text-sm" aria-label="Save">
            <div
              class="flex flex-wrap items-center gap-2 text-xs"
              data-testid="report-counts"
            >
              <span class={counts.errors > 0 ? 'text-red-300' : 'text-zinc-400'}
                >{counts.errors} error{counts.errors === 1 ? '' : 's'}</span
              >
              <span class={counts.warnings > 0 ? 'text-amber-300' : 'text-zinc-400'}
                >{counts.warnings} warning{counts.warnings === 1 ? '' : 's'}</span
              >
              <span class="text-zinc-400"
                >{counts.info} note{counts.info === 1 ? '' : 's'}</span
              >
              {#if ed.validating}<span class="text-zinc-500">checking…</span>{/if}
            </div>
            {#if ed.validateError}
              <p class="text-xs text-red-300">
                Could not check the draft: {ed.validateError}
              </p>
            {/if}
            {#if ed.editable}
              <div class="flex flex-wrap gap-2">
                <button
                  type="button"
                  class="btn btn-primary btn-sm"
                  disabled={!ed.canSave}
                  onclick={() => void doSave()}
                  data-testid="pack-save">{ed.saving ? 'Saving…' : 'Save'}</button
                >
                <button
                  type="button"
                  class="btn btn-sm"
                  disabled={!ed.dirty || ed.saving}
                  onclick={() => void ed.reloadLatest()}>Discard edits</button
                >
              </div>
              <p class="text-xs text-zinc-500">
                {ed.dirty ? 'Unsaved edits.' : 'No unsaved edits.'} Saving writes a new revision;
                it doesn't change the active pack.
              </p>
            {/if}
            {#if ed.saveError}
              <p class="text-xs text-red-300" data-testid="save-error">{ed.saveError}</p>
            {/if}
            {#if ed.conflict}
              <div
                class="space-y-2 rounded border border-amber-500/40 bg-amber-500/10 p-2 text-xs text-amber-100"
                data-testid="save-conflict"
              >
                <p>{ed.conflict.message}</p>
                <div class="flex flex-wrap gap-2">
                  <button
                    type="button"
                    class="btn btn-sm"
                    onclick={() => void ed.reloadLatest()}
                    >Reload{ed.conflict.currentRevision != null
                      ? ` revision ${ed.conflict.currentRevision}`
                      : ''} (discard my edits)</button
                  >
                  {#if ed.conflict.currentRevision != null}
                    <button
                      type="button"
                      class="btn btn-sm"
                      onclick={() => ed.keepMine()}
                      data-testid="keep-mine">Keep my edits</button
                    >
                  {/if}
                </div>
              </div>
            {/if}
            {#if ed.remoteChanged}
              <div
                class="space-y-2 rounded border border-sky-500/40 bg-sky-500/10 p-2 text-xs text-sky-100"
                data-testid="remote-changed"
              >
                <p>This pack changed on the server while you were editing.</p>
                <button
                  type="button"
                  class="btn btn-sm"
                  onclick={() => void ed.reloadLatest()}
                  >Load it (discard my edits)</button
                >
              </div>
            {/if}
            <div class="border-t border-zinc-800 pt-2">
              <button
                type="button"
                class="btn btn-sm"
                onclick={() => openActivate(d)}
                data-testid="pack-activate"
                >Activate{d.revision != null ? ` revision ${d.revision}` : ''}</button
              >
            </div>
          </section>

          {#if ed.revisions}
            <section
              class="surface flex flex-col gap-2 p-4 text-sm"
              aria-label="Revisions"
            >
              <h2 class="text-sm font-semibold">Revisions</h2>
              {#if ed.revisionError}<p class="text-xs text-red-300">
                  {ed.revisionError}
                </p>{/if}
              <ul class="space-y-1" data-testid="revisions">
                {#each ed.revisions as r (r.revision)}
                  <li
                    class="flex flex-wrap items-center gap-2 text-xs"
                    data-testid="revision-row"
                    data-revision={r.revision}
                  >
                    <span class="font-mono">r{r.revision}</span>
                    <span class="text-zinc-500" title={r.saved_at}
                      >{formatTimestamp(r.saved_at)}</span
                    >
                    {#if r.revision === d.revision}<span class="text-zinc-400"
                        >latest</span
                      >{/if}
                    <span class="grow"></span>
                    <button
                      type="button"
                      class="btn btn-sm"
                      disabled={ed.viewing?.revision === r.revision}
                      onclick={() => void ed.viewRevision(r.revision)}>View</button
                    >
                    {#if r.description}
                      <span class="w-full text-zinc-400">{r.description}</span>
                    {/if}
                    {#if r.cloned_from}
                      <span class="w-full font-mono text-zinc-500"
                        >cloned from {r.cloned_from}</span
                      >
                    {/if}
                  </li>
                {/each}
              </ul>
            </section>
          {/if}
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
  {@const target = activating}
  {@const rep = ed.active.activateReport ?? target.validation}
  <ConfirmDialog
    title="Activate {target.name}{target.revision != null
      ? ` revision ${target.revision}`
      : ''}"
    confirmLabel={force ? 'Activate anyway' : 'Activate'}
    danger={force}
    busy={ed.active.busy}
    onconfirm={() => void doActivate()}
    oncancel={() => {
      activating = null;
      ed.active.clearAction();
    }}
  >
    <p data-testid="activate-from-to">
      <span class="font-mono"
        >{ed.active.active?.active.name ?? 'none'}{ed.active.active?.active.revision !=
        null
          ? ` r${ed.active.active.active.revision}`
          : ''}</span
      >
      →
      <strong class="font-mono"
        >{target.name}{target.revision != null ? ` r${target.revision}` : ''}</strong
      >
    </p>
    <p class="text-xs text-zinc-400">
      Every VLM step that doesn't pick its own pack uses the active pack.
    </p>
    {#if ed.dirty && !ed.viewing}
      <p class="text-xs text-amber-300" data-testid="activate-unsaved-note">
        This activates the saved revision; your unsaved edits are not included.
      </p>
    {/if}
    {#if rep}
      <ConfigIssueList issues={[...rep.errors, ...rep.warnings]} showField />
    {/if}
    {#if ed.active.actionError}
      <p class="text-red-300" data-testid="activate-error">{ed.active.actionError}</p>
    {/if}
    {#if ed.active.activateReport?.force_allowed}
      <label class="flex items-center gap-2 text-xs text-amber-200">
        <input type="checkbox" bind:checked={force} data-testid="activate-force" />
        Activate anyway (the server allows overriding these errors)
      </label>
    {/if}
  </ConfirmDialog>
{/if}

{#if restoring && ed.viewing}
  <ConfirmDialog
    title="Restore revision {ed.viewing.revision}"
    confirmLabel="Restore"
    busy={ed.saving}
    onconfirm={() => void doRestore()}
    oncancel={() => (restoring = false)}
  >
    <p>
      Saves revision {ed.viewing.revision}'s text as a new revision. It doesn't change the
      active pack.
    </p>
    {#if ed.dirty}
      <p class="text-xs text-amber-300">Your unsaved edits are replaced.</p>
    {/if}
    {#if ed.saveError}<p class="text-red-300">{ed.saveError}</p>{/if}
    {#if ed.conflict}<p class="text-red-300">{ed.conflict.message}</p>{/if}
  </ConfirmDialog>
{/if}

{#if cloning}
  <ConfirmDialog
    title="Clone {packName}"
    confirmLabel="Clone"
    busy={cloner.busy}
    confirmDisabled={cloneName.trim() === ''}
    onconfirm={() => void doClone()}
    oncancel={() => (cloning = false)}
  >
    <label class="flex flex-col gap-1 text-xs text-zinc-400">
      New pack name
      <input class="input input-sm font-mono" bind:value={cloneName} />
    </label>
    {#if cloner.cloneError}<p class="text-red-300">{cloner.cloneError}</p>{/if}
    {#if cloner.cloneReport}
      <ConfigIssueList
        issues={[...cloner.cloneReport.errors, ...cloner.cloneReport.warnings]}
        showField
      />
    {/if}
  </ConfirmDialog>
{/if}
