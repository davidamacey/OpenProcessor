<script lang="ts">
  /**
   * `/settings/open-vocab/[name]`: one open-vocabulary set's editor. The
   * form's rows, their ranges and help, every validation issue, the
   * revisions and the active set are served; `OpenVocabEditor` holds the
   * draft and follows the server.
   */
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import ConfigActivateDialog from '$components/config/ConfigActivateDialog.svelte';
  import ConfigActivePanel from '$components/config/ConfigActivePanel.svelte';
  import ConfigCloneDialog from '$components/config/ConfigCloneDialog.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import ConfigRestoreDialog from '$components/config/ConfigRestoreDialog.svelte';
  import ConfigRevisions from '$components/config/ConfigRevisions.svelte';
  import ConfigSavePanel from '$components/config/ConfigSavePanel.svelte';
  import ConfigViewingBanner from '$components/config/ConfigViewingBanner.svelte';
  import SegmenterNotice from '$components/openVocab/SegmenterNotice.svelte';
  import OpenVocabTargetsTable from '$components/openVocab/OpenVocabTargetsTable.svelte';
  import OpenVocabTestPanel from '$components/openVocab/OpenVocabTestPanel.svelte';
  import ProfileFieldEditor from '$components/profiles/ProfileFieldEditor.svelte';
  import { issuesForField } from '$lib/config/validationIssues';
  import { openVocabAvailability } from '$lib/openVocab/openVocabAvailability.svelte';
  import { OPEN_VOCAB_ACTIVE_COPY } from '$lib/openVocab/openVocabCopy';
  import { createOpenVocabEditor } from '$lib/openVocab/openVocabEditorController.svelte';
  import {
    issuePath,
    openVocabFieldAsProfileField,
    rowsByScope,
    unplacedOpenVocabIssues,
  } from '$lib/openVocab/openVocabFields';
  import { createOpenVocabList } from '$lib/openVocab/openVocabListController.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { ConfigDocBase } from '$lib/types_config';
  import type {
    OpenVocabBody,
    OpenVocabFieldSchema,
    OpenVocabGatingBody,
  } from '$lib/types_openVocab';
  import type { ProfileFieldValue } from '$lib/types_profiles';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const setName = $derived(page.params.name ?? '');
  const ed = $derived(createOpenVocabEditor(setName));
  // Clone-to-edit reuses the list controller's clone call.
  const cloner = createOpenVocabList();

  $effect(() => {
    if (openVocabAvailability.available !== true) return;
    const e = ed;
    e.start();
    return () => e.stop();
  });

  const shown = $derived<OpenVocabBody>(ed.viewing ? ed.viewing.body : ed.draftBody);
  const shownReport = $derived(ed.viewing ? ed.viewing.validation : ed.report);
  const by = $derived(ed.schema ? rowsByScope(ed.schema) : null);
  const classNames = $derived(
    classesStore.classes.filter((c) => !c.deprecated).map((c) => c.name),
  );
  const advancedCount = $derived(ed.schema?.fields.filter((f) => f.advanced).length ?? 0);

  let showAdvanced = $state(false);
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
    const doc = await cloner.clone({ name: setName, source: null }, name, '');
    if (!doc) return;
    cloning = false;
    toastStore.success(`Created ${doc.name}`);
    await goto(
      resolve(projectHref(`/settings/open-vocab/${encodeURIComponent(doc.name)}`)),
    );
  }

  /** The row as the profile field editor's row, named by its issue path so
   *  every cell on the page has its own label target. */
  const cell = (row: OpenVocabFieldSchema) => ({
    ...openVocabFieldAsProfileField(row),
    field: issuePath(row.scope, row.field),
  });

  function valueOf(row: OpenVocabFieldSchema): ProfileFieldValue | undefined {
    if (row.scope === 'set')
      return shown[row.field as keyof OpenVocabBody] as ProfileFieldValue | undefined;
    const gating: OpenVocabGatingBody = shown.gating ?? {};
    if (row.scope === 'gating')
      return gating[row.field as keyof OpenVocabGatingBody] as
        ProfileFieldValue | undefined;
    return (gating.tier3_hit_rate as Record<string, ProfileFieldValue> | undefined)?.[
      row.field
    ];
  }

  function change(row: OpenVocabFieldSchema, v: ProfileFieldValue): void {
    if (row.scope === 'set') ed.setField(row.field, v as never);
    else if (row.scope === 'gating') ed.setGatingField(row.field, v);
    else ed.setHitRateField(row.field, v);
  }

  const visible = (rows: OpenVocabFieldSchema[]) =>
    rows.filter((r) => showAdvanced || !r.advanced);

  const lastActivation = $derived(ed.active.lastActivation);
  const activationWarnings = $derived([
    ...(lastActivation?.validation?.errors ?? []),
    ...(lastActivation?.validation?.warnings ?? []),
  ]);
</script>

<div class="mx-auto flex max-w-7xl flex-col gap-4 p-6">
  <header class="flex flex-wrap items-baseline gap-3">
    <h1 class="font-mono text-2xl font-semibold tracking-tight">{setName}</h1>
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
        {#if d.revision != null && d.active_revision != null && d.revision !== d.active_revision}
          <span class="text-xs text-amber-300" data-testid="saved-vs-active"
            >saved r{d.revision}, active r{d.active_revision}</span
          >
        {/if}
      {/if}
      {#if d.cloned_from}
        <span class="font-mono text-xs text-zinc-500">cloned from {d.cloned_from}</span>
      {/if}
    {/if}
    <span class="grow"></span>
    <a
      class="text-xs text-blue-300 hover:underline"
      href={resolve(projectHref('/settings/open-vocab'))}>All open-vocabulary sets</a
    >
  </header>

  <ConfigGate
    store={openVocabAvailability}
    what="open-vocabulary sets"
    unavailableText="Open-vocabulary sets are not available on this backend."
    testid="open-vocab-unavailable"
  >
    {#if ed.loadError && !ed.doc}
      <p class="text-sm text-red-300" data-testid="open-vocab-load-error">
        {ed.loadError}
      </p>
    {:else if !ed.doc || !ed.schema || !by}
      <p class="text-sm text-zinc-500">Loading…</p>
    {:else}
      {@const d = ed.doc}
      <SegmenterNotice segmenter={ed.segmenter} />

      <ConfigActivePanel
        ctl={ed.active}
        copy={OPEN_VOCAB_ACTIVE_COPY}
        onrollback={() => ed.active.rollback()}
        ondeactivate={() => ed.active.deactivate()}
      />

      {#if lastActivation}
        <section
          class="surface flex flex-col gap-2 p-4 text-sm"
          aria-label="Activation result"
          data-testid="activation-result"
        >
          <h2 class="text-sm font-semibold">
            Activated {lastActivation.active.name}{lastActivation.active.revision != null
              ? ` r${lastActivation.active.revision}`
              : ''}
          </h2>
          {#if activationWarnings.length > 0}
            <ConfigIssueList issues={activationWarnings} showField />
          {/if}
        </section>
      {/if}

      <div class="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <section
          class="surface flex min-w-0 flex-col gap-4 p-4"
          aria-label="Open-vocabulary set fields"
        >
          <ConfigViewingBanner
            {ed}
            onrestore={() => (restoring = true)}
            onactivate={() => ed.viewing && openActivate(ed.viewing)}
          />

          {#if d.read_only}
            <div class="flex flex-wrap items-center gap-2 text-sm text-zinc-400">
              <span>This set is read-only. Clone it to make an editable copy.</span>
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

          {#if advancedCount > 0}
            <label class="flex items-center gap-2 text-xs text-zinc-400">
              <input
                type="checkbox"
                bind:checked={showAdvanced}
                data-testid="show-advanced"
              />
              Show advanced fields ({advancedCount})
            </label>
          {/if}

          <ConfigIssueList
            issues={unplacedOpenVocabIssues(shownReport, ed.schema)}
            showField
          />

          {#if visible(by.set).length > 0}
            <section
              class="flex flex-col gap-3"
              data-testid="open-vocab-group"
              data-group="set"
            >
              <h2 class="text-sm font-semibold text-zinc-200">Set</h2>
              {#each visible(by.set) as r (r.field)}
                <ProfileFieldEditor
                  field={cell(r)}
                  value={valueOf(r)}
                  issues={issuesForField(shownReport, issuePath(r.scope, r.field))}
                  choices={null}
                  applies={null}
                  readonly={!ed.editable}
                  onchange={(v) => change(r, v)}
                />
              {/each}
            </section>
          {/if}

          <OpenVocabTargetsTable
            targets={shown.targets ?? []}
            rows={by.target}
            report={shownReport}
            readonly={!ed.editable}
            {classNames}
            maxEnabledTargets={shown.max_enabled_targets ?? null}
            maxEnabledTargetsCeiling={ed.schema.max_enabled_targets_ceiling}
            onadd={() => ed.addTarget()}
            onremove={(i) => ed.removeTarget(i)}
            onmove={(i, dir) => ed.moveTarget(i, dir)}
            onchange={(i, f, v) => ed.setTargetField(i, f, v)}
          />

          {#if visible([...by.gating, ...by.tier3_hit_rate]).length > 0}
            <section
              class="flex flex-col gap-3"
              data-testid="open-vocab-group"
              data-group="gating"
            >
              <h2 class="text-sm font-semibold text-zinc-200">Gating</h2>
              {#each visible( [...by.gating, ...by.tier3_hit_rate] ) as r (r.scope + r.field)}
                <ProfileFieldEditor
                  field={cell(r)}
                  value={valueOf(r)}
                  issues={issuesForField(shownReport, issuePath(r.scope, r.field))}
                  choices={null}
                  applies={null}
                  readonly={!ed.editable}
                  onchange={(v) => change(r, v)}
                />
              {/each}
            </section>
          {/if}
        </section>

        <aside
          class="order-first flex min-w-0 flex-col gap-4 lg:sticky lg:top-4 lg:order-none lg:self-start"
        >
          <ConfigSavePanel
            {ed}
            report={shownReport}
            noun="set"
            onsave={() => void doSave()}
            onactivate={() => openActivate(d)}
          >
            {#snippet extra()}
              {#if !ed.viewing}
                <div class="flex flex-col gap-2 border-t border-zinc-800 pt-2">
                  <button
                    type="button"
                    class="btn btn-sm self-start"
                    disabled={ed.activationChecking}
                    data-testid="check-activation"
                    onclick={() => void ed.checkActivation()}
                    >{ed.activationChecking
                      ? 'Checking…'
                      : 'Check the draft for activation'}</button
                  >
                  {#if ed.activationCheckError}
                    <p class="text-xs text-red-300">{ed.activationCheckError}</p>
                  {/if}
                  {#if ed.activationReport}
                    {@const r = ed.activationReport}
                    <div class="space-y-1" data-testid="activation-check">
                      <p class="text-xs text-zinc-400">
                        As it would be checked on activation: {r.errors.length} error{r
                          .errors.length === 1
                          ? ''
                          : 's'}, {r.warnings.length} warning{r.warnings.length === 1
                          ? ''
                          : 's'}{r.force_allowed ? '; the server allows overriding' : ''}.
                      </p>
                      <ConfigIssueList issues={[...r.errors, ...r.warnings]} showField />
                    </div>
                  {/if}
                </div>
              {/if}
            {/snippet}
          </ConfigSavePanel>
          <ConfigRevisions {ed} />
        </aside>
      </div>

      {#key setName}
        <OpenVocabTestPanel
          targets={shown.targets ?? []}
          imageMaxSide={shown.image_max_side}
          dedupIou={shown.dedup_iou}
          vocabulary={ed.schema.vocabulary}
        />
      {/key}
    {/if}
  </ConfigGate>
</div>

{#if activating}
  <ConfigActivateDialog
    {ed}
    target={activating}
    blurb="Passes that run from now on use the active set. Items already found keep their results; running it over existing images is a separate step."
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
    noun="set"
    onclose={() => (restoring = false)}
    onrestored={(from, to) =>
      toastStore.success(`Restored revision ${from} as revision ${to}`)}
  />
{/if}

{#if cloning}
  <ConfigCloneDialog
    title="Clone {setName}"
    nameLabel="New set name"
    busy={cloner.busy}
    error={cloner.cloneError}
    report={cloner.cloneReport}
    onconfirm={(name) => void doClone(name)}
    oncancel={() => (cloning = false)}
  />
{/if}
