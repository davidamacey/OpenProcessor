<script lang="ts">
  /**
   * `/settings/open-vocab/[name]`: one open-vocabulary set's editor. The
   * form's rows, their ranges and help, every validation issue, the
   * revisions and the active set are served; `OpenVocabEditor` holds the
   * draft and follows the server.
   */
  import { page } from '$app/state';
  import ConfigActivePanel from '$components/config/ConfigActivePanel.svelte';
  import ConfigDocHeader from '$components/config/ConfigDocHeader.svelte';
  import ConfigEditorDialogs from '$components/config/ConfigEditorDialogs.svelte';
  import ConfigActivationCheck from '$components/config/ConfigActivationCheck.svelte';
  import ConfigGate from '$components/config/ConfigGate.svelte';
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import ConfigRevisions from '$components/config/ConfigRevisions.svelte';
  import ConfigSavePanel from '$components/config/ConfigSavePanel.svelte';
  import ConfigEditorLead from '$components/config/ConfigEditorLead.svelte';
  import ConfigDescriptionField from '$components/config/ConfigDescriptionField.svelte';
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
  import { createConfigEditorFlow } from '$lib/config/editorFlow.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type {
    OpenVocabBody,
    OpenVocabFieldSchema,
    OpenVocabGatingBody,
  } from '$lib/types_openVocab';
  import type { ProfileFieldValue } from '$lib/types_profiles';
  import { classesStore } from '$stores/classes.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';

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
  const flow = createConfigEditorFlow({
    ed: () => ed,
    cloner,
    name: () => setName,
    editorPath: (n) => projectHref(`/settings/open-vocab/${encodeURIComponent(n)}`),
  });

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
  <ConfigDocHeader
    name={setName}
    doc={ed.doc}
    listPath="/settings/open-vocab"
    listLabel="All open-vocabulary sets"
    lineage
  />

  <ConfigGate
    store={openVocabAvailability}
    what="open-vocabulary sets"
    unavailableText="Open-vocabulary sets are not available on this backend."
    testid="open-vocab-unavailable"
  >
    {#if ed.loadError && !ed.doc}
      <p class="text-sm text-danger-text" data-testid="open-vocab-load-error">
        {ed.loadError}
      </p>
    {:else if !ed.doc || !ed.schema || !by}
      <p class="text-sm text-fg-3">Loading…</p>
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
          <ConfigEditorLead {ed} {flow} noun="set" />

          <ConfigDescriptionField {ed} />

          {#if advancedCount > 0}
            <label class="flex items-center gap-2 text-xs text-fg-2">
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
              <h2 class="text-sm font-semibold text-fg-1">Set</h2>
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
              <h2 class="text-sm font-semibold text-fg-1">Gating</h2>
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
            onsave={() => void flow.doSave()}
            onactivate={() => flow.openActivate(d)}
          >
            {#snippet extra()}
              <ConfigActivationCheck {ed} oncheck={() => void ed.checkActivation()} />
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

<ConfigEditorDialogs
  {flow}
  {ed}
  {cloner}
  name={setName}
  noun="set"
  blurb="Passes that run from now on use the active set. Items already found keep their results; running it over existing images is a separate step."
/>
