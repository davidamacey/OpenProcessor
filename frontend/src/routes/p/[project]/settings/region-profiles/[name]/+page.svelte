<script lang="ts">
  /**
   * `/settings/region-profiles/[name]` — one region profile's editor
   * (any_domain_plan.md §4, §7.3, §7.4, §7.6 items 2 and 4; docs/design/
   * w4-profile-editor-ui-plan-2026-09-27.md §4). The form, its groups,
   * types, ranges and help, every model list, every validation issue, the
   * revisions, the active profile and the activation impact are served;
   * `ProfileEditor` holds the draft and follows the server.
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
  import ProfileFieldEditor from '$components/profiles/ProfileFieldEditor.svelte';
  import ProfileImpactPanel from '$components/profiles/ProfileImpactPanel.svelte';
  import SegmenterStatus from '$components/profiles/SegmenterStatus.svelte';
  import { issuesForField, unplacedIssues } from '$lib/config/validationIssues';
  import { PROFILE_ACTIVE_COPY } from '$lib/profiles/profileCopy';
  import { createProfileEditor } from '$lib/profiles/profileEditorController.svelte';
  import { appliesInSaved, choiceList, groupFields } from '$lib/profiles/profileFields';
  import { createProfileList } from '$lib/profiles/profileListController.svelte';
  import { profilesAvailability } from '$lib/profiles/profilesAvailability.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { ConfigDocBase } from '$lib/types_config';
  import type { RegionProfileDoc } from '$lib/types_profiles';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { toastStore } from '$stores/toast.svelte';

  $effect(() => {
    keyboardStore.setScope('settings');
  });

  const profileName = $derived(page.params.name ?? '');
  const ed = $derived(createProfileEditor(profileName));
  // Clone-to-edit reuses the list controller's clone call.
  const cloner = createProfileList();

  $effect(() => {
    if (profilesAvailability.available !== true) return;
    const e = ed;
    e.start();
    return () => e.stop();
  });

  /** What the form shows: a revision being viewed, else the draft. */
  const shownDoc = $derived<RegionProfileDoc | null>(ed.viewing ?? ed.doc);
  const shown = $derived(ed.viewing ? ed.viewing.body : ed.draftBody);
  const shownReport = $derived(ed.viewing ? ed.viewing.validation : ed.report);
  const fieldIds = $derived(ed.schema?.fields.map((f) => f.field) ?? []);
  const groups = $derived(ed.schema ? groupFields(ed.schema) : []);
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
    const doc = await cloner.clone({ name: profileName, source: null }, name, '');
    if (!doc) return;
    cloning = false;
    toastStore.success(`Created ${doc.name}`);
    await goto(
      resolve(projectHref(`/settings/region-profiles/${encodeURIComponent(doc.name)}`)),
    );
  }

  const lastActivation = $derived(ed.active.lastActivation);
  const activationWarnings = $derived([
    ...(lastActivation?.validation?.errors ?? []),
    ...(lastActivation?.validation?.warnings ?? []),
  ]);
</script>

<div class="mx-auto flex max-w-7xl flex-col gap-4 p-6">
  <header class="flex flex-wrap items-baseline gap-3">
    <h1 class="font-mono text-2xl font-semibold tracking-tight">{profileName}</h1>
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
      href={resolve(projectHref('/settings/region-profiles'))}>All region profiles</a
    >
  </header>

  <ConfigGate
    store={profilesAvailability}
    what="region profiles"
    unavailableText="Region-profile editing is not available on this backend."
    testid="profiles-unavailable"
  >
    {#if ed.loadError && !ed.doc}
      <p class="text-sm text-red-300" data-testid="profile-load-error">{ed.loadError}</p>
    {:else if !ed.doc || !ed.schema}
      <p class="text-sm text-zinc-500">Loading…</p>
    {:else}
      {@const d = ed.doc}
      <ConfigActivePanel
        ctl={ed.active}
        copy={PROFILE_ACTIVE_COPY}
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
          <p class="text-xs text-zinc-400">
            The app shows a reload notice when the region profile it runs on changes.
          </p>
          {#if activationWarnings.length > 0}
            <ConfigIssueList issues={activationWarnings} showField />
          {/if}
          {#if lastActivation.impact}
            <ProfileImpactPanel impact={lastActivation.impact} />
          {/if}
        </section>
      {/if}

      <div class="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <section
          class="surface flex min-w-0 flex-col gap-4 p-4"
          aria-label="Profile fields"
        >
          <ConfigViewingBanner
            {ed}
            onrestore={() => (restoring = true)}
            onactivate={() => ed.viewing && openActivate(ed.viewing)}
          />

          {#if d.read_only}
            <div class="flex flex-wrap items-center gap-2 text-sm text-zinc-400">
              <span>This profile is read-only. Clone it to make an editable copy.</span>
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

          {#if shownDoc?.effective}
            {@const eff = shownDoc.effective}
            <p class="text-xs text-zinc-400" data-testid="profile-effective">
              {ed.viewing ? `Revision ${ed.viewing.revision}` : 'The saved revision'}:
              finds regions with
              <span class="font-mono text-zinc-200"
                >{eff.legs.length > 0 ? eff.legs.join(' + ') : 'nothing'}</span
              >;
              {eff.reads_text ? 'reads text' : 'reads no text'}{eff.text_hint_active
                ? ' (with an OCR hint)'
                : ''}; segmenter {eff.segmenter_enabled ? 'enabled' : 'not enabled'}.
            </p>
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

          <div class="flex flex-wrap items-center gap-4 text-xs text-zinc-400">
            {#if advancedCount > 0}
              <label class="flex items-center gap-2">
                <input
                  type="checkbox"
                  bind:checked={showAdvanced}
                  data-testid="show-advanced"
                />
                Show advanced fields ({advancedCount})
              </label>
            {/if}
            <label class="flex items-center gap-2">
              <input
                type="checkbox"
                checked={ed.includeOtherProjects}
                data-testid="include-other-projects"
                onchange={(e) =>
                  void ed.setIncludeOtherProjects(
                    (e.currentTarget as HTMLInputElement).checked,
                  )}
              />
              Include other projects' shared models
            </label>
          </div>
          {#if ed.vocabularyError}
            <p class="text-xs text-red-300" data-testid="vocabulary-error">
              Could not load the model lists: {ed.vocabularyError}
            </p>
          {/if}

          <ConfigIssueList issues={unplacedIssues(shownReport, fieldIds)} showField />

          {#each groups as g (g.id)}
            {@const visible = g.fields.filter((f) => showAdvanced || !f.advanced)}
            {#if visible.length > 0 || (g.id === 'segmenter' && ed.vocabulary)}
              <section
                class="flex flex-col gap-3"
                data-testid="profile-group"
                data-group={g.id}
              >
                <h2 class="text-sm font-semibold text-zinc-200">{g.label}</h2>
                {#if g.id === 'segmenter' && ed.vocabulary}
                  <SegmenterStatus segmenters={ed.vocabulary.segmenters} />
                {/if}
                {#each visible as f (f.field)}
                  <ProfileFieldEditor
                    field={f}
                    value={shown[f.field]}
                    issues={issuesForField(shownReport, f.field)}
                    choices={choiceList(ed.vocabulary, f.choices_from)}
                    applies={appliesInSaved(f.applies_when, shownDoc?.effective)}
                    readonly={!ed.editable}
                    onchange={(v) => ed.setField(f.field, v)}
                  />
                {/each}
                {#if !showAdvanced && visible.length < g.fields.length}
                  <p class="text-xs text-zinc-500">
                    {g.fields.length - visible.length} advanced field{g.fields.length -
                      visible.length ===
                    1
                      ? ''
                      : 's'} hidden
                  </p>
                {/if}
              </section>
            {/if}
          {/each}
        </section>

        <aside
          class="order-first flex min-w-0 flex-col gap-4 lg:sticky lg:top-4 lg:order-none lg:self-start"
        >
          <ConfigSavePanel
            {ed}
            report={shownReport}
            noun="profile"
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
                    onclick={() => void ed.checkForActivation()}
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
    {/if}
  </ConfigGate>
</div>

{#if activating}
  <ConfigActivateDialog
    {ed}
    target={activating}
    blurb="Items still waiting are detected with the active profile. Items already processed keep their results; a re-run is a separate step."
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
    noun="profile"
    onclose={() => (restoring = false)}
    onrestored={(from, to) =>
      toastStore.success(`Restored revision ${from} as revision ${to}`)}
  />
{/if}

{#if cloning}
  <ConfigCloneDialog
    title="Clone {profileName}"
    nameLabel="New profile name"
    busy={cloner.busy}
    error={cloner.cloneError}
    report={cloner.cloneReport}
    onconfirm={(name) => void doClone(name)}
    oncancel={() => (cloning = false)}
  />
{/if}
