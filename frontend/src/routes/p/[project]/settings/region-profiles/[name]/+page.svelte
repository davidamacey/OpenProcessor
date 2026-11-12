<script lang="ts">
  /**
   * `/settings/region-profiles/[name]` — one region profile's editor
   * (any_domain_plan.md §4, §7.3, §7.4, §7.6 items 2 and 4; docs/design/
   * w4-profile-editor-ui-plan-2026-09-27.md §4). The form, its groups,
   * types, ranges and help, every model list, every validation issue, the
   * revisions, the active profile and the activation impact are served;
   * `ProfileEditor` holds the draft and follows the server.
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
  import ProfileFieldEditor from '$components/profiles/ProfileFieldEditor.svelte';
  import ProfileImpactPanel from '$components/profiles/ProfileImpactPanel.svelte';
  import ProfileTestPanel from '$components/profiles/ProfileTestPanel.svelte';
  import SegmenterStatus from '$components/profiles/SegmenterStatus.svelte';
  import { issuesForField, unplacedIssues } from '$lib/config/validationIssues';
  import { PROFILE_ACTIVE_COPY } from '$lib/profiles/profileCopy';
  import { createProfileEditor } from '$lib/profiles/profileEditorController.svelte';
  import { appliesInSaved, choiceList, groupFields } from '$lib/profiles/profileFields';
  import { createProfileList } from '$lib/profiles/profileListController.svelte';
  import { profilesAvailability } from '$lib/profiles/profilesAvailability.svelte';
  import { createConfigEditorFlow } from '$lib/config/editorFlow.svelte';
  import { projectHref } from '$lib/projectPaths';
  import type { RegionProfileDoc } from '$lib/types_profiles';
  import { keyboardStore } from '$stores/keyboard.svelte';

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
  const flow = createConfigEditorFlow({
    ed: () => ed,
    cloner,
    name: () => profileName,
    editorPath: (n) => projectHref(`/settings/region-profiles/${encodeURIComponent(n)}`),
  });

  const lastActivation = $derived(ed.active.lastActivation);
  const activationWarnings = $derived([
    ...(lastActivation?.validation?.errors ?? []),
    ...(lastActivation?.validation?.warnings ?? []),
  ]);
</script>

<div class="flex flex-col gap-4">
  <ConfigDocHeader
    name={profileName}
    doc={ed.doc}
    listPath="/settings/region-profiles"
    listLabel="All region profiles"
    lineage
  />

  <ConfigGate
    store={profilesAvailability}
    what="region profiles"
    unavailableText="Region-profile editing is not available on this backend."
    testid="profiles-unavailable"
  >
    {#if ed.loadError && !ed.doc}
      <p class="text-body text-danger-text" data-testid="profile-load-error">
        {ed.loadError}
      </p>
    {:else if !ed.doc || !ed.schema}
      <p class="text-body text-fg-3">Loading…</p>
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
          class="surface flex flex-col gap-2 p-4 text-body"
          aria-label="Activation result"
          data-testid="activation-result"
        >
          <h2 class="text-body font-semibold">
            Activated {lastActivation.active.name}{lastActivation.active.revision != null
              ? ` r${lastActivation.active.revision}`
              : ''}
          </h2>
          <p class="text-meta text-fg-2">
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
          <ConfigEditorLead {ed} {flow} noun="profile" />

          {#if shownDoc?.effective}
            {@const eff = shownDoc.effective}
            <p class="text-meta text-fg-2" data-testid="profile-effective">
              {ed.viewing ? `Revision ${ed.viewing.revision}` : 'The saved revision'}:
              finds regions with
              <span class="font-mono text-fg-1"
                >{eff.legs.length > 0 ? eff.legs.join(' + ') : 'nothing'}</span
              >;
              {eff.reads_text ? 'reads text' : 'reads no text'}{eff.text_hint_active
                ? ' (with an OCR hint)'
                : ''}; segmenter {eff.segmenter_enabled ? 'enabled' : 'not enabled'}.
            </p>
          {/if}

          <ConfigDescriptionField {ed} />

          <div class="flex flex-wrap items-center gap-4 text-meta text-fg-2">
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
            <p class="text-meta text-danger-text" data-testid="vocabulary-error">
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
                <h2 class="text-body font-semibold text-fg-1">{g.label}</h2>
                {#if g.id === 'segmenter' && ed.vocabulary}
                  <SegmenterStatus segmenters={ed.vocabulary.segmenters} />
                {/if}
                {#if g.id === 'segmenter'}
                  <div class="flex flex-col gap-2">
                    <button
                      type="button"
                      class="btn btn-sm self-start"
                      disabled={ed.promptChecking}
                      data-testid="check-segmenter-prompt"
                      onclick={() => void ed.checkSegmenterPrompt()}
                      >{ed.promptChecking
                        ? 'Checking…'
                        : 'Check segmenter prompt'}</button
                    >
                    {#if ed.promptCheckError}
                      <p class="text-meta text-danger-text">{ed.promptCheckError}</p>
                    {/if}
                    {#if ed.promptReport}
                      {@const r = ed.promptReport}
                      <div class="space-y-1" data-testid="segmenter-prompt-check">
                        <p class="text-meta text-fg-2">
                          {r.ok ? 'The prompt passes.' : 'The prompt has errors.'}
                        </p>
                        <ConfigIssueList
                          issues={[...r.errors, ...r.warnings]}
                          showField
                        />
                      </div>
                    {/if}
                  </div>
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
                  <p class="text-meta text-fg-3">
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
            onsave={() => void flow.doSave()}
            onactivate={() => flow.openActivate(d)}
          >
            {#snippet extra()}
              <ConfigActivationCheck {ed} oncheck={() => void ed.checkForActivation()} />
            {/snippet}
          </ConfigSavePanel>
          <ConfigRevisions {ed} />
        </aside>
      </div>

      {#key profileName}
        <ProfileTestPanel
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
  name={profileName}
  noun="profile"
  blurb="Items still waiting are detected with the active profile. Items already processed keep their results; a re-run is a separate step."
/>
