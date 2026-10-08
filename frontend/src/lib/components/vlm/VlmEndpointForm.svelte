<script lang="ts">
  /**
   * The endpoint form, rendered from the served schema (§7.8.5): one
   * `ProfileFieldEditor` per row through the `vlmFieldAsProfileField`
   * adapter, grouped by the served `groups[]`, the advanced rows behind a
   * toggle. `api_key_ref` is a picker over the served `secret_refs` (its
   * `empty_choice` first) with whether the host has that secret beside it;
   * there is no key input anywhere (the row's served help names the host
   * command). `catalog_id` is a picker over the served catalog. Issues are
   * the server's, placed on the row that owns them.
   */
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import ProfileFieldEditor from '$components/profiles/ProfileFieldEditor.svelte';
  import { issuesForField, unplacedIssues } from '$lib/config/validationIssues';
  import { groupFields } from '$lib/profiles/profileFields';
  import type { ValidationReport } from '$lib/types_config';
  import type { ProfileFieldValue } from '$lib/types_profiles';
  import type {
    VlmCatalogResponse,
    VlmEndpointBody,
    VlmEndpointList,
    VlmEndpointSchema,
  } from '$lib/types_vlm';
  import {
    vlmChoices,
    vlmFieldAsProfileField,
    vlmSchemaAsProfileSchema,
  } from '$lib/vlm/vlmFields';

  interface Props {
    schema: VlmEndpointSchema;
    body: VlmEndpointBody;
    report: ValidationReport | null;
    list: VlmEndpointList | null;
    catalog: VlmCatalogResponse | null;
    readonly?: boolean;
    onchange: (field: string, value: ProfileFieldValue) => void;
  }

  let {
    schema,
    body,
    report,
    list,
    catalog,
    readonly = false,
    onchange,
  }: Props = $props();

  let showAdvanced = $state(false);

  const groups = $derived(groupFields(vlmSchemaAsProfileSchema(schema)));
  const fieldIds = $derived(schema.fields.map((f) => f.field));
  const advancedCount = $derived(schema.fields.filter((f) => f.advanced).length);

  function keyPresent(): boolean | null {
    const ref = body.api_key_ref;
    if (ref == null) return null;
    return list?.secret_refs.find((r) => r.ref === ref)?.present ?? null;
  }
</script>

<div class="flex flex-col gap-4" data-testid="vlm-endpoint-form">
  {#if advancedCount > 0}
    <label class="flex items-center gap-2 text-xs text-zinc-400">
      <input type="checkbox" bind:checked={showAdvanced} data-testid="show-advanced" />
      Show advanced fields ({advancedCount})
    </label>
  {/if}

  <ConfigIssueList issues={unplacedIssues(report, fieldIds)} showField />

  {#each groups as g (g.id)}
    {@const visible = g.fields.filter((f) => showAdvanced || !f.advanced)}
    {#if visible.length > 0}
      <section class="flex flex-col gap-3" data-testid="vlm-group" data-group={g.id}>
        <h2 class="text-sm font-semibold text-zinc-200">{g.label}</h2>
        {#each visible as f (f.field)}
          {@const row = schema.fields.find((r) => r.field === f.field)!}
          <ProfileFieldEditor
            field={vlmFieldAsProfileField(row)}
            value={body[f.field as keyof VlmEndpointBody] as
              ProfileFieldValue | undefined}
            issues={issuesForField(report, f.field)}
            choices={vlmChoices(row, list, catalog)}
            applies={null}
            {readonly}
            onchange={(v) => onchange(f.field, v)}
          />
          {#if f.field === 'api_key_ref' && body.api_key_ref != null}
            <p class="-mt-2 text-xs text-zinc-500" data-testid="vlm-key-present">
              {keyPresent() === true
                ? 'The host has this secret.'
                : keyPresent() === false
                  ? 'The host does not have this secret.'
                  : 'Whether the host has this secret is not known.'}
            </p>
          {/if}
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
</div>
