<script lang="ts">
  /**
   * The fields of a preset body as the server describes them (its served JSON Schema): type,
   * required, default, choices, limits and the help text. Beside the JSON editor; it judges nothing,
   * the server's refusal on Save does.
   */
  import { schemaFields } from '$lib/presets/presetSchemaFields';
  import type { PresetSchemaWire } from '$lib/types_presets';

  interface Props {
    schema: PresetSchemaWire | null;
    error: string | null;
  }

  let { schema, error }: Props = $props();

  const fields = $derived(schema ? schemaFields(schema) : []);
</script>

<section class="surface flex min-w-0 flex-col gap-2 p-4 text-body" aria-label="Fields">
  <h2 class="text-title font-semibold">Fields</h2>
  {#if error}
    <p class="text-danger-text" data-testid="preset-schema-error">{error}</p>
  {:else if !schema}
    <p class="text-fg-3">Loading…</p>
  {:else}
    <ul class="flex flex-col gap-2" data-testid="preset-schema-fields">
      {#each fields as f (f.path)}
        <li
          class="flex flex-col gap-0.5 border-l border-line-strong pl-2"
          data-testid="preset-schema-field"
          data-path={f.path}
        >
          <div class="flex flex-wrap items-baseline gap-x-2">
            <code class="text-meta break-all text-fg-1">{f.path}</code>
            <span class="text-meta text-fg-2">{f.type}</span>
            {#if f.required}<span class="text-meta text-warn-text">required</span>{/if}
          </div>
          {#if f.description}<p class="text-meta text-fg-2">{f.description}</p>{/if}
          {#if f.choices.length > 0}
            <p class="text-meta text-fg-2">
              One of <code>{f.choices.join(', ')}</code>
            </p>
          {/if}
          {#if f.limits.length > 0}
            <p class="text-meta text-fg-2">{f.limits.join(', ')}</p>
          {/if}
          {#if f.default !== null}
            <p class="text-meta text-fg-2">
              Default <code>{f.default}</code>
            </p>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</section>
