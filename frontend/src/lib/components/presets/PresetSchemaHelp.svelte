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

<section class="surface flex min-w-0 flex-col gap-2 p-4 text-sm" aria-label="Fields">
  <h2 class="text-base font-semibold">Fields</h2>
  {#if error}
    <p class="text-red-300" data-testid="preset-schema-error">{error}</p>
  {:else if !schema}
    <p class="text-zinc-500">Loading…</p>
  {:else}
    <ul class="flex flex-col gap-2" data-testid="preset-schema-fields">
      {#each fields as f (f.path)}
        <li
          class="flex flex-col gap-0.5 border-l border-zinc-700 pl-2"
          data-testid="preset-schema-field"
          data-path={f.path}
        >
          <div class="flex flex-wrap items-baseline gap-x-2">
            <span class="font-mono text-xs break-all text-zinc-100">{f.path}</span>
            <span class="text-xs text-zinc-400">{f.type}</span>
            {#if f.required}<span class="text-xs text-amber-300">required</span>{/if}
          </div>
          {#if f.description}<p class="text-xs text-zinc-300">{f.description}</p>{/if}
          {#if f.choices.length > 0}
            <p class="text-xs text-zinc-400">
              One of <span class="font-mono">{f.choices.join(', ')}</span>
            </p>
          {/if}
          {#if f.limits.length > 0}
            <p class="text-xs text-zinc-400">{f.limits.join(', ')}</p>
          {/if}
          {#if f.default !== null}
            <p class="text-xs text-zinc-400">
              Default <span class="font-mono">{f.default}</span>
            </p>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</section>
