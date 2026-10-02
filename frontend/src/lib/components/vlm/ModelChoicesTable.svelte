<script lang="ts">
  /**
   * Every model choice the deployment exposes, read only
   * (`GET /config/vocabulary` `model_choices`, W9.8): the label, where it
   * is set (the served scope label), its current value, the dimensions,
   * the choices, and whether it is settable (with the served
   * `settable_via` / `reason` as help). A role that has an editor links to
   * it (`modelChoiceLinks.ts`).
   */
  import { resolve } from '$app/paths';
  import { modelChoiceLink } from '$lib/vlm/modelChoiceLinks';
  import { projectHref } from '$lib/projectPaths';
  import type { ModelChoice } from '$lib/types_profiles';

  interface Props {
    choices: ModelChoice[];
    /** The served `labels.scope`. */
    scopeLabels: Record<string, string>;
  }

  let { choices, scopeLabels }: Props = $props();
</script>

{#if choices.length === 0}
  <p class="text-sm text-zinc-500" data-testid="model-choices-empty">
    No model choices served.
  </p>
{:else}
  <div class="overflow-x-auto">
    <table class="w-full text-left text-xs" data-testid="model-choices">
      <thead class="text-zinc-500">
        <tr>
          <th class="py-1 pr-3 font-normal">Choice</th>
          <th class="py-1 pr-3 font-normal">Set</th>
          <th class="py-1 pr-3 font-normal">Current</th>
          <th class="py-1 pr-3 font-normal">Dimensions</th>
          <th class="py-1 pr-3 font-normal">Choices</th>
          <th class="py-1 font-normal">Changing it</th>
        </tr>
      </thead>
      <tbody>
        {#each choices as c (c.role)}
          {@const link = modelChoiceLink(c.role)}
          <tr
            class="border-t border-zinc-800 align-top"
            data-testid="model-choice-row"
            data-role={c.role}
          >
            <td class="py-1.5 pr-3">
              {#if link}
                <a class="text-blue-300 hover:underline" href={resolve(projectHref(link))}
                  >{c.label}</a
                >
              {:else}
                <span class="text-zinc-100">{c.label}</span>
              {/if}
            </td>
            <td class="py-1.5 pr-3">{scopeLabels[c.scope] ?? c.scope}</td>
            <td class="py-1.5 pr-3 font-mono">{c.current ?? '—'}</td>
            <td class="py-1.5 pr-3 font-mono">{c.dims ?? '—'}</td>
            <td class="py-1.5 pr-3">
              {#each c.choices as ch (ch.id)}
                <span class="mr-1 inline-block">{ch.label}</span>
              {:else}
                <span class="text-zinc-500">—</span>
              {/each}
            </td>
            <td class="py-1.5 text-zinc-400">
              {c.settable ? 'settable' : 'not settable'}
              {#if c.settable_via}<span class="block">{c.settable_via}</span>{/if}
              {#if c.reason}<span class="block">{c.reason}</span>{/if}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
{/if}
