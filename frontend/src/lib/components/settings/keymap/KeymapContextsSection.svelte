<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  import { formatShortcutKey } from '$lib/keyboardDisplay';
  import KeyChips from '$components/settings/keymap/KeyChips.svelte';
  import { keymapStore } from '$stores/keymap.svelte';
  import { contextRowActions } from '$components/settings/keymap/keymapModel';
  import type { KeymapCardController } from '$components/settings/keymap/keymapCardController.svelte';

  let { ctl }: { ctl: KeymapCardController } = $props();

  const contexts = $derived(keymapStore.document.contexts);
</script>

{#each contexts as ctx (ctx.id)}
  {@const rows = contextRowActions(ctx.id).filter((a) =>
    ctl.matches(a.label, a.id, ctx.label),
  )}
  {#if rows.length > 0}
    <details class="mb-3 rounded border border-line" open>
      <summary class="cursor-pointer px-3 py-2 text-meta font-semibold text-fg-2">
        {ctx.label}
        <span class="ml-1 font-normal text-fg-3">{ctx.description}</span>
      </summary>
      <table class="w-full text-meta">
        <tbody>
          {#each rows as action (action.id)}
            {@const field = `overrides.${action.id}`}
            {@const fieldIssues = ctl.issuesForField(field)}
            <tr class="border-t border-line">
              <td class="px-3 py-1.5 text-fg-2">
                {action.label}
                {#if !action.available}
                  <span class="ml-1 text-fg-3">(no region profile in this project)</span>
                {/if}
                {#if fieldIssues.errors.length > 0}
                  <div class="text-danger-text">{fieldIssues.errors.join('; ')}</div>
                {/if}
                {#if fieldIssues.warnings.length > 0}
                  <div class="text-warn-text">{fieldIssues.warnings.join('; ')}</div>
                {/if}
              </td>
              <td class="px-3 py-1.5">
                {#if !action.modifiable}
                  <span class="rounded border border-line-strong px-1.5 py-0.5 text-fg-3">
                    <Icon name="lock" size={12} class="mr-1 -mt-0.5 inline-block" />{ctl
                      .keysOf(action.id)
                      .map(formatShortcutKey)
                      .join(' / ') || '—'}
                  </span>
                {:else}
                  <KeyChips
                    {ctl}
                    keys={ctl.keysOf(action.id)}
                    capturing={ctl.capturingActionId === action.id}
                    onremove={(k) => ctl.removeKey(action.id, k)}
                    onstart={() => ctl.startCapture(action.id)}
                  >
                    {#snippet extra()}
                      {#if ctl.isChanged(action.id)}
                        <button
                          class="text-fg-3 underline hover:text-fg-2"
                          onclick={() => ctl.resetOneLocal(action.id)}
                        >
                          reset
                        </button>
                      {/if}
                    {/snippet}
                  </KeyChips>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </details>
  {/if}
{/each}
