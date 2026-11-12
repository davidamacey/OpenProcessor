<script lang="ts">
  import { formatShortcutKey } from '$lib/keyboardDisplay';
  import KeyChips from '$components/settings/keymap/KeyChips.svelte';
  import { keymapStore } from '$stores/keymap.svelte';
  import { contextRowActions } from '$components/settings/keymap/keymapModel';
  import type { KeymapCardController } from '$components/settings/keymap/keymapCardController.svelte';

  let { ctl }: { ctl: KeymapCardController } = $props();

  const contexts = $derived(keymapStore.document.contexts);
</script>

{#each contexts as ctx (ctx.id)}
  <details class="mb-3 rounded border border-zinc-800" open>
    <summary class="cursor-pointer px-3 py-2 text-xs font-semibold text-zinc-300">
      {ctx.label}
      <span class="ml-1 font-normal text-zinc-600">{ctx.description}</span>
    </summary>
    <table class="w-full text-xs">
      <tbody>
        {#each contextRowActions(ctx.id) as action (action.id)}
          {@const field = `overrides.${action.id}`}
          {@const fieldIssues = ctl.issuesForField(field)}
          <tr class="border-t border-zinc-900">
            <td class="px-3 py-1.5 text-zinc-300">
              {action.label}
              {#if !action.available}
                <span class="ml-1 text-zinc-600">(no region profile in this project)</span
                >
              {/if}
              {#if fieldIssues.errors.length > 0}
                <div class="text-red-400">{fieldIssues.errors.join('; ')}</div>
              {/if}
              {#if fieldIssues.warnings.length > 0}
                <div class="text-amber-400">{fieldIssues.warnings.join('; ')}</div>
              {/if}
            </td>
            <td class="px-3 py-1.5">
              {#if !action.modifiable}
                <span class="rounded border border-zinc-700 px-1.5 py-0.5 text-zinc-500">
                  🔒 {ctl.keysOf(action.id).map(formatShortcutKey).join(' / ') || '—'}
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
                        class="text-zinc-500 underline hover:text-zinc-300"
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
{/each}
