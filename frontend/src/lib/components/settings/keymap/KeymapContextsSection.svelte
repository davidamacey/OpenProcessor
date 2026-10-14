<script lang="ts">
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { formatShortcutKey } from '$lib/keyboardDisplay';
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
                <div class="flex flex-wrap items-center gap-1">
                  {#each ctl.keysOf(action.id) as k (k)}
                    <span
                      class="flex items-center gap-1 rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-blue-300"
                    >
                      {formatShortcutKey(k)}
                      <button
                        class="text-zinc-500 hover:text-red-400"
                        aria-label={`remove ${k}`}
                        onclick={() => ctl.removeKey(action.id, k)}
                      >
                        ×
                      </button>
                    </span>
                  {/each}
                  {#if ctl.capturingActionId === action.id}
                    <button
                      data-capture
                      use:focusOnMount
                      class="rounded border border-blue-600 bg-blue-950/40 px-1.5 py-0.5 text-blue-300"
                      onkeydown={ctl.onCaptureKeydown}
                      onblur={ctl.cancelCapture}
                    >
                      press a key…
                    </button>
                  {:else}
                    <button
                      class="rounded border border-zinc-700 px-1.5 py-0.5 text-zinc-400 hover:text-white"
                      onclick={() => ctl.startCapture(action.id)}
                    >
                      Change
                    </button>
                  {/if}
                  {#if ctl.isChanged(action.id)}
                    <button
                      class="text-zinc-500 underline hover:text-zinc-300"
                      onclick={() => ctl.resetOneLocal(action.id)}
                    >
                      reset
                    </button>
                  {/if}
                </div>
              {/if}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </details>
{/each}
