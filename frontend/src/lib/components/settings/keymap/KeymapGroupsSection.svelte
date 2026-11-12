<script lang="ts">
  import KeyChips from '$components/settings/keymap/KeyChips.svelte';
  import {
    contextLabel,
    groupLabel,
    groupMembers,
    namedGroups,
  } from '$components/settings/keymap/keymapModel';
  import type { KeymapCardController } from '$components/settings/keymap/keymapCardController.svelte';

  let { ctl }: { ctl: KeymapCardController } = $props();
</script>

{#each namedGroups() as g (g)}
  {@const members = groupMembers(g)}
  {@const canonical = ctl.groupCanonicalKeys(g)}
  <details class="mb-3 rounded border border-zinc-800" open data-testid={`group-${g}`}>
    <summary class="cursor-pointer px-3 py-2 text-xs font-semibold text-zinc-300">
      {groupLabel(g)}
      <span class="ml-1 font-normal text-zinc-600">applies on every page</span>
    </summary>
    <table class="w-full text-xs">
      <tbody>
        <tr class="border-t border-zinc-900">
          <td class="px-3 py-1.5 text-zinc-300">{groupLabel(g)} (every page)</td>
          <td class="px-3 py-1.5">
            <KeyChips
              {ctl}
              keys={canonical}
              capturing={ctl.capturingGroupId === g}
              onremove={(k) => ctl.removeGroupKey(g, k)}
              onstart={() => ctl.startCaptureGroup(g)}
            />
          </td>
        </tr>
      </tbody>
    </table>
    <details class="border-t border-zinc-900 px-3 py-1.5">
      <summary class="cursor-pointer text-xs text-zinc-500">Customize per page</summary>
      <table class="w-full text-xs">
        <tbody>
          {#each members as action (action.id)}
            {@const field = `overrides.${action.id}`}
            {@const fieldIssues = ctl.issuesForField(field)}
            {@const detached = ctl.memberDetached(action.id)}
            <tr class="border-t border-zinc-900" data-testid={`member-${action.id}`}>
              <td class="px-3 py-1.5 text-zinc-300">
                {contextLabel(action.context)}
                {#if detached}
                  <span class="ml-1 text-amber-400" data-testid={`detached-${action.id}`}>
                    differs from the group
                  </span>
                {/if}
                {#if fieldIssues.errors.length > 0}
                  <div class="text-red-400">{fieldIssues.errors.join('; ')}</div>
                {/if}
                {#if fieldIssues.warnings.length > 0}
                  <div class="text-amber-400">{fieldIssues.warnings.join('; ')}</div>
                {/if}
              </td>
              <td class="px-3 py-1.5">
                <KeyChips
                  {ctl}
                  keys={ctl.keysOf(action.id)}
                  capturing={ctl.capturingActionId === action.id}
                  onremove={(k) => ctl.removeKey(action.id, k)}
                  onstart={() => ctl.startCapture(action.id)}
                >
                  {#snippet extra()}
                    {#if detached}
                      <button
                        class="text-amber-500 underline hover:text-amber-300"
                        onclick={() => ctl.resetMemberToGroup(action.id)}
                      >
                        reset to group
                      </button>
                    {/if}
                  {/snippet}
                </KeyChips>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </details>
  </details>
{/each}
