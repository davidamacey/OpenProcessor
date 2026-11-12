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

{#each namedGroups().filter( (g) => ctl.matches(groupLabel(g), ...groupMembers(g).flatMap( (a) => [a.label, contextLabel(a.context)] )) ) as g (g)}
  {@const members = groupMembers(g)}
  {@const canonical = ctl.groupCanonicalKeys(g)}
  <details class="mb-3 rounded border border-line" open data-testid={`group-${g}`}>
    <summary class="cursor-pointer px-3 py-2 text-xs font-semibold text-fg-2">
      {groupLabel(g)}
      <span class="ml-1 font-normal text-fg-3">applies on every page</span>
    </summary>
    <table class="w-full text-xs">
      <tbody>
        <tr class="border-t border-line">
          <td class="px-3 py-1.5 text-fg-2">{groupLabel(g)} (every page)</td>
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
    <details class="border-t border-line px-3 py-1.5">
      <summary class="cursor-pointer text-xs text-fg-3">Customize per page</summary>
      <table class="w-full text-xs">
        <tbody>
          {#each members as action (action.id)}
            {@const field = `overrides.${action.id}`}
            {@const fieldIssues = ctl.issuesForField(field)}
            {@const detached = ctl.memberDetached(action.id)}
            <tr class="border-t border-line" data-testid={`member-${action.id}`}>
              <td class="px-3 py-1.5 text-fg-2">
                {contextLabel(action.context)}
                {#if detached}
                  <span class="ml-1 text-warn-text" data-testid={`detached-${action.id}`}>
                    differs from the group
                  </span>
                {/if}
                {#if fieldIssues.errors.length > 0}
                  <div class="text-danger-text">{fieldIssues.errors.join('; ')}</div>
                {/if}
                {#if fieldIssues.warnings.length > 0}
                  <div class="text-warn-text">{fieldIssues.warnings.join('; ')}</div>
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
                        class="text-warn-text underline hover:text-warn-text"
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
