<script lang="ts">
  import Icon from '$components/ui/Icon.svelte';
  import type { RegistryClass } from '$lib/types_classes';
  import { classesStore } from '$stores/classes.svelte';

  // Collapsible list of deprecated classes; a merged class names its target
  // instead of offering Restore.
  let {
    deprecatedRows,
    busy,
    restoreClassAction,
  }: {
    deprecatedRows: RegistryClass[];
    busy: boolean;
    restoreClassAction: (cls: RegistryClass) => Promise<void>;
  } = $props();

  let showDeprecated = $state<boolean>(false);
</script>

{#if deprecatedRows.length > 0}
  <section class="mt-4">
    <button
      type="button"
      class="flex items-center gap-2 text-xs text-fg-2 hover:text-fg-1"
      onclick={() => (showDeprecated = !showDeprecated)}
    >
      <Icon name={showDeprecated ? 'chevron-down' : 'chevron-right'} size={14} />
      Deprecated classes ({deprecatedRows.length})
    </button>
    {#if showDeprecated}
      <div class="surface mt-2 overflow-auto">
        <table class="w-full text-sm">
          <thead class="border-b border-line text-left text-xs uppercase text-fg-3">
            <tr>
              <th class="px-3 py-2 font-medium">ID</th>
              <th class="px-3 py-2 font-medium">Name</th>
              <th class="px-3 py-2 font-medium">Group</th>
              <th class="px-3 py-2"><span class="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            {#each deprecatedRows as cls (cls.id)}
              <tr class="border-b border-line text-fg-3">
                <td class="px-3 py-1.5 font-mono text-xs">{cls.id}</td>
                <td class="px-3 py-1.5 line-through">{cls.name}</td>
                <td class="px-3 py-1.5">{cls.group ?? '—'}</td>
                <td class="px-3 py-1.5 text-right">
                  {#if cls.merged_into != null}
                    <!-- 51b05d7: a merged class can't be restored (its
                         crops live on the target); say where they went. -->
                    <span class="text-xs text-fg-3" data-testid="merged-into-{cls.id}">
                      merged into {classesStore.byId(cls.merged_into)?.name ??
                        `#${cls.merged_into}`}
                    </span>
                  {:else}
                    <button
                      type="button"
                      class="btn"
                      data-testid="restore-{cls.id}"
                      onclick={() => void restoreClassAction(cls)}
                      disabled={busy}
                    >
                      Restore
                    </button>
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </section>
{/if}
