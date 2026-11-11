<script lang="ts">
  /**
   * `/near-duplicates`: the project's near-duplicate groups. Status and runs,
   * the groups of each level with their members and pair decisions, and the
   * advanced thresholds (`near_dup_profile`). Everything shown is served by
   * `{prefix}/near_duplicates/*` and `{prefix}/near_dup_profiles/*`.
   */
  import { onMount } from 'svelte';
  import NearDupGroupList from '$lib/components/nearDup/NearDupGroupList.svelte';
  import NearDupGroupModal from '$lib/components/nearDup/NearDupGroupModal.svelte';
  import NearDupProfilePanel from '$lib/components/nearDup/NearDupProfilePanel.svelte';
  import NearDupRunsTable from '$lib/components/nearDup/NearDupRunsTable.svelte';
  import NearDupStatusCard from '$lib/components/nearDup/NearDupStatusCard.svelte';
  import { createNearDupPageController } from '$lib/nearDup/nearDupPageController.svelte';
  import { createNearDupProfileController } from '$lib/nearDup/nearDupProfileController.svelte';
  import type { NearDupGroupSummary } from '$lib/types_nearDup';
  import { keyboardStore } from '$stores/keyboard.svelte';

  $effect(() => {
    keyboardStore.setScope('near-duplicates');
  });

  const ctl = createNearDupPageController();
  const profile = createNearDupProfileController(() => void ctl.loadOverview());
  let open = $state<NearDupGroupSummary | null>(null);
  let openLevel = $state(ctl.level);

  onMount(() => {
    void ctl.loadOverview();
    void ctl.showGroups(ctl.level, ctl.purpose);
    return () => ctl.dispose();
  });

  function openGroup(g: NearDupGroupSummary): void {
    openLevel = ctl.level;
    open = g;
  }
</script>

<div class="mx-auto flex max-w-6xl flex-col gap-4 p-6">
  <header class="flex flex-wrap items-center gap-3">
    <h1 class="text-2xl font-semibold tracking-tight">Near-duplicates</h1>
    <span class="grow"></span>
    <button
      type="button"
      class="btn"
      disabled={ctl.loading}
      onclick={() => {
        void ctl.loadOverview();
        void ctl.showGroups(ctl.level, ctl.purpose);
      }}>Reload</button
    >
  </header>

  {#if ctl.loadError}
    <section
      class="surface flex flex-col gap-3 p-6 text-sm"
      data-testid="near-dup-load-error"
    >
      <p class="text-red-300" role="alert">{ctl.loadError}</p>
      <button type="button" class="btn w-fit" onclick={() => void ctl.loadOverview()}
        >Retry</button
      >
    </section>
  {:else if ctl.loading && !ctl.status}
    <section class="surface p-6 text-sm text-zinc-500">
      Loading near-duplicate status…
    </section>
  {:else}
    <NearDupStatusCard {ctl} />
    <NearDupGroupList {ctl} onopen={openGroup} />
    <NearDupRunsTable runs={ctl.runs} />
  {/if}
  <NearDupProfilePanel ctl={profile} />
</div>

{#if open}
  {#key open.group_id}
    <NearDupGroupModal
      level={openLevel}
      group={open}
      onclose={() => (open = null)}
      ondecided={() => void ctl.loadOverview()}
    />
  {/key}
{/if}
