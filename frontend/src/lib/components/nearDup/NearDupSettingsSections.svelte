<script lang="ts">
  /**
   * The near-duplicate sections of Settings › Near-duplicate profiles, under the profile list:
   * what near-duplicates are and what the app does with them, the runs (status, full rebuild,
   * history), the pinned pairs and the object-level leak groups. Everything shown is served by
   * `{prefix}/near_duplicates/*`; the thresholds are the profile documents above.
   */
  import { onMount } from 'svelte';
  import Section from '$components/ui/Section.svelte';
  import NearDupDecisionsPanel from '$lib/components/nearDup/NearDupDecisionsPanel.svelte';
  import NearDupLeakGroupsTable from '$lib/components/nearDup/NearDupLeakGroupsTable.svelte';
  import NearDupRunsPanel from '$lib/components/nearDup/NearDupRunsPanel.svelte';
  import NearDupRunsTable from '$lib/components/nearDup/NearDupRunsTable.svelte';
  import { createNearDupDecisionsController } from '$lib/nearDup/nearDupDecisionsController.svelte';
  import { createNearDupPageController } from '$lib/nearDup/nearDupPageController.svelte';

  const ctl = createNearDupPageController();
  const decisions = createNearDupDecisionsController(() => void ctl.loadOverview());

  onMount(() => {
    void ctl.loadOverview();
    void ctl.showGroups('item', 'leak');
    void decisions.load();
    return () => {
      ctl.dispose();
      decisions.dispose();
    };
  });
</script>

<div class="mt-6 flex flex-col gap-2" data-testid="near-dup-settings">
  <Section title="What near-duplicates are for">
    <p class="max-w-3xl text-body text-fg-2" data-testid="near-dup-explainer">
      Near-duplicates are photos that look almost the same, like burst shots or the same
      scene a second apart. They are grouped so you can see one photo per group in the
      Gallery, and so a test photo never has a near-copy in training. Nothing is ever
      deleted. Collapsing near-duplicates in an export or a training run is a choice you
      make for that run, and it is off unless you turn it on. The thresholds in the
      profile above decide what counts as almost the same.
    </p>
  </Section>

  {#if ctl.loadError}
    <p class="text-body text-danger-text" role="alert" data-testid="near-dup-load-error">
      {ctl.loadError}
    </p>
  {:else}
    <Section
      title="Runs"
      description="Groups are rebuilt from the active profile. New photos are grouped from the Gallery's Duplicates control."
    >
      <NearDupRunsPanel {ctl} />
      <div class="mt-3"><NearDupRunsTable runs={ctl.runs} /></div>
    </Section>
  {/if}

  <Section
    title="Pinned pairs"
    description="Pairs you marked as the same or different. A pin applies from the next run."
  >
    <NearDupDecisionsPanel ctl={decisions} />
  </Section>

  <Section
    title="Object leak groups (advanced)"
    description="Crops that must stay on one side of a train/test split."
  >
    <NearDupLeakGroupsTable {ctl} />
  </Section>
</div>
