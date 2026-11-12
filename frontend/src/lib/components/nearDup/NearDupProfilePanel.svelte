<!--
  Advanced: the project's near-duplicate thresholds (`near_dup_profile`). The
  seeded defaults are shown beside each field. A save writes the next revision
  and activates it; the server then asks for a rebuild of both levels.
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import type { NearDupProfileController } from '$lib/nearDup/nearDupProfileController.svelte';

  let { ctl }: { ctl: NearDupProfileController } = $props();

  onMount(() => {
    void ctl.load();
    return () => ctl.dispose();
  });
</script>

<details class="surface p-4" data-testid="near-dup-profile">
  <summary class="cursor-pointer text-base font-semibold">Advanced: thresholds</summary>
  <div class="mt-3 flex flex-col gap-3">
    <p class="text-xs text-fg-2">
      Cosine similarity cuts per level. Members at or above the duplicate cut are
      duplicates; members at or above the leak cut must stay on one split side, so the
      leak cut cannot exceed the duplicate cut. Few projects need to change these; the
      defaults are the seeded values.
    </p>
    {#if ctl.loading && !ctl.draft}
      <p class="text-xs text-fg-3">Loading the profile…</p>
    {:else if ctl.loadError}
      <p
        class="text-sm text-danger-text"
        role="alert"
        data-testid="near-dup-profile-error"
      >
        {ctl.loadError}
      </p>
    {:else if ctl.draft}
      {@const d = ctl.draft}
      {@const s = ctl.seeded}
      <p class="text-xs text-fg-2">
        Active: <span class="font-mono">{ctl.name}</span> revision {ctl.revision}
      </p>
      <div class="grid gap-4 md:grid-cols-2">
        <fieldset class="flex flex-col gap-2 rounded border border-line p-3">
          <legend class="px-1 text-sm font-medium">Image level</legend>
          <label class="flex flex-col gap-1 text-xs">
            <span class="text-fg-2"
              >Duplicate cosine{#if s}&nbsp;(default {s.image.dedup_cosine}){/if}</span
            >
            <input
              class="input w-32"
              type="number"
              step="any"
              data-testid="np-image-dedup"
              bind:value={d.image.dedup_cosine}
            />
          </label>
          <label class="flex flex-col gap-1 text-xs">
            <span class="text-fg-2"
              >Leak cosine{#if s}&nbsp;(default {s.image.leak_cosine}){/if}</span
            >
            <input
              class="input w-32"
              type="number"
              step="any"
              data-testid="np-image-leak"
              bind:value={d.image.leak_cosine}
            />
          </label>
          <label class="flex flex-col gap-1 text-xs">
            <span class="text-fg-2"
              >Neighbours per member (k){#if s}&nbsp;(default {s.image.k}){/if}</span
            >
            <input
              class="input w-32"
              type="number"
              step="1"
              data-testid="np-image-k"
              bind:value={d.image.k}
            />
          </label>
          <label class="flex flex-col gap-1 text-xs">
            <span class="text-fg-2"
              >Largest group{#if s}&nbsp;(default {s.image.max_group_size}){/if}</span
            >
            <input
              class="input w-32"
              type="number"
              step="1"
              data-testid="np-image-max"
              bind:value={d.image.max_group_size}
            />
          </label>
        </fieldset>
        <fieldset class="flex flex-col gap-2 rounded border border-line p-3">
          <legend class="px-1 text-sm font-medium">Item level</legend>
          <label class="flex flex-col gap-1 text-xs">
            <span class="text-fg-2"
              >Duplicate cosine{#if s}&nbsp;(default {s.item.dedup_cosine}){/if}</span
            >
            <input
              class="input w-32"
              type="number"
              step="any"
              data-testid="np-item-dedup"
              bind:value={d.item.dedup_cosine}
            />
          </label>
          <label class="flex flex-col gap-1 text-xs">
            <span class="text-fg-2"
              >Leak cosine{#if s}&nbsp;(default {s.item.leak_cosine}){/if}</span
            >
            <input
              class="input w-32"
              type="number"
              step="any"
              data-testid="np-item-leak"
              bind:value={d.item.leak_cosine}
            />
          </label>
          <label class="flex flex-col gap-1 text-xs">
            <span class="text-fg-2"
              >Largest group{#if s}&nbsp;(default {s.item.max_group_size}){/if}</span
            >
            <input
              class="input w-32"
              type="number"
              step="1"
              data-testid="np-item-max"
              bind:value={d.item.max_group_size}
            />
          </label>
          <label class="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              data-testid="np-item-same-class"
              bind:checked={d.item.same_detector_class}
            />
            <span class="text-fg-2"
              >Only group crops of the same detector class{#if s}&nbsp;(default {s.item
                  .same_detector_class
                  ? 'on'
                  : 'off'}){/if}</span
            >
          </label>
        </fieldset>
      </div>
      {#if ctl.saveError}
        <div
          class="text-xs text-danger-text"
          role="alert"
          data-testid="near-dup-profile-refused"
        >
          <p>{ctl.saveError}</p>
          {#if ctl.reasons.length > 0}
            <ul class="list-disc pl-5">
              {#each ctl.reasons as reason (reason)}<li>{reason}</li>{/each}
            </ul>
          {/if}
        </div>
      {/if}
      {#if ctl.notice}
        <p
          class="text-xs text-success-text"
          role="status"
          data-testid="near-dup-profile-notice"
        >
          {ctl.notice}
        </p>
      {/if}
      <div class="flex flex-wrap gap-2">
        <button
          type="button"
          class="btn btn-primary"
          disabled={!ctl.dirty || ctl.saving}
          data-testid="near-dup-profile-save"
          onclick={() => void ctl.save()}
          >{ctl.saving ? 'Saving…' : 'Save and activate'}</button
        >
        <button
          type="button"
          class="btn"
          disabled={!ctl.dirty || ctl.saving}
          onclick={() => ctl.reset()}>Revert</button
        >
        {#if s}
          <button
            type="button"
            class="btn"
            disabled={ctl.saving}
            data-testid="near-dup-profile-defaults"
            onclick={() => ctl.useSeeded()}>Use the seeded defaults</button
          >
        {/if}
      </div>
    {/if}
  </div>
</details>
