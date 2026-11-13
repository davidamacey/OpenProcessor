<script lang="ts">
  /**
   * The Gallery's "Duplicates" control: which photos to list (all, one per near-duplicate group,
   * the members of a group, the ones in no group), what near-duplicates are, the served status of
   * the grouping and the one button that runs it. Nothing is deleted. The status and the run are
   * the near-duplicate page controller's; the thresholds and the history live in Settings.
   */
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import Icon from '$components/ui/Icon.svelte';
  import Popover from '$components/ui/Popover.svelte';
  import { formatTimestamp } from '$lib/format/date';
  import { createNearDupPageController } from '$lib/nearDup/nearDupPageController.svelte';
  import { runModeFor } from '$lib/nearDup/staleRefusal';
  import { PRESET_KINDS, presetPath } from '$lib/presets/presetKinds';
  import { projectHref } from '$lib/projectPaths';
  import type { NearDupFilter } from '$lib/types_images';

  const NEAR_DUP_KIND = PRESET_KINDS.find((k) => k.id === 'near_dup_profile')!;

  /** The served `near_dup` values, in plain words; `all` leaves the filter off. */
  const OPTIONS = [
    { value: 'all', label: 'All photos' },
    {
      value: 'collapsed',
      label: 'One per group',
      title: 'Collapse each near-duplicate group to its representative',
    },
    {
      value: 'member',
      label: 'In a group',
      title: 'Photos that belong to a near-duplicate group',
    },
    {
      value: 'none',
      label: 'Not in a group',
      title: 'Photos in no near-duplicate group',
    },
  ];

  interface Props {
    value: NearDupFilter | null;
    onchange: (v: NearDupFilter | null) => void;
  }
  let { value, onchange }: Props = $props();

  const ctl = createNearDupPageController();
  onMount(() => {
    void ctl.loadStatus();
    return () => ctl.dispose();
  });

  const image = $derived(ctl.status?.levels.find((l) => l.level === 'image') ?? null);
  const statusText = $derived.by(() => {
    if (!image) return null;
    if (!image.current_run) return 'Not grouped yet.';
    const run = image.current_run;
    const when = formatTimestamp(run.finished_at ?? run.started_at);
    if (image.pending_count > 0)
      return `${image.pending_count.toLocaleString()} new photo${image.pending_count === 1 ? ' is' : 's are'} not grouped yet.`;
    return `Grouped ${when} · ${run.dedup_groups.toLocaleString()} group${run.dedup_groups === 1 ? '' : 's'} among ${run.members_scored.toLocaleString()} photos · up to date.`;
  });
  const runLabel = $derived(
    !image?.current_run
      ? 'Find near-duplicates'
      : image.pending_count > 0
        ? `Group ${image.pending_count.toLocaleString()} new photo${image.pending_count === 1 ? '' : 's'}`
        : null,
  );
  const current = $derived(
    OPTIONS.find((o) => o.value === (value ?? 'all')) ?? OPTIONS[0]!,
  );
</script>

<Popover label="Duplicates" panelClass="w-96 p-3">
  {#snippet trigger({ open, attrs })}
    <button
      type="button"
      class="btn {value ? 'border-accent/60 text-accent-text' : ''}"
      data-testid="duplicates-toggle"
      aria-label="Duplicates: {current.label}"
      {...attrs}
    >
      Duplicates: {current.label}
      <Icon name={open ? 'chevron-up' : 'chevron-down'} size={14} />
    </button>
  {/snippet}
  {#snippet children(close)}
    <div class="flex flex-col gap-3 text-meta" data-testid="duplicates-panel">
      <p class="text-fg-2" data-testid="duplicates-purpose">
        Near-duplicates are photos that look almost the same, such as burst shots,
        re-uploads or video frames. Nothing is deleted. You decide per export or training
        run whether to keep one photo per group. One rule is fixed: a group never sits on
        both sides of the train/test split, so the test score is not inflated.
      </p>
      <div role="radiogroup" aria-label="Duplicates" class="flex flex-col gap-1">
        {#each OPTIONS as o (o.value)}
          <button
            type="button"
            role="radio"
            aria-checked={(value ?? 'all') === o.value}
            title={o.title}
            class="flex items-center justify-between rounded px-2 py-1 text-left text-body hover:bg-surface-2 {(value ??
              'all') === o.value
              ? 'bg-surface-2 text-white'
              : 'text-fg-1'}"
            data-testid="duplicates-option-{o.value}"
            onclick={() => {
              onchange(o.value === 'all' ? null : (o.value as NearDupFilter));
              close();
            }}
          >
            {o.label}
            {#if (value ?? 'all') === o.value}<Icon name="check" size={14} />{/if}
          </button>
        {/each}
      </div>
      <div class="flex flex-col gap-2 border-t border-line pt-2">
        {#if ctl.loadError}
          <p class="text-danger-text" role="alert">{ctl.loadError}</p>
        {:else if statusText}
          <p class="text-fg-1 tabular-nums" data-testid="duplicates-status">
            {statusText}
          </p>
        {/if}
        {#if ctl.runMessage}
          <p class="text-fg-2" role="status">{ctl.runMessage}</p>
        {/if}
        {#if ctl.runError}
          <p class="text-danger-text" role="alert">{ctl.runError}</p>
        {/if}
        <div class="flex flex-wrap items-center gap-2">
          {#if runLabel && image}
            <button
              type="button"
              class="btn"
              disabled={ctl.busyLevel !== null}
              data-testid="duplicates-run"
              onclick={() => void ctl.startRun('image', runModeFor(image), 'status')}
            >
              {ctl.busyLevel ? 'Grouping…' : runLabel}
            </button>
          {/if}
          <a
            class="text-accent-text underline underline-offset-2"
            href={resolve(projectHref(presetPath(NEAR_DUP_KIND)))}
            >Thresholds and history</a
          >
        </div>
      </div>
    </div>
  {/snippet}
</Popover>
