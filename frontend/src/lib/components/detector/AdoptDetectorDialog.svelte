<!--
  Adopt the detector's classes: the dry run is shown first (classes to create, items, what
  is blocked), then Confirm writes; the result names the items left over and links to the
  Label history, where the recorded operation is undone.
-->
<script lang="ts">
  import { resolve } from '$app/paths';
  import { onMount } from 'svelte';
  import Modal from '$lib/components/Modal.svelte';
  import Button from '$components/ui/Button.svelte';
  import ErrorState from '$components/ui/ErrorState.svelte';
  import { AdoptDetectorController } from '$lib/detector/adoptDetectorController.svelte';
  import { formatCount } from '$lib/format/count';
  import { projectHref } from '$lib/projectPaths';

  interface Props {
    ctl?: AdoptDetectorController;
    onclose: () => void;
    /** Called after a write finished, so the page re-reads the next steps. */
    ondone?: () => void;
  }
  let { ctl = new AdoptDetectorController(), onclose, ondone }: Props = $props();

  onMount(() => void ctl.check());

  async function confirm(): Promise<void> {
    await ctl.confirm();
    if (ctl.phase === 'done') ondone?.();
  }

  const busy = $derived(ctl.phase === 'checking' || ctl.phase === 'writing');
</script>

<Modal
  label="Adopt detector classes"
  onclose={busy ? undefined : onclose}
  data-testid="adopt-dialog"
>
  <h2 class="mb-2 text-title font-semibold text-fg-1">Adopt detector classes</h2>
  <p class="mb-3 text-meta text-fg-2">
    Gives each item that has only a detector class that class as a machine label. Missing
    registry classes are created. Human and imported labels are never changed.
  </p>

  {#if ctl.phase === 'checking'}
    <p class="text-body text-fg-2" role="status">Checking what would change…</p>
  {:else if ctl.phase === 'failed'}
    <ErrorState
      title="Could not adopt the detector classes"
      error={ctl.failure}
      onretry={() => void (ctl.preview ? confirm() : ctl.check())}
    />
  {:else if ctl.preview}
    {@const p = ctl.phase === 'done' && ctl.written ? ctl.written : ctl.preview}
    <dl
      class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-body"
      data-testid="adopt-summary"
    >
      <dt class="text-fg-3">Items to label</dt>
      <dd class="text-fg-1 tabular-nums" data-testid="adopt-items">
        {formatCount(p.items)}
      </dd>
      <dt class="text-fg-3">Classes to create</dt>
      <dd class="text-fg-1" data-testid="adopt-create">
        {p.classes_to_create.length ? p.classes_to_create.join(', ') : 'None'}
      </dd>
      <dt class="text-fg-3">Blocked</dt>
      <dd class="text-fg-1" data-testid="adopt-blocked">
        {#if p.blocked.length}
          {p.blocked.join(', ')} ({formatCount(p.blocked_items)} items stay without a class)
        {:else}
          None
        {/if}
      </dd>
    </dl>
    {#if ctl.phase === 'done' && ctl.written}
      <p class="mt-3 text-body text-fg-1" role="status" data-testid="adopt-result">
        Labelled {formatCount(ctl.outcome?.updated ?? 0)} items.
        {#if ctl.written.remaining > 0}
          {formatCount(ctl.written.remaining)} more are left: adopt again.
        {/if}
        {#if ctl.written.op_id}
          <a
            class="text-accent-text underline"
            data-testid="adopt-undo-link"
            href={resolve(projectHref('/quality/history'))}
            >Undo it in the Label history</a
          >.
        {/if}
      </p>
    {/if}
  {/if}

  <div class="mt-4 flex justify-end gap-2">
    <Button onclick={onclose} disabled={busy}
      >{ctl.phase === 'done' ? 'Close' : 'Cancel'}</Button
    >
    {#if ctl.phase === 'ready' || ctl.phase === 'writing'}
      <Button
        variant="primary"
        loading={ctl.phase === 'writing'}
        disabled={!ctl.preview || ctl.preview.items === 0}
        onclick={() => void confirm()}
        >Adopt {formatCount(ctl.preview?.items ?? 0)} items</Button
      >
    {/if}
  </div>
</Modal>
