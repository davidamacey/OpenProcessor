<script lang="ts">
  /**
   * Confirm a sharing change for one of the active project's promoted
   * models (`PUT {scoped}/models/{name}/sharing`, projects P2 §5.5).
   * `model` is the page's CURRENT served entry, so after a
   * `revision_conflict` reload the next confirm carries the fresh served
   * revision. A refusal shows the served message verbatim; a 409 `in_use`
   * lists the served projects and offers the served `force`.
   */
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import {
    forceUnshareText,
    shareConfirmText,
    unshareConfirmText,
  } from '$lib/modelSharing';
  import type { ModelSharing } from '$lib/models/modelSharingController.svelte';
  import type { ModelInfo } from '$lib/types';
  import { toastStore } from '$stores/toast.svelte';

  interface Props {
    model: ModelInfo | null;
    sharing: ModelSharing;
    onclose: () => void;
  }
  let { model, sharing, onclose }: Props = $props();

  let errorText = $state<string | null>(null);
  let conflict = $state(false);
  let inUse = $state<string[] | null>(null);
  let forceArmed = $state(false);
  let openFor = $state<string | null>(null);

  $effect(() => {
    const name = model?.name ?? null;
    if (name !== openFor) {
      openFor = name;
      errorText = null;
      conflict = false;
      inUse = null;
      forceArmed = false;
    }
  });

  const busy = $derived(model != null && sharing.pending === model.name);

  async function confirm(force = false): Promise<void> {
    if (!model) return;
    const res = await sharing.toggle(model, force);
    if (res.ok) {
      const users = res.response.used_by ?? [];
      toastStore.success(
        (res.response.shared
          ? `${res.response.name} is shared with other projects.`
          : `${res.response.name} is no longer shared.`) +
          (users.length
            ? ` Used by: ${users.map((u) => (u.profile ? `${u.project} (${u.profile})` : u.project)).join(', ')}.`
            : ''),
      );
      onclose();
      return;
    }
    forceArmed = false;
    errorText = res.message;
    conflict = res.code === 'revision_conflict';
    inUse = res.code === 'in_use' ? res.projects : null;
  }
</script>

{#if model}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Model sharing"
    use:focusOnMount
    use:trapFocus={{ onEscape: onclose }}
    tabindex="-1"
    data-testid="share-model-dialog"
    onclick={(e) => {
      if (e.target === e.currentTarget) onclose();
    }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-1 text-base font-semibold">
        {model.shared ? 'Stop sharing' : 'Share with other projects'}
      </h3>
      <p class="mb-3 break-all font-mono text-xs text-zinc-500">{model.name}</p>
      <p class="mb-3 text-sm text-zinc-300" data-testid="share-model-text">
        {model.shared ? unshareConfirmText(model.name) : shareConfirmText(model.name)}
      </p>

      {#if errorText}
        <div class="mb-3 text-xs text-red-300" data-testid="share-model-error">
          <p>{errorText}</p>
          {#if conflict}
            <p class="mt-1 text-zinc-400" data-testid="share-model-reloaded">
              Reloaded the latest sharing state. Confirm again to apply it.
            </p>
          {/if}
          {#if inUse}
            {#if inUse.length}
              <p class="mt-1 text-zinc-400" data-testid="share-model-in-use">
                Used by: {inUse.join(', ')}
              </p>
            {/if}
            {#if forceArmed}
              <p class="mt-2 text-amber-300" data-testid="share-model-force-warning">
                {forceUnshareText(inUse)}
              </p>
              <button
                type="button"
                class="btn btn-sm mt-2 text-red-300"
                data-testid="share-model-force-confirm"
                disabled={busy}
                onclick={() => void confirm(true)}>Confirm: unshare anyway</button
              >
            {:else}
              <button
                type="button"
                class="btn btn-sm mt-2 text-red-300"
                data-testid="share-model-force"
                disabled={busy}
                onclick={() => (forceArmed = true)}>Unshare anyway</button
              >
            {/if}
          {/if}
        </div>
      {/if}

      <div class="flex items-center justify-end gap-2">
        <button type="button" class="btn" onclick={onclose} disabled={busy}>Cancel</button
        >
        <button
          type="button"
          class="btn btn-primary"
          data-testid="share-model-confirm"
          disabled={busy}
          onclick={() => void confirm()}
          >{busy ? 'Saving…' : model.shared ? 'Stop sharing' : 'Share'}</button
        >
      </div>
    </div>
  </div>
{/if}
