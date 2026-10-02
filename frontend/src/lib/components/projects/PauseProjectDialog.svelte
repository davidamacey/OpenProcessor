<script lang="ts">
  /**
   * Confirm pausing or resuming one project's pipeline (`POST
   * {prefix}/pause` / `{prefix}/resume`, projects P2 §5.1), addressed
   * through that project's own served `prefix`. A refusal shows the
   * served message verbatim.
   */
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import type { ProjectSummary } from '$lib/types_projects';
  import { projectPauseStore } from '$stores/projectPause.svelte';
  import { toastStore } from '$stores/toast.svelte';

  interface Props {
    /** The project, and whether this dialog pauses (true) or resumes it. */
    target: { project: ProjectSummary; pause: boolean } | null;
    onclose: () => void;
    /** Fires after the server accepted the change, so the caller re-reads
     *  the served list (whose `paused` is what its chips show). */
    onchanged?: () => void;
  }
  let { target, onclose, onchanged }: Props = $props();

  let busy = $state(false);
  let errorText = $state<string | null>(null);
  let openFor = $state<string | null>(null);

  $effect(() => {
    const key = target ? `${target.project.slug}:${target.pause}` : null;
    if (key !== openFor) {
      openFor = key;
      errorText = null;
    }
  });

  async function confirm(): Promise<void> {
    if (!target) return;
    const { project, pause } = target;
    busy = true;
    errorText = null;
    const res = await projectPauseStore.set(project, pause);
    busy = false;
    if (!res.ok) {
      errorText = res.message;
      return;
    }
    toastStore.success(
      res.paused
        ? `Paused "${project.display_name}".`
        : `Resumed "${project.display_name}".`,
    );
    onchanged?.();
    onclose();
  }
</script>

{#if target}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label={target.pause ? 'Pause project' : 'Resume project'}
    use:focusOnMount
    use:trapFocus={{ onEscape: onclose }}
    tabindex="-1"
    data-testid="pause-project-dialog"
    onclick={(e) => {
      if (e.target === e.currentTarget) onclose();
    }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-1 text-base font-semibold">
        {target.pause ? 'Pause pipeline' : 'Resume pipeline'}
      </h3>
      <p class="mb-3 font-mono text-xs text-zinc-500">{target.project.slug}</p>
      <p class="mb-3 text-sm text-zinc-300" data-testid="pause-project-text">
        {#if target.pause}
          Pause {target.project.display_name}'s pipeline? Workers skip this project until
          it's resumed. Nothing is deleted, and labeling in the app keeps working.
        {:else}
          Resume {target.project.display_name}'s pipeline? Workers pick it up on their
          next cycle.
        {/if}
      </p>
      {#if errorText}
        <p class="mb-3 text-xs text-red-300" data-testid="pause-project-error">
          {errorText}
        </p>
      {/if}
      <div class="flex items-center justify-end gap-2">
        <button type="button" class="btn" onclick={onclose} disabled={busy}>Cancel</button
        >
        <button
          type="button"
          class="btn btn-primary"
          data-testid="pause-project-confirm"
          disabled={busy}
          onclick={() => void confirm()}
          >{busy ? 'Saving…' : target.pause ? 'Pause' : 'Resume'}</button
        >
      </div>
    </div>
  </div>
{/if}
