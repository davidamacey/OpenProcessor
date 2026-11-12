<script lang="ts">
  /**
   * The dashboard's VLM action. VLM work starts from the Workspace's AI
   * menu, so this links there. Whether the project has a VLM is the backend's call
   * (`GET /vlm/endpoints/active`, `source: 'off'`): with none, the action
   * links to the Models settings instead of a ready button that could only
   * fail. An unreadable answer leaves the action as it is; the run itself
   * reports its refusal.
   */
  import { resolve } from '$app/paths';
  import { onMount } from 'svelte';
  import { getActiveVlm } from '$lib/api_vlm';
  import { projectHref } from '$lib/projectPaths';

  /** False when the served next step belongs to another action (one primary per view). */
  let { primary = true }: { primary?: boolean } = $props();

  let off = $state(false);

  onMount(() => {
    const ctl = new AbortController();
    getActiveVlm(ctl.signal)
      .then((r) => {
        off = r.source === 'off';
      })
      .catch(() => {});
    return () => ctl.abort();
  });
</script>

{#if off}
  <span class="text-meta text-fg-3" data-testid="vlm-activate-first">
    No VLM is active for this project.
    <a class="text-accent-text underline" href={resolve(projectHref('/settings/models'))}
      >Activate one</a
    > to assist with it.
  </span>
{:else}
  <a
    class="btn {primary ? 'btn-primary' : ''}"
    href={resolve(projectHref('/workspace'))}
    title="Open the Workspace and use its AI menu: Assist, the auto-label loop, Auto-label"
    >Open the Workspace AI menu</a
  >
{/if}
