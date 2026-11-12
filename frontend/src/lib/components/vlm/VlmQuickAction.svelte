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
  <p class="text-body text-fg-2" data-testid="vlm-activate-first">
    Assisting with the VLM needs an active VLM.
    <a class="text-accent-text underline" href={resolve(projectHref('/settings/models'))}
      >Activate one in Models settings</a
    >
  </p>
{:else}
  <a
    class="btn btn-primary"
    href={resolve(projectHref('/workspace'))}
    title="Open the Workspace and use its AI menu: Assist, the auto-label loop, Auto-label"
    >Open the Workspace AI menu</a
  >
{/if}
