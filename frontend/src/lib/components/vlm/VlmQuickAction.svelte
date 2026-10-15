<script lang="ts">
  /**
   * The dashboard's VLM action. A VLM run is an assist run on a cluster,
   * sub-cluster, selection or crop, started from the cluster page, so this
   * links there. Whether the project has a VLM is the backend's call
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
  <a
    class="btn"
    data-testid="vlm-activate-first"
    href={resolve(projectHref('/settings/models'))}
    title="This project has no VLM. Activate one on the Models settings page."
    >Assist with the VLM: activate one first</a
  >
{:else}
  <a
    class="btn btn-primary"
    href={resolve(projectHref('/clusters'))}
    title="Open a cluster and use Assist to ask the VLM about it"
    >Assist with the VLM: pick a cluster</a
  >
{/if}
