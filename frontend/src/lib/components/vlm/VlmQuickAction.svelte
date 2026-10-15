<script lang="ts">
  /**
   * The dashboard's "Run VLM Labeling" action. Whether the project has a VLM
   * is the backend's call (`GET /vlm/endpoints/active`, `source: 'off'`): with
   * none, the action is a link to the Models settings instead of a ready
   * primary button that could only fail. An unreadable answer leaves the
   * action as it is; the run itself reports its refusal.
   */
  import { resolve } from '$app/paths';
  import { onMount } from 'svelte';
  import { getActiveVlm } from '$lib/api_vlm';
  import { projectHref } from '$lib/projectPaths';

  let { onrun }: { onrun: () => void } = $props();

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
    >Run VLM Labeling: activate a VLM first</a
  >
{:else}
  <button class="btn btn-primary" type="button" onclick={onrun}>Run VLM Labeling</button>
{/if}
