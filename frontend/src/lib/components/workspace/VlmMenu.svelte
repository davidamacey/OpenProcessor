<script lang="ts">
  /**
   * The one place AI work starts from in the Workspace: Auto-label (embed and
   * finalize, which needs no VLM), and, while the backend serves an active VLM
   * (`GET /vlm/endpoints/active`, `source !== 'off'`), Assist on the selection
   * or the scope and the auto-label loop with its status. Without a VLM, or
   * when the answer is unreadable, those two are disabled and point to Models
   * settings, as buttons that could only fail.
   */
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import AssistLoopPanel from '$components/assistLoop/AssistLoopPanel.svelte';
  import AssistRunDialog from '$components/assistRun/AssistRunDialog.svelte';
  import AutoLabelPanel from '$components/AutoLabelPanel.svelte';
  import { getActiveVlm } from '$lib/api_vlm';
  import { AssistRunController } from '$lib/assistRun/assistRunController.svelte';
  import { AssistLoopController } from '$lib/assistLoop/assistLoopController.svelte';
  import { projectHref } from '$lib/projectPaths';
  import { vlmAssistOpener } from '$lib/workspace/vlmAssistOpener.svelte';
  import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';

  let { ws }: { ws: WorkspaceController } = $props();

  let active = $state(false);
  let open = $state(false);
  const assist = new AssistRunController();
  const loop = new AssistLoopController();

  onMount(() => {
    const ctl = new AbortController();
    getActiveVlm(ctl.signal)
      .then((r) => {
        active = r.source !== 'off';
      })
      .catch(() => {});
    return () => ctl.abort();
  });

  const opener = $derived(
    vlmAssistOpener(ws.selection, ws.itemSelection(), ws.itemBody()),
  );
</script>

<div class="relative" data-testid="vlm-menu">
  <button
    type="button"
    class="btn"
    aria-expanded={open}
    aria-controls="vlm-menu-body"
    data-testid="vlm-menu-toggle"
    onclick={() => (open = !open)}
  >
    AI
    {#if active && loop.loop}
      <span class="chip font-normal text-fg-2" data-testid="vlm-menu-loop-status"
        >{loop.loop.status}</span
      >
    {/if}
  </button>
  <div
    id="vlm-menu-body"
    class="fixed inset-x-2 top-24 z-30 max-h-[80vh] overflow-auto rounded-md border border-line-strong bg-surface-0 shadow-xl md:absolute md:inset-x-auto md:right-0 md:top-full md:mt-1 md:w-[40rem]"
    hidden={!open}
    data-testid="vlm-menu-body"
  >
    <div class="flex flex-wrap items-center gap-2 border-b border-line px-4 py-2">
      <button
        type="button"
        class="btn btn-primary"
        data-testid="vlm-menu-assist"
        disabled={!active}
        onclick={() => assist.open(opener)}
      >
        Assist on {opener.title.toLowerCase()}…
      </button>
      {#if !active}
        <span class="text-xs text-fg-2" data-testid="vlm-menu-off"
          >Assist and the loop need an active VLM.</span
        >
      {/if}
      <span class="grow"></span>
      <a
        class="text-xs text-fg-2 underline"
        data-testid="vlm-menu-models"
        href={resolve(projectHref('/settings/models'))}
        >{active ? 'Models settings' : 'Activate a VLM in Models settings'}</a
      >
    </div>
    {#if active}<AssistLoopPanel ctl={loop} />{/if}
    {#if open}
      <div class="p-3"><AutoLabelPanel /></div>
    {/if}
  </div>
  <AssistRunDialog ctl={assist} />
</div>
