<script lang="ts">
  /**
   * The one place VLM work starts from in the Workspace: Assist on the
   * selection or the scope, the auto-label loop and its status, and the
   * Auto-label run with its settings. It exists only while the backend serves
   * an active VLM (`GET /vlm/endpoints/active`, `source !== 'off'`); an
   * unreadable answer leaves it out, as a button that could only fail.
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

{#if active}
  <div class="relative" data-testid="vlm-menu">
    <button
      type="button"
      class="btn"
      aria-expanded={open}
      aria-controls="vlm-menu-body"
      data-testid="vlm-menu-toggle"
      onclick={() => (open = !open)}
    >
      VLM
      {#if loop.loop}
        <span class="chip font-normal text-zinc-400" data-testid="vlm-menu-loop-status"
          >{loop.loop.status}</span
        >
      {/if}
    </button>
    <div
      id="vlm-menu-body"
      class="fixed inset-x-2 top-24 z-30 max-h-[80vh] overflow-auto rounded-md border border-zinc-700 bg-zinc-950 shadow-xl md:absolute md:inset-x-auto md:right-0 md:top-full md:mt-1 md:w-[40rem]"
      hidden={!open}
      data-testid="vlm-menu-body"
    >
      <div class="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-4 py-2">
        <button
          type="button"
          class="btn btn-primary"
          data-testid="vlm-menu-assist"
          onclick={() => assist.open(opener)}
        >
          Assist on {opener.title.toLowerCase()}…
        </button>
        <span class="grow"></span>
        <a
          class="text-xs text-zinc-400 underline"
          data-testid="vlm-menu-models"
          href={resolve(projectHref('/settings/models'))}>Models settings</a
        >
      </div>
      <AssistLoopPanel ctl={loop} />
      {#if open}
        <div class="p-3"><AutoLabelPanel /></div>
      {/if}
    </div>
    <AssistRunDialog ctl={assist} />
  </div>
{/if}
