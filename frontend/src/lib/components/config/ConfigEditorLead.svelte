<script lang="ts">
  /** The top of a config editor form: the banner for a revision being viewed, and, on a
   *  read-only document, the way to clone it into an editable one. */
  import ConfigViewingBanner from '$components/config/ConfigViewingBanner.svelte';
  import type { ConfigEditorView } from '$lib/config/configEditor.svelte';
  import type { ConfigEditorFlow } from '$lib/config/editorFlow.svelte';

  interface Props {
    ed: ConfigEditorView;
    flow: ConfigEditorFlow;
    /** "pack", "profile", "set". */
    noun: string;
  }

  let { ed, flow, noun }: Props = $props();
</script>

<ConfigViewingBanner
  {ed}
  onrestore={() => (flow.restoring = true)}
  onactivate={() => ed.viewing && flow.openActivate(ed.viewing)}
/>

{#if ed.doc?.read_only}
  <div class="flex flex-wrap items-center gap-2 text-body text-fg-2">
    <span>This {noun} is read-only. Clone it to make an editable copy.</span>
    <button type="button" class="btn btn-sm" onclick={() => flow.startClone()}
      >Clone to edit</button
    >
  </div>
{/if}
