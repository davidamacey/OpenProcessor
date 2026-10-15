<script lang="ts">
  /**
   * An "Assist…" button with its own dialog, for the places that ask the
   * VLM about one crop or one box (the crop panel, the review box row).
   * The cluster page owns a controller of its own (`createClusterAssist`).
   */
  import AssistRunDialog from '$components/assistRun/AssistRunDialog.svelte';
  import {
    AssistRunController,
    type AssistOpener,
  } from '$lib/assistRun/assistRunController.svelte';

  interface Props {
    opener: AssistOpener;
    label?: string;
    buttonClass?: string;
    disabled?: boolean;
    /** After the dialog closes, so the page can reload what a run changed. */
    onclosed?: () => void;
  }

  let {
    opener,
    label = 'Assist…',
    buttonClass = 'btn btn-sm',
    disabled = false,
    onclosed,
  }: Props = $props();

  const ctl = new AssistRunController();
</script>

<button
  type="button"
  class={buttonClass}
  {disabled}
  title="Ask the VLM about {opener.title.toLowerCase()}"
  data-testid="assist-button"
  onclick={() => ctl.open(opener)}
>
  {label}
</button>

<AssistRunDialog {ctl} {onclosed} />
