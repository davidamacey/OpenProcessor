<script lang="ts">
  /**
   * Which VLM endpoint a test-on-crop call asks (W9 per-run picker in its
   * "test" mode: the no-pick option is the active endpoint). Absent when
   * `/methods` serves no VLM entry to pick. The test route is the gate for
   * an external endpoint's acknowledgement; its refusal shows in the
   * panel's error.
   */
  import VlmRunPicker from '$components/vlm/VlmRunPicker.svelte';
  import { fromTestVlmSelection, toTestVlmSelection } from '$lib/configTest/vlmSelection';
  import type { TestVlmSelection } from '$lib/types_configTest';

  interface Props {
    selection: TestVlmSelection | null;
    disabled?: boolean;
    onchange: (next: TestVlmSelection | null) => void;
  }

  let { selection, disabled = false, onchange }: Props = $props();
  const pick = $derived(fromTestVlmSelection(selection));
</script>

<div class="text-xs" data-testid="test-vlm-picker">
  <VlmRunPicker
    vlm={pick.vlm}
    acknowledgeExternal={pick.acknowledgeExternal}
    {disabled}
    defaultLabel="Active endpoint"
    onchange={(next) => onchange(toTestVlmSelection(next))}
  />
</div>
