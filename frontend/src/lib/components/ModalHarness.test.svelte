<script lang="ts">
  import Modal from './Modal.svelte';

  interface Props {
    onclose?: () => void;
    align?: 'center' | 'top';
  }
  let { onclose, align }: Props = $props();
</script>

<Modal label="Harness dialog" {onclose} {align} data-testid="harness">
  <button type="button" id="first">First</button>
  <input id="field" aria-label="field" />
  <button type="button" id="last">Last</button>
</Modal>
