<script lang="ts">
  /**
   * "New": a name, a description and the body as JSON. The body is the server's to judge: a refusal
   * comes back with one reason per field and stays in the dialog.
   */
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import { parseBodyText } from '$lib/presets/presetBodyText';
  import type { PresetList } from '$lib/presets/presetListController.svelte';
  import type { PresetBody } from '$lib/types_presets';

  interface Props {
    list: PresetList;
    oncreate: (name: string, description: string, body: PresetBody) => void;
    oncancel: () => void;
  }

  let { list, oncreate, oncancel }: Props = $props();

  let name = $state('');
  let description = $state('');
  let text = $state('{}');

  const parsed = $derived(parseBodyText(text));
</script>

<ConfirmDialog
  title="New {list.kind.noun}"
  confirmLabel="Create"
  busy={list.busy}
  confirmDisabled={name.trim() === '' || 'error' in parsed}
  onconfirm={() => 'body' in parsed && oncreate(name, description, parsed.body)}
  {oncancel}
>
  <label class="flex flex-col gap-1 text-xs text-zinc-400">
    Name
    <input
      class="input input-sm font-mono"
      bind:value={name}
      data-testid="preset-new-name"
    />
  </label>
  <label class="flex flex-col gap-1 text-xs text-zinc-400">
    Description (optional)
    <input class="input input-sm" bind:value={description} />
  </label>
  <label class="flex flex-col gap-1 text-xs text-zinc-400">
    Body (JSON)
    <textarea
      class="input min-h-40 font-mono text-xs"
      spellcheck="false"
      bind:value={text}
      data-testid="preset-new-body"></textarea>
  </label>
  {#if 'error' in parsed}
    <p class="text-xs text-amber-300" data-testid="preset-new-json-error">
      {parsed.error}
    </p>
  {/if}
  <p class="text-xs text-zinc-500">
    The server checks the body when you create it and names each field it refuses. To
    start from a document that exists, clone it from the list.
  </p>
  {#if list.createError}
    <p class="text-red-300" data-testid="preset-create-error">{list.createError}</p>
  {/if}
</ConfirmDialog>
