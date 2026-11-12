<script lang="ts">
  /** The description box of a config editor: the viewed revision's, else the draft's. */
  import type { ConfigEditorView } from '$lib/config/configEditor.svelte';

  interface Props {
    ed: Pick<ConfigEditorView, 'viewing' | 'editable'> & {
      readonly draftDescription: string;
      setDescription(value: string): void;
    };
  }

  let { ed }: Props = $props();
</script>

<label class="flex flex-col gap-1 text-xs text-zinc-400">
  Description
  <input
    class="input input-sm text-sm"
    value={ed.viewing ? (ed.viewing.description ?? '') : ed.draftDescription}
    readonly={!ed.editable}
    oninput={(e) => ed.setDescription((e.currentTarget as HTMLInputElement).value)}
  />
</label>
