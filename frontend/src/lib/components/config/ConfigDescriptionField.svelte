<script lang="ts">
  /** The description box of a config editor: the viewed revision's, else the draft's. */
  interface Props {
    ed: {
      readonly viewing: { readonly description: string | null } | null;
      readonly editable: boolean;
      readonly draftDescription: string;
      setDescription(value: string): void;
    };
  }

  let { ed }: Props = $props();
</script>

<label class="flex flex-col gap-1 text-xs text-fg-2">
  Description
  <input
    class="input input-sm text-sm"
    value={ed.viewing ? (ed.viewing.description ?? '') : ed.draftDescription}
    readonly={!ed.editable}
    oninput={(e) => ed.setDescription((e.currentTarget as HTMLInputElement).value)}
  />
</label>
