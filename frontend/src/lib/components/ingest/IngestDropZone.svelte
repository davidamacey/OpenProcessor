<script lang="ts">
  /**
   * The `/ingest` drop zone: a drag-drop target plus "Choose files" /
   * "Choose folder" file pickers. Emits the collected `IngestFile[]` —
   * owns no upload logic (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md
   * §A.2).
   */
  import {
    collectFromDrop,
    collectFromInput,
    isAcceptedFile,
  } from '$lib/ingest/fileSource';
  import type { IngestFile } from '$lib/ingest/fileSource';

  interface Props {
    acceptedExtensions: string[];
    onselect: (files: IngestFile[]) => void;
    disabled?: boolean;
  }
  let { acceptedExtensions, onselect, disabled = false }: Props = $props();

  let dragOver = $state(false);
  let filesInput: HTMLInputElement | undefined = $state();
  let folderInput: HTMLInputElement | undefined = $state();

  function filterAccepted(files: IngestFile[]): IngestFile[] {
    return files.filter((f) => isAcceptedFile(f.relPath, acceptedExtensions));
  }

  function onFilesChange(e: Event): void {
    const input = e.currentTarget as HTMLInputElement;
    if (!input.files) return;
    onselect(filterAccepted(collectFromInput(input.files)));
    input.value = '';
  }

  async function onDrop(e: DragEvent): Promise<void> {
    e.preventDefault();
    dragOver = false;
    if (disabled) return;
    const items = e.dataTransfer?.items;
    if (!items) return;
    const collected: IngestFile[] = [];
    for await (const f of collectFromDrop(items)) {
      collected.push(f);
    }
    onselect(filterAccepted(collected));
  }

  function onDragOver(e: DragEvent): void {
    e.preventDefault();
    if (!disabled) dragOver = true;
  }

  function onDragLeave(): void {
    dragOver = false;
  }
</script>

<div
  class="rounded-lg border-2 border-dashed p-8 text-center transition-colors {dragOver
    ? 'border-blue-500 bg-blue-950/20'
    : 'border-zinc-700'} {disabled ? 'opacity-50' : ''}"
  ondrop={onDrop}
  ondragover={onDragOver}
  ondragleave={onDragLeave}
  role="region"
  aria-label="Drop images or a folder here"
>
  <p class="mb-3 text-sm text-zinc-400">Drag files or a folder here</p>
  <div class="flex justify-center gap-2">
    <button
      class="btn btn-sm"
      type="button"
      {disabled}
      onclick={() => filesInput?.click()}
    >
      Choose files
    </button>
    <button
      class="btn btn-sm"
      type="button"
      {disabled}
      onclick={() => folderInput?.click()}
    >
      Choose folder
    </button>
  </div>
  <input
    bind:this={filesInput}
    type="file"
    multiple
    accept={acceptedExtensions.join(',')}
    class="hidden"
    onchange={onFilesChange}
  />
  <input
    bind:this={folderInput}
    type="file"
    multiple
    webkitdirectory
    class="hidden"
    onchange={onFilesChange}
  />
</div>
