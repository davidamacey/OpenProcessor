/**
 * The Clone and Delete dialogs' state on a config list page (prompt packs, region profiles,
 * open-vocabulary sets): which document is being cloned or deleted, and what happens after
 * the server accepts: a toast, and for a clone, the editor of the new document.
 */
import { goto } from '$app/navigation';
import { resolve } from '$app/paths';
import { toastStore } from '$stores/toast.svelte';
import type { CloneSource } from '$lib/config/configList.svelte';

interface FlowList<Row> {
  clearClone: () => void;
  clone: (
    from: CloneSource,
    name: string,
    description: string,
    fromProject: string | null,
  ) => Promise<{ name: string } | null>;
  remove: (row: Row) => Promise<boolean>;
  deleteError: string | null;
}

export function createCloneDeleteFlow<Row extends { name: string }>(
  list: FlowList<Row>,
  /** The project-relative editor path of a document, before `resolve`. */
  editorPath: (name: string) => string,
) {
  let cloneFrom = $state<CloneSource | null>(null);
  let deleting = $state<Row | null>(null);

  return {
    get cloneFrom(): CloneSource | null {
      return cloneFrom;
    },
    set cloneFrom(v: CloneSource | null) {
      cloneFrom = v;
    },
    get deleting(): Row | null {
      return deleting;
    },
    set deleting(v: Row | null) {
      deleting = v;
    },
    openClone(from: CloneSource): void {
      list.clearClone();
      cloneFrom = from;
    },
    askDelete(row: Row): void {
      list.deleteError = null;
      deleting = row;
    },
    async doClone(
      name: string,
      description: string,
      fromProject: string | null,
    ): Promise<void> {
      if (!cloneFrom) return;
      const doc = await list.clone(cloneFrom, name, description, fromProject);
      if (!doc) return;
      cloneFrom = null;
      toastStore.success(`Created ${doc.name}`);
      await goto(resolve(editorPath(doc.name)));
    },
    async doDelete(): Promise<void> {
      const row = deleting;
      if (!row) return;
      if (await list.remove(row)) {
        deleting = null;
        toastStore.success(`Deleted ${row.name}`);
      }
    },
  };
}

export type CloneDeleteFlow<Row extends { name: string }> = ReturnType<
  typeof createCloneDeleteFlow<Row>
>;
