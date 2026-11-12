/**
 * The dialogs' state on a config editor page (prompt packs, region profiles,
 * open-vocabulary sets): which document is being activated, whether the restore and
 * clone-to-edit dialogs are open, and what follows a save or clone.
 */
import { goto } from '$app/navigation';
import { resolve } from '$app/paths';
import { toastStore } from '$stores/toast.svelte';
import type { CloneSource } from '$lib/config/configList.svelte';
import type { ConfigEditorView } from '$lib/config/configEditor.svelte';
import type { ConfigDocBase } from '$lib/types_config';

interface EditorLike {
  readonly active: Pick<ConfigEditorView['active'], 'clearAction'>;
  readonly doc: ConfigEditorView['doc'];
  save(): Promise<boolean>;
}

interface ClonerLike {
  clearClone: () => void;
  clone: (
    from: CloneSource,
    name: string,
    description: string,
  ) => Promise<{ name: string } | null>;
}

export function createConfigEditorFlow(opts: {
  /** The editor for the document on screen (it changes with the route). */
  ed: () => EditorLike;
  cloner: ClonerLike;
  name: () => string;
  /** The project-relative editor path of a document, before `resolve`. */
  editorPath: (name: string) => string;
}) {
  let activating = $state<ConfigDocBase<unknown> | null>(null);
  let restoring = $state(false);
  let cloning = $state(false);

  return {
    get activating(): ConfigDocBase<unknown> | null {
      return activating;
    },
    set activating(v: ConfigDocBase<unknown> | null) {
      activating = v;
    },
    get restoring(): boolean {
      return restoring;
    },
    set restoring(v: boolean) {
      restoring = v;
    },
    get cloning(): boolean {
      return cloning;
    },
    set cloning(v: boolean) {
      cloning = v;
    },
    openActivate(target: ConfigDocBase<unknown>): void {
      opts.ed().active.clearAction();
      activating = target;
    },
    startClone(): void {
      opts.cloner.clearClone();
      cloning = true;
    },
    async doSave(): Promise<void> {
      const ed = opts.ed();
      if (await ed.save()) toastStore.success(`Saved revision ${ed.doc?.revision}`);
    },
    async doClone(name: string): Promise<void> {
      const doc = await opts.cloner.clone({ name: opts.name(), source: null }, name, '');
      if (!doc) return;
      cloning = false;
      toastStore.success(`Created ${doc.name}`);
      await goto(resolve(opts.editorPath(doc.name)));
    },
  };
}

export type ConfigEditorFlow = ReturnType<typeof createConfigEditorFlow>;
