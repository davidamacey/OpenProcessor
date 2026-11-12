/**
 * The image editor's write commands and the session stack that undoes and redoes them. Each
 * command knows its forward write and its inverse; a successful write returns the served image
 * and the stack hands it to the page, so the screen always shows what the server holds. A 409
 * drops the entry and asks the page to reload instead of guessing.
 */
import {
  excludeCrops,
  patchItemGeometry,
  putCropLabel,
  undoCropLabel,
  undoCropRegion,
  undoItemGeometry,
  unexcludeCrops,
} from '$lib/api_cropWrites';
import { apiErrorText, ApiError } from '$lib/api_errors';
import { createImageItem, getImage } from '$lib/api_images';
import { putRegionBoxes } from '$lib/api_regionBoxes';
import type { RegionBoxInput } from '$lib/annotations/multiBox';
import type { XYXY } from '$lib/annotations/types';
import type { Crop, ImageDetailResponse } from '$lib/types';
import type { RegistryClass } from '$lib/types_classes';
import { newUuid } from '$lib/uuid';

export type CommandResult =
  | { ok: true; image: ImageDetailResponse }
  | {
      ok: false;
      message: string;
      /** The box or label changed since the editor read it. */
      conflict: boolean;
      /** Region boxes a move would leave outside the item: confirm to delete them with it. */
      regionBoxIds?: string[];
    };

export interface EditorCommand {
  /** Shown in the undo tooltip, e.g. `Draw car`. */
  label: string;
  apply(): Promise<CommandResult>;
  revert(): Promise<CommandResult>;
}

function failure(e: unknown): CommandResult {
  const message = apiErrorText(e);
  if (e instanceof ApiError) {
    const detail = (e.body as { detail?: { error?: string; box_ids?: string[] } } | null)
      ?.detail;
    if (detail?.error === 'regions_outside_item') {
      return {
        ok: false,
        message,
        conflict: false,
        regionBoxIds: detail.box_ids ?? [],
      };
    }
    return { ok: false, message, conflict: e.status === 409 };
  }
  return { ok: false, message, conflict: false };
}

/** Runs `write`, then reads the image back so the screen shows what the server holds. */
async function settle(
  imageId: string,
  write: () => Promise<void>,
): Promise<CommandResult> {
  try {
    await write();
    return { ok: true, image: await getImage(imageId) };
  } catch (e) {
    return failure(e);
  }
}

/** A box drawn on the picture. Redo restores the item it made rather than drawing a second. */
export function drawCommand(
  imageId: string,
  box: XYXY,
  className: string,
): EditorCommand {
  const requestId = newUuid();
  let created: Crop | null = null;
  return {
    label: `Draw ${className}`,
    apply: () =>
      settle(imageId, async () => {
        if (created) {
          await unexcludeCrops([created.id]);
          return;
        }
        const res = await createImageItem(imageId, {
          bbox_norm: box,
          class_name: className,
          request_id: requestId,
        });
        created = res.item;
      }),
    revert: () =>
      settle(imageId, async () => {
        if (created) await excludeCrops([created.id], 'not_an_object');
      }),
  };
}

/**
 * A move or resize. It holds the item's geometry revision and takes the one each response
 * serves, so a redo after an undo sends the current revision, not the one it started with.
 */
export function moveCommand(
  item: Crop,
  box: XYXY,
  dropRegionBoxIds: string[] = [],
): EditorCommand {
  let revision = item.geometry_revision;
  const imageId = item.image_id ?? '';
  return {
    label: 'Move box',
    apply: () =>
      settle(imageId, async () => {
        const res = await patchItemGeometry(item.id, {
          bbox_norm: box,
          expected_geometry_revision: revision,
          ...(dropRegionBoxIds.length ? { drop_region_box_ids: dropRegionBoxIds } : {}),
        });
        revision = res.item.geometry_revision;
      }),
    revert: () =>
      settle(imageId, async () => {
        const res = await undoItemGeometry(item.id, revision);
        revision = res.item.geometry_revision;
      }),
  };
}

/** A soft delete: the item is excluded as not an object, and brought back by undo. */
export function deleteCommand(item: Crop): EditorCommand {
  const imageId = item.image_id ?? '';
  return {
    label: 'Delete box',
    apply: () =>
      settle(imageId, async () => {
        await excludeCrops([item.id], 'not_an_object');
      }),
    revert: () =>
      settle(imageId, async () => {
        await unexcludeCrops([item.id]);
      }),
  };
}

/** A class change, sent as the registry entry's id and undone by the server's label undo. */
export function classCommand(item: Crop, cls: RegistryClass): EditorCommand {
  const imageId = item.image_id ?? '';
  return {
    label: `Set class ${cls.name}`,
    apply: () =>
      settle(imageId, async () => {
        await putCropLabel(item.id, cls.id);
      }),
    revert: () =>
      settle(imageId, async () => {
        await undoCropLabel(item.id);
      }),
  };
}

/** A replacement of one item's region boxes. */
export function regionCommand(item: Crop, boxes: RegionBoxInput[]): EditorCommand {
  const imageId = item.image_id ?? '';
  return {
    label: 'Edit region boxes',
    apply: () =>
      settle(imageId, async () => {
        await putRegionBoxes(item.id, boxes);
      }),
    revert: () =>
      settle(imageId, async () => {
        await undoCropRegion(item.id);
      }),
  };
}

export type SaveStatus = 'saved' | 'saving' | 'conflict' | 'failed';

export interface StackHooks {
  /** A write succeeded: draw this served image. */
  onImage(image: ImageDetailResponse): void;
  /** A write found the item changed: reload it. */
  onConflict(): void;
}

/** The undo and redo history of one image's editing session, one write at a time. */
export class CommandStack {
  status = $state<SaveStatus>('saved');
  /** The served words for the last failure. */
  message = $state<string | null>(null);
  #done = $state.raw<EditorCommand[]>([]);
  #undone = $state.raw<EditorCommand[]>([]);
  #pending = $state(0);
  #tail: Promise<void> = Promise.resolve();
  readonly #hooks: StackHooks;

  constructor(hooks: StackHooks) {
    this.#hooks = hooks;
  }

  get canUndo(): boolean {
    return this.#done.length > 0;
  }
  get canRedo(): boolean {
    return this.#undone.length > 0;
  }
  get saving(): boolean {
    return this.#pending > 0;
  }
  get undoLabel(): string | null {
    return this.#done.at(-1)?.label ?? null;
  }
  get redoLabel(): string | null {
    return this.#undone.at(-1)?.label ?? null;
  }

  run(cmd: EditorCommand): Promise<void> {
    return this.#enqueue(async () => {
      const r = await cmd.apply();
      if (r.ok) {
        this.#done = [...this.#done, cmd];
        this.#undone = [];
      }
      this.#settle(r);
    });
  }

  undo(): Promise<void> {
    return this.#enqueue(async () => {
      const cmd = this.#done.at(-1);
      if (!cmd) return;
      const r = await cmd.revert();
      if (r.ok) this.#undone = [...this.#undone, cmd];
      if (r.ok || r.conflict) this.#done = this.#done.slice(0, -1);
      this.#settle(r);
    });
  }

  redo(): Promise<void> {
    return this.#enqueue(async () => {
      const cmd = this.#undone.at(-1);
      if (!cmd) return;
      const r = await cmd.apply();
      if (r.ok) this.#done = [...this.#done, cmd];
      if (r.ok || r.conflict) this.#undone = this.#undone.slice(0, -1);
      this.#settle(r);
    });
  }

  #enqueue(job: () => Promise<void>): Promise<void> {
    this.#pending += 1;
    this.status = 'saving';
    const next = this.#tail.then(job).finally(() => {
      this.#pending -= 1;
    });
    this.#tail = next.catch(() => {});
    return next;
  }

  #settle(r: CommandResult): void {
    if (r.ok) {
      this.message = null;
      this.status = this.#pending > 1 ? 'saving' : 'saved';
      this.#hooks.onImage(r.image);
      return;
    }
    this.message = r.message;
    this.status = r.conflict ? 'conflict' : 'failed';
    if (r.conflict) this.#hooks.onConflict();
  }
}
