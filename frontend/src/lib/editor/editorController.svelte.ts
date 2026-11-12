/**
 * The image editor's state and actions: the served image, the selection, the tool, the class
 * picker for a drawn box, the undo stack and the filmstrip of the list the user came from. It draws
 * nothing and holds no boxes of its own: every write goes through a command and the screen shows the
 * image the server returns.
 */
import { apiErrorText } from '$lib/api_errors';
import { getImage } from '$lib/api_images';
import { resizeXyxy, translateXyxy } from '$lib/annotations/boxGeometry';
import type { XYXY } from '$lib/annotations/types';
import type { EditorTool } from '$components/editor/EditorCanvas.svelte';
import type { BoxTone, Crop, ImageBox, ImageDetailResponse } from '$lib/types';
import type { RegistryClass } from '$lib/types_classes';
import type { ImageLocate } from '$lib/types_editor';
import {
  classCommand,
  CommandStack,
  deleteCommand,
  drawCommand,
  moveCommand,
} from './editorCommands.svelte';

/** How many images after the current one the filmstrip shows. */
const FILMSTRIP_AFTER = 12;

/** A page of the list the editor was opened from; `listImages` returns this shape. */
export interface FilmstripPage {
  items: { image_id: string }[];
}

export interface EditorDeps {
  imageId: string;
  /** Where this image sits in the list the editor was opened from (the Workspace query). */
  locate(signal: AbortSignal): Promise<ImageLocate>;
  /** One page of that list at `cursor`. */
  fetchPage(cursor: string, signal: AbortSignal): Promise<FilmstripPage>;
  /** Open another image, or leave for the Workspace with `null`. */
  navigate(imageId: string | null): void;
}

/** What the class picker is choosing a class for. */
export type PickerTarget =
  { kind: 'draw'; box: XYXY } | { kind: 'class'; cropId: string };

/** The canvas operations the keys drive; the page hands them over once the canvas exists. */
export interface CanvasApi {
  zoomBy(factor: number): void;
  zoomFit(): void;
  zoomActual(): void;
  /** Drops a draw or drag under way; true when there was one. */
  cancelGesture(): boolean;
  /** One screen pixel as a fraction of the picture. */
  pixelStep(): { x: number; y: number };
}

export interface ObjectRow {
  cropId: string;
  label: string;
  tone: BoxTone;
  className: string | null;
  /** The served words for what locks this box; null when it is free. */
  lockLabel: string | null;
  locked: boolean;
  deleted: boolean;
  regionCount: number;
}

export interface Filmstrip {
  before: string | null;
  after: string[];
  /** 0-based position in the list and its size; null when the image is not in it. */
  index: number | null;
  total: number;
  /** The served sentence for an image that is not in the list. */
  message: string | null;
}

const ZOOM_STEP = 1.25;
const FAR = 10;

export function createEditorController(deps: EditorDeps) {
  const abort = new AbortController();

  let image = $state.raw<ImageDetailResponse | null>(null);
  let loadError = $state<string | null>(null);
  let selectedCropId = $state<string | null>(null);
  let tool = $state<EditorTool>('select');
  let sticky = $state(false);
  let labels = $state(true);
  let gesture = $state(false);
  let picker = $state.raw<PickerTarget | null>(null);
  let lastClass = $state<string | null>(null);
  let regionsOpen = $state(false);
  let drop = $state.raw<{ cropId: string; box: XYXY; regionBoxIds: string[] } | null>(
    null,
  );
  let filmstrip = $state.raw<Filmstrip | null>(null);
  let zoomPercent = $state(100);
  let canvas: CanvasApi | null = null;

  const stack = new CommandStack({
    onImage: (next) => {
      image = next;
      if (selectedCropId && !next.items.some((i) => i.id === selectedCropId)) {
        selectedCropId = null;
      }
    },
    onConflict: () => void load(),
  });

  const items = $derived(image?.items ?? []);
  const objects = $derived.by<ObjectRow[]>(() => {
    const boxes = image?.boxes ?? [];
    return boxes
      .filter((b) => b.kind === 'item')
      .map((b) => {
        const item = items.find((i) => i.id === b.crop_id);
        return {
          cropId: b.crop_id,
          label: b.label,
          tone: b.tone,
          className: b.class_name,
          lockLabel: item?.geometry_locked ? (item.lock_label ?? null) : null,
          locked: item?.geometry_locked ?? false,
          deleted: item?.class_excluded ?? false,
          regionCount: boxes.filter((r) => r.kind === 'region' && r.crop_id === b.crop_id)
            .length,
        };
      });
  });
  /** The geometry-locked objects, by id, with the served words for the lock. */
  const locks = $derived<Record<string, string | null>>(
    Object.fromEntries(
      objects.filter((o) => o.locked).map((o) => [o.cropId, o.lockLabel]),
    ),
  );
  // Tab walks the live objects only; a deleted one stays listed until the next reload.
  const live = $derived(objects.filter((o) => !o.deleted));

  function itemFor(cropId: string | null): Crop | null {
    return cropId ? (items.find((i) => i.id === cropId) ?? null) : null;
  }
  function boxFor(cropId: string): ImageBox | null {
    return image?.boxes.find((b) => b.kind === 'item' && b.crop_id === cropId) ?? null;
  }

  async function load(): Promise<void> {
    loadError = null;
    try {
      image = await getImage(deps.imageId, abort.signal);
      void loadFilmstrip();
    } catch (e) {
      if (!abort.signal.aborted) loadError = apiErrorText(e);
    }
  }

  async function loadFilmstrip(): Promise<void> {
    try {
      const where = await deps.locate(abort.signal);
      if (!where.in_view || where.cursor_after === null) {
        filmstrip = {
          before: null,
          after: [],
          index: null,
          total: where.total,
          message: where.reason_message,
        };
        return;
      }
      const page = await deps.fetchPage(where.cursor_after, abort.signal);
      filmstrip = {
        before: where.previous_image_id,
        after: page.items.slice(0, FILMSTRIP_AFTER).map((i) => i.image_id),
        index: where.index,
        total: where.total,
        message: null,
      };
    } catch {
      // The walk is a convenience: a failed locate leaves the editor usable without it.
      if (!abort.signal.aborted) filmstrip = null;
    }
  }

  // -- tools and selection ---------------------------------------------------

  function setTool(next: EditorTool, keepDrawing = false): void {
    tool = next;
    sticky = keepDrawing && next === 'draw';
  }

  function select(cropId: string | null): void {
    selectedCropId = cropId;
  }

  function cycle(step: 1 | -1): void {
    if (live.length === 0) return;
    const at = live.findIndex((o) => o.cropId === selectedCropId);
    const next =
      at === -1
        ? step === 1
          ? 0
          : live.length - 1
        : (at + step + live.length) % live.length;
    selectedCropId = live[next].cropId;
  }

  // -- writes ----------------------------------------------------------------

  /** A finished draw gesture: ask for the class, then write. */
  function draw(box: XYXY): void {
    picker = { kind: 'draw', box };
    if (!sticky) tool = 'select';
  }

  function openClassPicker(): void {
    if (selectedCropId && itemFor(selectedCropId)) {
      picker = { kind: 'class', cropId: selectedCropId };
    }
  }

  function cancelPicker(): void {
    picker = null;
  }

  async function pick(cls: RegistryClass): Promise<void> {
    const target = picker;
    picker = null;
    lastClass = cls.name;
    if (target?.kind === 'draw') {
      const before = items.map((i) => i.id);
      await stack.run(drawCommand(deps.imageId, target.box, cls.name));
      const made = items.find((i) => !before.includes(i.id));
      if (made) selectedCropId = made.id;
    } else if (target?.kind === 'class') {
      const item = itemFor(target.cropId);
      if (item) await stack.run(classCommand(item, cls));
    }
  }

  /** A class hotkey: set the selected object's class. */
  async function setClass(cls: RegistryClass): Promise<void> {
    const item = itemFor(selectedCropId);
    if (!item || item.class_excluded) return;
    lastClass = cls.name;
    await stack.run(classCommand(item, cls));
  }

  async function deleteSelected(): Promise<void> {
    const item = itemFor(selectedCropId);
    if (!item || item.class_excluded) return;
    const at = live.findIndex((o) => o.cropId === item.id);
    await stack.run(deleteCommand(item));
    selectedCropId = live.length > 0 ? live[Math.min(at, live.length - 1)].cropId : null;
  }

  async function move(cropId: string, box: XYXY): Promise<void> {
    const item = itemFor(cropId);
    if (!item || item.geometry_locked || item.class_excluded) return;
    await stack.run(moveCommand(item, box));
    if (stack.regionBoxIds) drop = { cropId, box, regionBoxIds: stack.regionBoxIds };
  }

  async function confirmDrop(): Promise<void> {
    const pending = drop;
    drop = null;
    const item = itemFor(pending?.cropId ?? null);
    if (pending && item) {
      await stack.run(moveCommand(item, pending.box, pending.regionBoxIds));
    }
  }

  function dismissDrop(): void {
    drop = null;
  }

  /** Moves the selected box by whole screen pixels. */
  function nudge(dx: number, dy: number): Promise<void> {
    const box = selectedCropId ? boxFor(selectedCropId) : null;
    if (!box || !canvas) return Promise.resolve();
    const step = canvas.pixelStep();
    return move(box.crop_id, translateXyxy(box.bbox_norm, dx * step.x, dy * step.y));
  }

  /** Moves the bottom-right corner of the selected box by whole screen pixels. */
  function resizeCorner(dx: number, dy: number): Promise<void> {
    const box = selectedCropId ? boxFor(selectedCropId) : null;
    if (!box || !canvas) return Promise.resolve();
    const step = canvas.pixelStep();
    return move(box.crop_id, resizeXyxy(box.bbox_norm, 'se', dx * step.x, dy * step.y));
  }

  const undo = () => stack.undo();
  const redo = () => stack.redo();

  // -- walking the list ------------------------------------------------------

  function nextImage(): void {
    const id = filmstrip?.after[0];
    if (id) deps.navigate(id);
  }
  function prevImage(): void {
    if (filmstrip?.before) deps.navigate(filmstrip.before);
  }

  /** Escape: the draw under way, then the picker, then the selection, then back to the Workspace. */
  function escape(): void {
    if (canvas?.cancelGesture()) return;
    if (picker) picker = null;
    else if (drop) drop = null;
    else if (regionsOpen) regionsOpen = false;
    else if (selectedCropId) selectedCropId = null;
    else deps.navigate(null);
  }

  void load();

  return {
    get image() {
      return image;
    },
    /** The boxes the server served for this image, exactly. */
    get boxes(): ImageBox[] {
      return image?.boxes ?? [];
    },
    get objects() {
      return objects;
    },
    get locks() {
      return locks;
    },
    get loadError() {
      return loadError;
    },
    get selectedCropId() {
      return selectedCropId;
    },
    get selected(): Crop | null {
      return itemFor(selectedCropId);
    },
    get tool() {
      return tool;
    },
    get sticky() {
      return sticky;
    },
    get labels() {
      return labels;
    },
    get gesture() {
      return gesture;
    },
    set gesture(v: boolean) {
      gesture = v;
    },
    get picker() {
      return picker;
    },
    /** The class used last in this session, ranked first in the picker. */
    get recentClasses(): string[] {
      return lastClass ? [lastClass] : [];
    },
    get regionsOpen() {
      return regionsOpen;
    },
    get drop() {
      return drop;
    },
    get filmstrip() {
      return filmstrip;
    },
    get zoomPercent() {
      return zoomPercent;
    },
    set zoomPercent(v: number) {
      zoomPercent = v;
    },
    stack,
    attachCanvas(api: CanvasApi | null) {
      canvas = api;
    },
    reload: load,
    setTool,
    select,
    selectNext: () => cycle(1),
    selectPrev: () => cycle(-1),
    toggleLabels() {
      labels = !labels;
    },
    editRegions() {
      if (selectedCropId) regionsOpen = !regionsOpen;
    },
    closeRegions() {
      regionsOpen = false;
    },
    draw,
    openClassPicker,
    cancelPicker,
    pick,
    setClass,
    deleteSelected,
    move,
    confirmDrop,
    dismissDrop,
    nudge,
    nudgeFar: (dx: number, dy: number) => nudge(dx * FAR, dy * FAR),
    resizeCorner,
    undo,
    redo,
    zoomIn: () => canvas?.zoomBy(ZOOM_STEP),
    zoomOut: () => canvas?.zoomBy(1 / ZOOM_STEP),
    zoomFit: () => canvas?.zoomFit(),
    zoomActual: () => canvas?.zoomActual(),
    nextImage,
    prevImage,
    escape,
    dispose() {
      abort.abort();
    },
  };
}

export type EditorController = ReturnType<typeof createEditorController>;
