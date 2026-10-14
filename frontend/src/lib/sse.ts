/**
 * SSE helper — wraps EventSource with reconnect-with-backoff and
 * topic / class filtering (Task #92).
 *
 * The labeler /review and /clusters pages call `subscribeCurationEvents()` on
 * mount and prepend incoming `crop.*` events into their lists. When the
 * connection drops (network blip, proxy timeout, page sleep) we retry
 * with a capped exponential backoff so a brief disconnect doesn't leave
 * the page stale.
 *
 * The backend endpoint is `GET {API_PREFIX}/events?topic=...&class_id=...` and
 * emits `text/event-stream` with `event:` lines naming the event type.
 * We surface every typed event back to the caller via a single
 * `onEvent` callback so the page can switch on `event.type`.
 */

import { apiBase, globalApi, scoped } from '$lib/api';
import { slotRegistry } from './annotations/registeredSlots';

// All event payloads share these fields; specific types add more.
export interface CurationBaseEvent {
  type: string;
  ts?: number;
  topic?: string;
  crop_id?: string;
}

export interface CropCreatedEvent extends CurationBaseEvent {
  type: 'crop.created';
  crop_id: string;
  image_path?: string;
}

export interface CropClassifiedEvent extends CurationBaseEvent {
  type: 'crop.classified';
  crop_id: string;
  class_id?: number | null;
  class_name?: string | null;
  class_source?: string;
}

/**
 * A slot's "human verified this box" event. `type` is any
 * `crop.<slot.key>_verified`
 * string (or the generic `'crop.region_verified'` OpenProcessor emits for
 * every region — see `slotVerifiedEventTypes()`), and
 * the slot-specific fields are carried untyped so a handler reads them
 * off the active slot's own wire field names (`capabilities.lifecycle
 * .statusField` / `capabilities.text.valueField`) rather than a
 * hardcoded `region_status`/`region_text` pair.
 */
export interface CropSlotVerifiedEvent extends CurationBaseEvent {
  type: string;
  crop_id: string;
  [wireField: string]: unknown;
}

/**
 * `config.changed` (K2, docs/design/configurable-keyboard-shortcuts-
 * plan-2026-09-26.md §5.1/§4.4) — fired on every successful
 * `PUT`/`POST /keymap*` write. `axis === 'keymap'` is the only one this
 * build reacts to today; other axes are ignored by the handler, not by
 * this type.
 */
export interface ConfigChangedEvent extends CurationBaseEvent {
  type: 'config.changed';
  axis: string;
  name?: string | null;
  config_revision?: number;
  keymap_revision?: number;
  project?: string | null;
}

export type CurationEvent =
  | CropCreatedEvent
  | CropClassifiedEvent
  | CropSlotVerifiedEvent
  | ConfigChangedEvent
  | CurationBaseEvent;

export interface CurationEventSubscribeOptions {
  topic?: string;
  class_id?: number;
  onEvent: (event: CurationEvent) => void;
  onError?: (err: Event | Error) => void;
  onOpen?: () => void;
}

export interface CurationEventSubscription {
  /** Close the EventSource and stop reconnecting. */
  close(): void;
}

const RECONNECT_INITIAL_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;

/** Absolute URL of an SSE endpoint. `apiBase` is empty in the default Docker
 *  deployment (nginx proxies `{API_PREFIX}/*` on the same origin) and
 *  `new URL('/path')` throws without a base, so a relative base is anchored to
 *  `window.location.origin`; the result is still same-origin. */
function streamUrl(path: string): URL {
  const origin =
    apiBase && /^https?:\/\//i.test(apiBase)
      ? ''
      : typeof window !== 'undefined'
        ? window.location.origin
        : '';
  return new URL(`${origin}${apiBase}${path}`);
}

interface StreamOptions {
  url: string;
  /** Event names to listen for; read again on every (re)connect. */
  eventTypes: () => string[];
  /** A frame whose `data` parsed as JSON. */
  onFrame: (type: string, payload: unknown) => void;
  onError?: (err: Event | Error) => void;
  onOpen?: () => void;
}

/**
 * Open an EventSource and keep it open: on any error close it and reconnect
 * after a capped exponential backoff (the browser retries on its own, but
 * Firefox gives up after a few rapid retries). A successful open resets the
 * backoff so one old blip does not slow a later reconnect. A frame that is
 * not valid JSON is logged and skipped; it never closes the channel. The
 * caller must `close()` on unmount, otherwise it reconnects forever.
 */
function openStream(opts: StreamOptions): CurationEventSubscription {
  let es: EventSource | null = null;
  let backoff = RECONNECT_INITIAL_MS;
  let closed = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  function open(): void {
    if (closed) return;
    es = new EventSource(opts.url);
    es.onopen = () => {
      backoff = RECONNECT_INITIAL_MS;
      opts.onOpen?.();
    };
    // The backend names each frame with an `event:` line, so EventSource
    // dispatches typed events and `message` never fires.
    for (const type of opts.eventTypes()) {
      es.addEventListener(type, (ev: MessageEvent) => {
        try {
          opts.onFrame(type, JSON.parse(ev.data));
        } catch (err) {
          console.warn('[sse] failed to parse event', type, err);
        }
      });
    }
    es.onerror = (err) => {
      opts.onError?.(err);
      if (closed) return;
      es?.close();
      es = null;
      reconnectTimer = setTimeout(() => {
        backoff = Math.min(backoff * 2, RECONNECT_MAX_MS);
        open();
      }, backoff);
    };
  }

  open();

  return {
    close(): void {
      closed = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      es?.close();
      es = null;
    },
  };
}

/**
 * Event types we care about. Everything else is silently ignored so a
 * future event type added on the backend doesn't trip up older clients
 * — but that same "ignore the unknown" behavior is a live bug for a
 * SECOND queue-capable slot: `EventSource.addEventListener` requires an
 * exact type name, so a slot whose verify event isn't in this list never
 * refreshes the queue, with no error surfaced anywhere (docs/design/
 * slot-generic-crop-mapping-plan-2026-09-21.md §7.1, C7). Reads
 * `slotRegistry` at CALL time (mirrors `mapCropSlots`'s own trap note)
 * so a slot installed after this module's first evaluation is covered.
 */
export function slotVerifiedEventTypes(): string[] {
  const derived = slotRegistry.queues.map((s) => `crop.${s.key}_verified`);
  // 'crop.region_verified' is the one generic event OpenProcessor emits
  // for every region verify (the region worker's HTTP event), whatever
  // the slot key — the per-slot names above cover a backend that
  // namespaces verify events per slot.
  return Array.from(new Set(['crop.region_verified', ...derived]));
}

function knownEventTypes(): string[] {
  return [
    'crop.created',
    'crop.classified',
    'config.changed',
    // advisory import progress; the job view re-reads the job.
    'dataset_import.progress',
    'dataset_import.finished',
    ...slotVerifiedEventTypes(),
  ];
}

/**
 * Open an SSE subscription to {API_PREFIX}/events (project-scoped). The caller
 * must `close()` the subscription on component unmount.
 */
export function subscribeCurationEvents(
  opts: CurationEventSubscribeOptions,
): CurationEventSubscription {
  const url = streamUrl(`${scoped()}/events`);
  if (opts.topic) url.searchParams.set('topic', opts.topic);
  if (opts.class_id != null) url.searchParams.set('class_id', String(opts.class_id));
  return openStream({
    url: url.toString(),
    eventTypes: knownEventTypes,
    onFrame: (_type, payload) => opts.onEvent(payload as CurationEvent),
    onError: opts.onError,
    onOpen: opts.onOpen,
  });
}

// ---------------------------------------------------------------------------
// Pipeline events SSE — {API_PREFIX}/pipeline/events
// ---------------------------------------------------------------------------
// The auto_label pipeline pushes three event types over this channel:
//   * `snapshot` — initial frame on connect with `{ state, stats }`.
//   * `state`    — pipeline state.json was rewritten (stage transition,
//                  progress advance, terminal status).
//   * `stats`    — dataset rollup was refreshed (only at stage boundaries
//                  or terminal status changes, not on every progress tick).
// Heartbeat `: keepalive` frames are emitted every 15s by the backend so
// proxies don't reap the connection. The browser EventSource silently
// drops comment lines, so callers don't see them.
//
// This replaces the 10s polling loop in DatasetStats.svelte — the
// dashboard now updates on push without burning CPU when nothing is
// happening on the pipeline.

export interface PipelineSnapshotEvent {
  type: 'snapshot';
  state: Record<string, unknown>;
  stats: Record<string, unknown>;
}

export interface PipelineStateEvent {
  type: 'state';
  state: Record<string, unknown>;
}

export interface PipelineStatsEvent {
  type: 'stats';
  stats: Record<string, unknown>;
}

export interface PipelineSubscribeOptions {
  onSnapshot?: (state: Record<string, unknown>, stats: Record<string, unknown>) => void;
  onState?: (state: Record<string, unknown>) => void;
  onStats?: (stats: Record<string, unknown>) => void;
  onError?: (err: Event | Error) => void;
  onOpen?: () => void;
}

/**
 * Open an SSE subscription to {API_PREFIX}/pipeline/events.
 *
 * The browser EventSource reconnects automatically (default 3s). On
 * top of that we add capped exponential backoff because Firefox is
 * known to give up after a few rapid retries.
 */
export function subscribePipelineEvents(
  opts: PipelineSubscribeOptions,
): CurationEventSubscription {
  return openStream({
    url: streamUrl(`${scoped()}/pipeline/events`).toString(),
    eventTypes: () => ['snapshot', 'state', 'stats'],
    onFrame: (type, payload) => {
      if (type === 'snapshot') {
        const p = payload as {
          state: Record<string, unknown>;
          stats: Record<string, unknown>;
        };
        opts.onSnapshot?.(p.state ?? {}, p.stats ?? {});
      } else if (type === 'state') {
        opts.onState?.(payload as Record<string, unknown>);
      } else {
        opts.onStats?.(payload as Record<string, unknown>);
      }
    },
    onError: opts.onError,
    onOpen: opts.onOpen,
  });
}

/**
 * Global project-lifecycle event, `GET {globalApi()}/events` (P1
 * projects cutover) — always `project: null`/`topic: 'project'`; a
 * scoped item/pipeline event never appears on this stream (those come
 * from `{scoped()}/events`). Fields beyond the envelope are carried
 * untyped since this build only reacts to "something about the project
 * list changed" (`projectsStore.load()` re-read), not per-event detail.
 */
export interface ProjectEvent {
  type: string;
  topic: 'project';
  project: null;
  target?: string;
  ts?: number;
  [field: string]: unknown;
}

export interface GlobalEventSubscribeOptions {
  onEvent: (event: ProjectEvent) => void;
  onError?: (err: Event | Error) => void;
  onOpen?: () => void;
}

const GLOBAL_EVENT_TYPES = [
  'project.created',
  'project.updated',
  'project.archived',
  'project.unarchived',
  'project.deleted',
  'project.paused',
  'project.resumed',
  'combine.progress',
  // the VLM registry / local-model catalog changed (`axis`:
  // `registry` | `local_vlm`); the models page re-reads on it.
  'vlm.changed',
];

/**
 * Open an SSE subscription to the GLOBAL `{globalApi()}/events` —
 * `project.*` events with no bound project. Held by `projectsStore` to
 * refresh the project list; distinct from `subscribeCurationEvents`/
 * `subscribePipelineEvents`, which are project-scoped.
 */
export function subscribeGlobalEvents(
  opts: GlobalEventSubscribeOptions,
): CurationEventSubscription {
  return openStream({
    url: streamUrl(`${globalApi()}/events`).toString(),
    eventTypes: () => GLOBAL_EVENT_TYPES,
    onFrame: (_type, payload) => opts.onEvent(payload as ProjectEvent),
    onError: opts.onError,
    onOpen: opts.onOpen,
  });
}
