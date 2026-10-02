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

import { apiBase, globalApi, scoped } from './api';
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
  // for every region verify (event_hub.publish_region_verified), whatever
  // the slot key — the per-slot names above cover a backend that
  // namespaces verify events per slot.
  return Array.from(new Set(['crop.region_verified', ...derived]));
}

function knownEventTypes(): string[] {
  return [
    'crop.created',
    'crop.classified',
    'config.changed',
    // W10.11: advisory import progress; the job view re-reads the job.
    'dataset_import.progress',
    'dataset_import.finished',
    ...slotVerifiedEventTypes(),
  ];
}

/**
 * Open an SSE subscription to {API_PREFIX}/events.
 *
 * Reconnects with exponential backoff on error. The caller must invoke
 * `subscription.close()` on component unmount, otherwise the
 * EventSource will keep reconnecting forever.
 */
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

export type PipelineEvent =
  PipelineSnapshotEvent | PipelineStateEvent | PipelineStatsEvent;

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
  let es: EventSource | null = null;
  let backoff = RECONNECT_INITIAL_MS;
  let closed = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  const url = (() => {
    const base =
      apiBase && /^https?:\/\//i.test(apiBase)
        ? `${apiBase}${scoped()}/pipeline/events`
        : `${
            typeof window !== 'undefined' ? window.location.origin : ''
          }${apiBase}${scoped()}/pipeline/events`;
    return new URL(base).toString();
  })();

  function open(): void {
    if (closed) return;
    es = new EventSource(url);
    es.onopen = () => {
      backoff = RECONNECT_INITIAL_MS;
      opts.onOpen?.();
    };
    es.addEventListener('snapshot', (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data) as {
          state: Record<string, unknown>;
          stats: Record<string, unknown>;
        };
        opts.onSnapshot?.(payload.state ?? {}, payload.stats ?? {});
      } catch (err) {
        console.warn('[sse] failed to parse snapshot', err);
      }
    });
    es.addEventListener('state', (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data) as Record<string, unknown>;
        opts.onState?.(payload);
      } catch (err) {
        console.warn('[sse] failed to parse state', err);
      }
    });
    es.addEventListener('stats', (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data) as Record<string, unknown>;
        opts.onStats?.(payload);
      } catch (err) {
        console.warn('[sse] failed to parse stats', err);
      }
    });
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
  let es: EventSource | null = null;
  let backoff = RECONNECT_INITIAL_MS;
  let closed = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  const url = (() => {
    const base =
      apiBase && /^https?:\/\//i.test(apiBase)
        ? `${apiBase}${globalApi()}/events`
        : `${typeof window !== 'undefined' ? window.location.origin : ''}${apiBase}${globalApi()}/events`;
    return new URL(base).toString();
  })();

  function open(): void {
    if (closed) return;
    es = new EventSource(url);
    es.onopen = () => {
      backoff = RECONNECT_INITIAL_MS;
      opts.onOpen?.();
    };
    for (const t of GLOBAL_EVENT_TYPES) {
      es.addEventListener(t, (ev: MessageEvent) => {
        try {
          const payload = JSON.parse(ev.data) as ProjectEvent;
          opts.onEvent(payload);
        } catch (err) {
          console.warn('[sse] failed to parse global event', t, err);
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

export function subscribeCurationEvents(
  opts: CurationEventSubscribeOptions,
): CurationEventSubscription {
  let es: EventSource | null = null;
  let backoff = RECONNECT_INITIAL_MS;
  let closed = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  const url = (() => {
    // apiBase is empty in the default Docker deployment (nginx proxies
    // {API_PREFIX}/* on the same origin). `new URL('{API_PREFIX}/events')` throws because
    // it lacks a base, so we anchor to window.location.origin when
    // apiBase is relative. The resulting URL is still same-origin and
    // hits the labeler's nginx proxy.
    const base =
      apiBase && /^https?:\/\//i.test(apiBase)
        ? `${apiBase}${scoped()}/events`
        : `${typeof window !== 'undefined' ? window.location.origin : ''}${apiBase}${scoped()}/events`;
    const u = new URL(base);
    if (opts.topic) u.searchParams.set('topic', opts.topic);
    if (opts.class_id != null) u.searchParams.set('class_id', String(opts.class_id));
    return u.toString();
  })();

  function open(): void {
    if (closed) return;
    es = new EventSource(url);
    es.onopen = () => {
      // Reset backoff on a successful connect — a long-lived connection
      // shouldn't pay the penalty of an old transient blip.
      backoff = RECONNECT_INITIAL_MS;
      opts.onOpen?.();
    };
    // Subscribe to each known event-type listener separately. The
    // backend uses `event: <type>` headers so EventSource dispatches
    // typed events instead of falling back to `message`.
    for (const t of knownEventTypes()) {
      es.addEventListener(t, (ev: MessageEvent) => {
        try {
          const payload = JSON.parse(ev.data) as CurationEvent;
          opts.onEvent(payload);
        } catch (err) {
          // Garbled payload — log and skip. A bad event shouldn't
          // close the channel.
          console.warn('[sse] failed to parse event', t, err);
        }
      });
    }
    es.onerror = (err) => {
      // The browser will set readyState to CLOSED on a permanent
      // failure (e.g. CORS, 4xx). On a transient drop it goes to
      // CONNECTING and the browser reconnects on its own — but to be
      // safe (Firefox is known to give up after a few retries) we
      // explicitly close + reconnect ourselves with backoff.
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
