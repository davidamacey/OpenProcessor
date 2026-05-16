/**
 * SSE helper — wraps EventSource with reconnect-with-backoff and
 * topic / class filtering (Task #92).
 *
 * The labeler /review and /clusters pages call `subscribeKbEvents()` on
 * mount and prepend incoming `crop.*` events into their lists. When the
 * connection drops (network blip, proxy timeout, page sleep) we retry
 * with a capped exponential backoff so a brief disconnect doesn't leave
 * the page stale.
 *
 * The backend endpoint is `GET /curation/events?topic=...&class_id=...` and
 * emits `text/event-stream` with `event:` lines naming the event type.
 * We surface every typed event back to the caller via a single
 * `onEvent` callback so the page can switch on `event.type`.
 */

import { apiBase } from './api';

// All event payloads share these fields; specific types add more.
export interface OpBaseEvent {
  type: string;
  ts?: number;
  topic?: string;
  crop_id?: string;
}

export interface OpCropCreatedEvent extends OpBaseEvent {
  type: 'crop.created';
  crop_id: string;
  image_path?: string;
}

export interface OpCropClassifiedEvent extends OpBaseEvent {
  type: 'crop.classified';
  crop_id: string;
  class_id?: number | null;
  class_name?: string | null;
  class_source?: string;
}

export interface OpCropPlateVerifiedEvent extends OpBaseEvent {
  type: 'crop.plate_verified';
  crop_id: string;
  plate_status?: string;
  plate_text?: string | null;
}

export type OpEvent =
  | OpCropCreatedEvent
  | OpCropClassifiedEvent
  | OpCropPlateVerifiedEvent
  | OpBaseEvent;

export interface OpEventSubscribeOptions {
  topic?: string;
  class_id?: number;
  onEvent: (event: OpEvent) => void;
  onError?: (err: Event | Error) => void;
  onOpen?: () => void;
}

export interface OpEventSubscription {
  /** Close the EventSource and stop reconnecting. */
  close(): void;
}

const RECONNECT_INITIAL_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;
// Event types we care about. Everything else is silently ignored so a
// future event type added on the backend doesn't trip up older clients.
const KNOWN_EVENT_TYPES = [
  'crop.created',
  'crop.classified',
  'crop.plate_verified',
];

/**
 * Open an SSE subscription to /curation/events.
 *
 * Reconnects with exponential backoff on error. The caller must invoke
 * `subscription.close()` on component unmount, otherwise the
 * EventSource will keep reconnecting forever.
 */
export function subscribeKbEvents(opts: OpEventSubscribeOptions): OpEventSubscription {
  let es: EventSource | null = null;
  let backoff = RECONNECT_INITIAL_MS;
  let closed = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  const url = (() => {
    // apiBase is empty in the default Docker deployment (nginx proxies
    // /curation/* on the same origin). `new URL('/curation/events')` throws because
    // it lacks a base, so we anchor to window.location.origin when
    // apiBase is relative. The resulting URL is still same-origin and
    // hits the labeler's nginx proxy.
    const base =
      apiBase && /^https?:\/\//i.test(apiBase)
        ? `${apiBase}/curation/events`
        : `${typeof window !== 'undefined' ? window.location.origin : ''}${apiBase}/curation/events`;
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
    for (const t of KNOWN_EVENT_TYPES) {
      es.addEventListener(t, (ev: MessageEvent) => {
        try {
          const payload = JSON.parse(ev.data) as OpEvent;
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
