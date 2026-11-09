/**
 * Event frames carry an opaque `id`. A browser-side reconnect (readyState
 * CONNECTING) sends `Last-Event-ID` itself and the server replays, so the
 * client does nothing. The client resyncs only when the server says `reset`
 * (the cursor was too old to replay) or when it had to open a new EventSource
 * (the browser gave up, or the server ended a slow client with `reconnect`),
 * which cannot carry the cursor.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { subscribeConfigEvents, subscribeGlobalEvents } from './sse';

class FakeEventSource {
  static CLOSED = 2;
  static instances: FakeEventSource[] = [];
  url: string;
  readyState = 0;
  closed = false;
  onopen: (() => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  listeners = new Map<string, () => void>();
  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }
  addEventListener(type: string, fn: () => void): void {
    this.listeners.set(type, fn);
  }
  close(): void {
    this.closed = true;
  }
}

const last = () => FakeEventSource.instances[FakeEventSource.instances.length - 1]!;

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.useFakeTimers();
  vi.stubGlobal('EventSource', FakeEventSource);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('subscribeConfigEvents', () => {
  it('does not resync on the first open', () => {
    const onResync = vi.fn();
    subscribeConfigEvents(() => {}, onResync);
    last().onopen?.();
    expect(onResync).not.toHaveBeenCalled();
  });

  it('leaves a browser reconnect (CONNECTING) to the browser: same source, no resync', () => {
    const onResync = vi.fn();
    subscribeConfigEvents(() => {}, onResync);
    last().onopen?.();
    last().readyState = 0;
    last().onerror?.(new Event('error'));
    vi.advanceTimersByTime(60_000);
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(last().closed).toBe(false);
    last().onopen?.();
    expect(onResync).not.toHaveBeenCalled();
  });

  it('resyncs once when the browser gave up and a new source had to be opened', () => {
    const onResync = vi.fn();
    subscribeConfigEvents(() => {}, onResync);
    last().onopen?.();

    last().readyState = 2;
    last().onerror?.(new Event('error'));
    vi.advanceTimersByTime(1_000);
    expect(FakeEventSource.instances).toHaveLength(2);
    expect(onResync).not.toHaveBeenCalled();
    last().onopen?.();
    expect(onResync).toHaveBeenCalledTimes(1);
    last().onopen?.();
    expect(onResync).toHaveBeenCalledTimes(1);
  });

  it('resyncs on a server reset without reopening', () => {
    const onResync = vi.fn();
    subscribeConfigEvents(() => {}, onResync);
    last().onopen?.();
    last().listeners.get('reset')?.();
    expect(onResync).toHaveBeenCalledTimes(1);
    expect(FakeEventSource.instances).toHaveLength(1);
  });

  it('reopens after the final reconnect event of a slow client and resyncs once open', () => {
    const onResync = vi.fn();
    subscribeConfigEvents(() => {}, onResync);
    last().onopen?.();
    const first = last();
    first.listeners.get('reconnect')?.();
    expect(first.closed).toBe(true);
    vi.advanceTimersByTime(1_000);
    expect(FakeEventSource.instances).toHaveLength(2);
    last().onopen?.();
    expect(onResync).toHaveBeenCalledTimes(1);
  });

  it('listens on the config topic', () => {
    subscribeConfigEvents(() => {});
    expect(last().url).toContain('topic=config');
  });
});

describe('subscribeGlobalEvents', () => {
  it('resyncs on a server reset', () => {
    const onResync = vi.fn();
    subscribeGlobalEvents({ onEvent: () => {}, onResync });
    last().listeners.get('reset')?.();
    expect(onResync).toHaveBeenCalledTimes(1);
  });
});
