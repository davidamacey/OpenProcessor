/**
 * The three subscriptions share one open / backoff / reconnect loop. These
 * tests drive a fake EventSource through that loop for each of them.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  subscribeCurationEvents,
  subscribeGlobalEvents,
  subscribePipelineEvents,
} from './sse';

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  url: string;
  closed = false;
  onopen: (() => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  listeners = new Map<string, (ev: MessageEvent) => void>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }
  addEventListener(type: string, fn: (ev: MessageEvent) => void): void {
    this.listeners.set(type, fn);
  }
  close(): void {
    this.closed = true;
  }
  emit(type: string, data: unknown): void {
    this.listeners.get(type)?.({ data: JSON.stringify(data) } as MessageEvent);
  }
}

const last = () => FakeEventSource.instances[FakeEventSource.instances.length - 1];

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.useFakeTimers();
  vi.stubGlobal('EventSource', FakeEventSource);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const subscribers = {
  curation: () => subscribeCurationEvents({ onEvent: () => {}, onError, onOpen }),
  global: () => subscribeGlobalEvents({ onEvent: () => {}, onError, onOpen }),
  pipeline: () => subscribePipelineEvents({ onError, onOpen }),
};
const onError = vi.fn();
const onOpen = vi.fn();

describe.each(Object.entries(subscribers))('%s subscription', (_name, subscribe) => {
  beforeEach(() => {
    onError.mockClear();
    onOpen.mockClear();
  });

  it('opens one EventSource on an absolute URL and reports open', () => {
    subscribe();
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(last().url).toMatch(/^http:\/\/.+\/events$|\/pipeline\/events$/);
    last().onopen?.();
    expect(onOpen).toHaveBeenCalledTimes(1);
  });

  it('closes the failed source and reconnects after the backoff, doubling it', () => {
    subscribe();
    const first = last();
    first.onerror?.(new Event('error'));
    expect(onError).toHaveBeenCalledTimes(1);
    expect(first.closed).toBe(true);
    vi.advanceTimersByTime(999);
    expect(FakeEventSource.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeEventSource.instances).toHaveLength(2);

    last().onerror?.(new Event('error'));
    vi.advanceTimersByTime(1999);
    expect(FakeEventSource.instances).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(FakeEventSource.instances).toHaveLength(3);
  });

  it('caps the backoff and resets it after a successful open', () => {
    subscribe();
    for (let i = 0; i < 8; i += 1) {
      last().onerror?.(new Event('error'));
      vi.advanceTimersByTime(30_000);
    }
    const n = FakeEventSource.instances.length;
    last().onerror?.(new Event('error'));
    vi.advanceTimersByTime(29_999);
    expect(FakeEventSource.instances).toHaveLength(n);
    vi.advanceTimersByTime(1);
    expect(FakeEventSource.instances).toHaveLength(n + 1);

    last().onopen?.();
    last().onerror?.(new Event('error'));
    vi.advanceTimersByTime(1000);
    expect(FakeEventSource.instances).toHaveLength(n + 2);
  });

  it('stops for good on close(), cancelling a pending reconnect', () => {
    const sub = subscribe();
    last().onerror?.(new Event('error'));
    sub.close();
    vi.advanceTimersByTime(60_000);
    expect(FakeEventSource.instances).toHaveLength(1);

    const sub2 = subscribe();
    const open = last();
    sub2.close();
    expect(open.closed).toBe(true);
  });
});

describe('payload handling', () => {
  it('delivers a parsed global event and survives a garbled one', () => {
    const onEvent = vi.fn();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    subscribeGlobalEvents({ onEvent });
    last().emit('project.created', { type: 'project.created', topic: 'project' });
    expect(onEvent).toHaveBeenCalledWith({ type: 'project.created', topic: 'project' });
    last().listeners.get('project.created')?.({ data: '{nope' } as MessageEvent);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(onEvent).toHaveBeenCalledTimes(1);
  });

  it('routes pipeline snapshot, state and stats frames', () => {
    const onSnapshot = vi.fn();
    const onState = vi.fn();
    const onStats = vi.fn();
    subscribePipelineEvents({ onSnapshot, onState, onStats });
    last().emit('snapshot', { state: { a: 1 }, stats: { b: 2 } });
    last().emit('state', { c: 3 });
    last().emit('stats', { d: 4 });
    expect(onSnapshot).toHaveBeenCalledWith({ a: 1 }, { b: 2 });
    expect(onState).toHaveBeenCalledWith({ c: 3 });
    expect(onStats).toHaveBeenCalledWith({ d: 4 });
  });

  it('puts the topic and class filter on the curation URL and re-reads slot event types on reconnect', () => {
    subscribeCurationEvents({ topic: 'config', class_id: 7, onEvent: () => {} });
    const u = new URL(last().url);
    expect(u.searchParams.get('topic')).toBe('config');
    expect(u.searchParams.get('class_id')).toBe('7');
    expect([...last().listeners.keys()]).toContain('crop.region_verified');
  });
});
