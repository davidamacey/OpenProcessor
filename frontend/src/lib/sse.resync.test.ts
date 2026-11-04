/**
 * The backend sends no SSE `id:`, so events missed while the connection was
 * down are lost for good. A config subscriber therefore re-reads its state
 * on every connection after the first.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { subscribeConfigEvents } from './sse';

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  url: string;
  onopen: (() => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }
  addEventListener(): void {}
  close(): void {}
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
});

describe('subscribeConfigEvents', () => {
  it('does not resync on the first open', () => {
    const onResync = vi.fn();
    subscribeConfigEvents(() => {}, onResync);
    last().onopen?.();
    expect(onResync).not.toHaveBeenCalled();
  });

  it('resyncs once when the stream errors and reopens', () => {
    const onResync = vi.fn();
    subscribeConfigEvents(() => {}, onResync);
    last().onopen?.();

    last().onerror?.(new Event('error'));
    vi.advanceTimersByTime(1_000);
    expect(FakeEventSource.instances).toHaveLength(2);
    expect(onResync).not.toHaveBeenCalled();
    last().onopen?.();

    expect(onResync).toHaveBeenCalledTimes(1);
  });

  it('listens on the config topic', () => {
    subscribeConfigEvents(() => {});
    expect(last().url).toContain('topic=config');
  });
});
