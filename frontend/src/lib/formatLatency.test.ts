import { describe, expect, it } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';
import { formatLatencyMs } from './formatLatency';
import TestRefs from '$lib/components/configTest/TestRefs.svelte';

describe('formatLatencyMs', () => {
  it('rounds an unrounded served float to whole ms with a unit', () => {
    expect(formatLatencyMs(1121.9730039592832)).toBe('1122 ms');
    expect(formatLatencyMs(0.4)).toBe('0 ms');
  });
  it('renders unknown as a dash, never 0', () => {
    expect(formatLatencyMs(null)).toBe('—');
    expect(formatLatencyMs(undefined)).toBe('—');
    expect(formatLatencyMs(Number.NaN)).toBe('—');
  });
});

describe('TestRefs latency', () => {
  it('shows the rounded latency', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const inst = mount(TestRefs, { target, props: { latencyMs: 1121.9730039592832 } });
    flushSync();
    expect(target.querySelector('[data-testid="test-latency"]')?.textContent).toBe(
      '1122 ms',
    );
    unmount(inst);
    target.remove();
  });
});
