import { describe, it, expect } from 'vitest';
import { AbortRegistry } from './abortRegistry';

describe('AbortRegistry', () => {
  it('aborts the previous controller when start() is called again for the same id', () => {
    const reg = new AbortRegistry();
    const first = reg.start('crop-1');
    expect(first.signal.aborted).toBe(false);
    const second = reg.start('crop-1');
    expect(first.signal.aborted).toBe(true);
    expect(second.signal.aborted).toBe(false);
  });

  it('does not abort a different id', () => {
    const reg = new AbortRegistry();
    const a = reg.start('crop-a');
    const b = reg.start('crop-b');
    expect(a.signal.aborted).toBe(false);
    expect(b.signal.aborted).toBe(false);
  });

  it('isCurrent is false once a newer start() supersedes the old controller', () => {
    const reg = new AbortRegistry();
    const first = reg.start('crop-1');
    expect(reg.isCurrent('crop-1', first)).toBe(true);
    const second = reg.start('crop-1');
    expect(reg.isCurrent('crop-1', first)).toBe(false);
    expect(reg.isCurrent('crop-1', second)).toBe(true);
  });

  it('finish() removes the entry only if it is still current (no-op if superseded)', () => {
    const reg = new AbortRegistry();
    const first = reg.start('crop-1');
    const second = reg.start('crop-1'); // supersedes `first`
    reg.finish('crop-1', first); // stale finish from the aborted request's `finally`
    expect(reg.isCurrent('crop-1', second)).toBe(true); // second must survive
    reg.finish('crop-1', second);
    expect(reg.isCurrent('crop-1', second)).toBe(false);
  });

  it('a fresh start() after finish() is not pre-aborted', () => {
    const reg = new AbortRegistry();
    const first = reg.start('crop-1');
    reg.finish('crop-1', first);
    const second = reg.start('crop-1');
    expect(second.signal.aborted).toBe(false);
  });
});
