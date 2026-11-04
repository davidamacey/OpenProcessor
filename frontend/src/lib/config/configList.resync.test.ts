import { describe, expect, it, vi } from 'vitest';
import { ConfigList } from './configList.svelte';
import type { ConfigActive } from './configActive.svelte';

function build() {
  let resync: (() => void) | undefined;
  const list = vi.fn().mockResolvedValue(['a']);
  const backend = {
    list,
    clone: vi.fn(),
    remove: vi.fn(),
    subscribe: vi.fn((_onEvent: (e: unknown) => void, onResync?: () => void) => {
      resync = onResync;
      return { close() {} };
    }),
    isEvent: () => true,
  };
  const active = {
    load: vi.fn().mockResolvedValue(undefined),
  } as unknown as ConfigActive;
  const configList = new ConfigList<string[], unknown, ConfigActive, unknown>(
    backend,
    active,
  );
  return { configList, list, fireResync: () => resync?.() };
}

describe('ConfigList reconnect', () => {
  it('reads the list once on start and once more per reconnect', async () => {
    const { configList, list, fireResync } = build();
    configList.start();
    await vi.waitFor(() => expect(list).toHaveBeenCalledTimes(1));

    fireResync();

    await vi.waitFor(() => expect(list).toHaveBeenCalledTimes(2));
    configList.stop();
  });
});
