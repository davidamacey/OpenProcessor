import { beforeEach, describe, expect, it, vi } from 'vitest';

const goto = vi.hoisted(() => vi.fn());
const success = vi.hoisted(() => vi.fn());
vi.mock('$app/navigation', () => ({ goto }));
vi.mock('$app/paths', () => ({ resolve: (p: string) => `/r${p}` }));
vi.mock('$stores/toast.svelte', () => ({ toastStore: { success } }));

import { createCloneDeleteFlow } from './cloneDeleteFlow.svelte';

interface Row {
  name: string;
}

function build(over: Partial<{ clone: unknown; remove: unknown }> = {}) {
  const list = {
    clearClone: vi.fn(),
    clone: vi.fn().mockResolvedValue({ name: 'copy' }),
    remove: vi.fn().mockResolvedValue(true),
    deleteError: 'old' as string | null,
    ...over,
  };
  return { list, flow: createCloneDeleteFlow<Row>(list as never, (n) => `/e/${n}`) };
}

beforeEach(() => {
  goto.mockReset();
  success.mockReset();
});

describe('createCloneDeleteFlow', () => {
  it('clones: clears the previous report, then toasts and opens the new editor', async () => {
    const { flow, list } = build();
    flow.openClone({ name: 'src', source: null });
    expect(list.clearClone).toHaveBeenCalled();
    expect(flow.cloneFrom).toEqual({ name: 'src', source: null });
    await flow.doClone('copy', 'd', null);
    expect(list.clone).toHaveBeenCalledWith(
      { name: 'src', source: null },
      'copy',
      'd',
      null,
    );
    expect(flow.cloneFrom).toBeNull();
    expect(success).toHaveBeenCalledWith('Created copy');
    expect(goto).toHaveBeenCalledWith('/r/e/copy');
  });

  it('keeps the dialog open when the clone is refused', async () => {
    const { flow } = build({ clone: vi.fn().mockResolvedValue(null) });
    flow.openClone({ name: 'src', source: 'template' });
    await flow.doClone('copy', '', null);
    expect(flow.cloneFrom).not.toBeNull();
    expect(goto).not.toHaveBeenCalled();
  });

  it('does nothing without a clone source', async () => {
    const { flow, list } = build();
    await flow.doClone('x', '', null);
    expect(list.clone).not.toHaveBeenCalled();
  });

  it('deletes: asking clears the old error; success closes and toasts, refusal keeps it open', async () => {
    const { flow, list } = build();
    flow.askDelete({ name: 'a' });
    expect(list.deleteError).toBeNull();
    expect(flow.deleting).toEqual({ name: 'a' });
    await flow.doDelete();
    expect(flow.deleting).toBeNull();
    expect(success).toHaveBeenCalledWith('Deleted a');

    const refused = build({ remove: vi.fn().mockResolvedValue(false) });
    refused.flow.askDelete({ name: 'b' });
    await refused.flow.doDelete();
    expect(refused.flow.deleting).toEqual({ name: 'b' });
  });
});
