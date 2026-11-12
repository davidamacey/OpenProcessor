import { beforeEach, describe, expect, it, vi } from 'vitest';

const goto = vi.hoisted(() => vi.fn());
const success = vi.hoisted(() => vi.fn());
vi.mock('$app/navigation', () => ({ goto }));
vi.mock('$app/paths', () => ({ resolve: (p: string) => `/r${p}` }));
vi.mock('$stores/toast.svelte', () => ({ toastStore: { success } }));

import { createConfigEditorFlow } from './editorFlow.svelte';

function build(over: { save?: boolean; clone?: unknown } = {}) {
  const ed = {
    active: { clearAction: vi.fn() },
    doc: { revision: 4 },
    save: vi.fn().mockResolvedValue(over.save ?? true),
  };
  const cloner = {
    clearClone: vi.fn(),
    clone: vi
      .fn()
      .mockResolvedValue(over.clone === undefined ? { name: 'copy' } : over.clone),
  };
  const flow = createConfigEditorFlow({
    ed: () => ed as never,
    cloner,
    name: () => 'orig',
    editorPath: (n) => `/e/${n}`,
  });
  return { flow, ed, cloner };
}

beforeEach(() => {
  goto.mockReset();
  success.mockReset();
});

describe('createConfigEditorFlow', () => {
  it('opening an activation clears the previous action first', () => {
    const { flow, ed } = build();
    const target = { name: 'x', revision: 2 } as never;
    flow.openActivate(target);
    expect(ed.active.clearAction).toHaveBeenCalled();
    expect(flow.activating).toBe(target);
  });

  it('toasts the saved revision only when the save went through', async () => {
    const ok = build();
    await ok.flow.doSave();
    expect(success).toHaveBeenCalledWith('Saved revision 4');
    success.mockReset();
    await build({ save: false }).flow.doSave();
    expect(success).not.toHaveBeenCalled();
  });

  it('clone-to-edit clones the document on screen and opens the copy', async () => {
    const { flow, cloner } = build();
    flow.startClone();
    expect(cloner.clearClone).toHaveBeenCalled();
    expect(flow.cloning).toBe(true);
    await flow.doClone('copy');
    expect(cloner.clone).toHaveBeenCalledWith({ name: 'orig', source: null }, 'copy', '');
    expect(flow.cloning).toBe(false);
    expect(goto).toHaveBeenCalledWith('/r/e/copy');
  });

  it('keeps the clone dialog open when the server refuses', async () => {
    const { flow } = build({ clone: null });
    flow.startClone();
    await flow.doClone('copy');
    expect(flow.cloning).toBe(true);
    expect(goto).not.toHaveBeenCalled();
  });
});
