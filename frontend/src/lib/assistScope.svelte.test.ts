import { describe, expect, it } from 'vitest';
import { createAssistScope } from './assistScope.svelte';

describe('createAssistScope', () => {
  it('starts fully unscoped, serializing to an empty object (the byte-identity guarantee)', () => {
    const scope = createAssistScope();
    expect(scope.classId).toBeNull();
    expect(scope.promptPack).toBeNull();
    expect(scope.isDefault).toBe(true);
    // `{}` here is what makes an unscoped `startAutoLabel({...})` call
    // produce the exact same URL this app has always sent.
    expect(scope.toStartParams()).toEqual({});
  });

  it('serializes a selected classId under class_id', () => {
    const scope = createAssistScope();
    scope.classId = 7;
    expect(scope.isDefault).toBe(false);
    expect(scope.toStartParams()).toEqual({ class_id: 7 });
  });

  it('a class id of 0 must serialize — `== null` is the right check, not `!classId`', () => {
    const scope = createAssistScope();
    scope.classId = 0;
    expect(scope.toStartParams()).toEqual({ class_id: 0 });
    expect(scope.isDefault).toBe(false);
  });

  it('serializes promptPack alone under prompt_pack', () => {
    const scope = createAssistScope();
    scope.promptPack = 'warehouse_v1';
    expect(scope.toStartParams()).toEqual({ prompt_pack: 'warehouse_v1' });
  });

  it('both set independently — setting one never disturbs the other', () => {
    const scope = createAssistScope();
    scope.classId = 3;
    scope.promptPack = 'warehouse_v1';
    expect(scope.toStartParams()).toEqual({
      class_id: 3,
      prompt_pack: 'warehouse_v1',
    });
    expect(scope.classId).toBe(3);
    expect(scope.promptPack).toBe('warehouse_v1');
  });

  it('setting a field back to null removes its key and restores isDefault when it was the only one set', () => {
    const scope = createAssistScope();
    scope.classId = 3;
    expect(scope.isDefault).toBe(false);
    scope.classId = null;
    expect(scope.isDefault).toBe(true);
    expect(scope.toStartParams()).toEqual({});
  });

  it('reset() returns every field to null and toStartParams() to {}', () => {
    const scope = createAssistScope();
    scope.classId = 3;
    scope.promptPack = 'warehouse_v1';
    scope.reset();
    expect(scope.classId).toBeNull();
    expect(scope.promptPack).toBeNull();
    expect(scope.isDefault).toBe(true);
    expect(scope.toStartParams()).toEqual({});
  });

  // Pinned wire names, confirmed against OpenProcessor main (f4551bf).
  // detection_profile must never be emitted: main rejects it with a 422
  // because region detection is startup config, not a per-run choice.
  it('pins the exact emitted key names, and never emits detection_profile', () => {
    const scope = createAssistScope();
    scope.classId = 1;
    scope.promptPack = 'b';
    expect(Object.keys(scope.toStartParams()).sort()).toEqual([
      'class_id',
      'prompt_pack',
    ]);
  });

  it('serializes a picked VLM endpoint under vlm; unset sends nothing', () => {
    const scope = createAssistScope();
    expect(scope.toStartParams()).not.toHaveProperty('vlm');
    scope.vlm = 'cloud_vlm';
    expect(scope.isDefault).toBe(false);
    expect(scope.toStartParams()).toEqual({ vlm: 'cloud_vlm' });
  });

  it('acknowledge_external is sent only when true, never as false', () => {
    const scope = createAssistScope();
    scope.vlm = 'cloud_vlm';
    expect(scope.toStartParams()).not.toHaveProperty('acknowledge_external');
    scope.acknowledgeExternal = true;
    expect(scope.toStartParams()).toEqual({
      vlm: 'cloud_vlm',
      acknowledge_external: true,
    });
    scope.acknowledgeExternal = false;
    expect(scope.toStartParams()).toEqual({ vlm: 'cloud_vlm' });
  });

  it('reset() clears the VLM pick and its acknowledgement', () => {
    const scope = createAssistScope();
    scope.vlm = 'cloud_vlm';
    scope.acknowledgeExternal = true;
    scope.reset();
    expect(scope.vlm).toBeNull();
    expect(scope.acknowledgeExternal).toBe(false);
    expect(scope.isDefault).toBe(true);
    expect(scope.toStartParams()).toEqual({});
  });
});
