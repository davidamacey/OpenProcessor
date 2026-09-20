import { describe, expect, it } from 'vitest';
import { createAssistScope } from './assistScope.svelte';

describe('createAssistScope', () => {
  it('starts fully unscoped, serializing to an empty object (the byte-identity guarantee)', () => {
    const scope = createAssistScope();
    expect(scope.classId).toBeNull();
    expect(scope.detectionProfile).toBeNull();
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

  it('serializes detectionProfile alone under detection_profile', () => {
    const scope = createAssistScope();
    scope.detectionProfile = 'grounding_v2';
    expect(scope.toStartParams()).toEqual({ detection_profile: 'grounding_v2' });
  });

  it('serializes promptPack alone under prompt_pack', () => {
    const scope = createAssistScope();
    scope.promptPack = 'warehouse_v1';
    expect(scope.toStartParams()).toEqual({ prompt_pack: 'warehouse_v1' });
  });

  it('all three set independently — setting one never disturbs the others', () => {
    const scope = createAssistScope();
    scope.classId = 3;
    scope.detectionProfile = 'grounding_v2';
    scope.promptPack = 'warehouse_v1';
    expect(scope.toStartParams()).toEqual({
      class_id: 3,
      detection_profile: 'grounding_v2',
      prompt_pack: 'warehouse_v1',
    });
    expect(scope.classId).toBe(3);
    expect(scope.detectionProfile).toBe('grounding_v2');
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
    scope.detectionProfile = 'grounding_v2';
    scope.promptPack = 'warehouse_v1';
    scope.reset();
    expect(scope.classId).toBeNull();
    expect(scope.detectionProfile).toBeNull();
    expect(scope.promptPack).toBeNull();
    expect(scope.isDefault).toBe(true);
    expect(scope.toStartParams()).toEqual({});
  });

  // Pinned-key test: the wire param names are provisional (Q1, plan
  // §1.4) — this is the one assertion that fails obviously, rather than
  // silently, if the peer session confirms a different name.
  it('pins the exact emitted key names (Q1 — provisional, confirm with the peer session before the live pass)', () => {
    const scope = createAssistScope();
    scope.classId = 1;
    scope.detectionProfile = 'a';
    scope.promptPack = 'b';
    expect(Object.keys(scope.toStartParams()).sort()).toEqual([
      'class_id',
      'detection_profile',
      'prompt_pack',
    ]);
  });
});
