/**
 * G5: resolveAutoLabelRunVlm decides whether AutoLabelPanel sends
 * `run_vlm: true`. Merged with createAssistScope().toStartParams() here
 * to prove the actual call site's inputs: a scoped run always implies
 * run_vlm, an unscoped run only sends it when the operator checked the
 * box, and an unscoped+unchecked run keeps `run_vlm` out of the request
 * entirely (byte-identical to every request this app has ever sent).
 */
import { describe, expect, it } from 'vitest';
import { createAssistScope } from './assistScope.svelte';
import { resolveAutoLabelRunVlm } from './autoLabelRunVlm';

describe('resolveAutoLabelRunVlm', () => {
  it('is false for an unscoped run with the checkbox unchecked', () => {
    const scope = createAssistScope();
    expect(resolveAutoLabelRunVlm(scope.toStartParams(), false)).toBe(false);
  });

  it('is true for an unscoped run when the checkbox is checked', () => {
    const scope = createAssistScope();
    expect(resolveAutoLabelRunVlm(scope.toStartParams(), true)).toBe(true);
  });

  it('is true whenever a class is scoped, regardless of the checkbox', () => {
    const scope = createAssistScope();
    scope.classId = 7;
    expect(resolveAutoLabelRunVlm(scope.toStartParams(), false)).toBe(true);
  });

  it('is true whenever a prompt pack is scoped, regardless of the checkbox', () => {
    const scope = createAssistScope();
    scope.promptPack = 'warehouse_v1';
    expect(resolveAutoLabelRunVlm(scope.toStartParams(), false)).toBe(true);
  });
});
