/**
 * The two event streams a W9 surface follows: the GLOBAL
 * `vlm.changed` (registry or local-model catalog changed) and the
 * project-scoped `config.changed axis=vlm` (this project's activation
 * changed). Both are only wake-ups to re-read; the served reads stay the
 * only source. The payload's `axis` (`registry` | `local_vlm`) is all that
 * is read (question A-4: `name` may not be served on a registry write).
 */
import { subscribeCurationEvents, subscribeGlobalEvents } from '$lib/sse';
import type { CurationEvent, ProjectEvent } from '$lib/sse';
import { isConfigAxisEvent } from '$lib/config/validationIssues';

export const VLM_AXIS = 'vlm';

export type VlmEvent = CurationEvent | ProjectEvent;

export function isVlmRegistryEvent(e: VlmEvent): boolean {
  return e.type === 'vlm.changed' && (e as { axis?: string }).axis === 'registry';
}

export function isVlmLocalEvent(e: VlmEvent): boolean {
  return e.type === 'vlm.changed' && (e as { axis?: string }).axis === 'local_vlm';
}

/** `config.changed` on the `vlm` axis: this project's activation moved. */
export function isVlmActiveEvent(e: VlmEvent): boolean {
  return isConfigAxisEvent(e as CurationEvent, VLM_AXIS);
}

/** Opens both streams; `close()` closes both. */
export function subscribeVlmEvents(onEvent: (e: VlmEvent) => void): {
  close(): void;
} {
  const global = subscribeGlobalEvents({ onEvent: (e) => onEvent(e) });
  const scoped = subscribeCurationEvents({ topic: 'config', onEvent });
  return {
    close() {
      global.close();
      scoped.close();
    },
  };
}
