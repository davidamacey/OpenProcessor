/**
 * How `/models` renders a model's served status. Every value comes from
 * `GET {API_PREFIX}/models/status`; nothing here decides whether a model
 * is optional or installed.
 *
 * `not_installed` (OpenProcessor ba88751) is served only for an entry the
 * backend marks `optional` — today, a region profile's dedicated detector
 * when a segmenter covers the same job — and only when the model is absent
 * from the Triton repository entirely. It is not a fault, so it renders
 * neutral rather than as a warning.
 */

import type { ModelInfo } from '$lib/types_models';

export interface StatusPill {
  label: string;
  className: string;
  title: string | null;
}

const NEUTRAL = 'bg-surface-3/40 text-fg-2 border-line-strong';

export function modelStatusPill(
  model: Pick<ModelInfo, 'status' | 'optional'>,
): StatusPill {
  switch (model.status) {
    case 'ready':
      return {
        label: 'ready',
        className: 'bg-success/20 text-success border-success/40',
        title: null,
      };
    case 'not_ready':
      return {
        label: 'not ready',
        className: 'bg-warn/20 text-warn border-warn/40',
        title: null,
      };
    case 'not_configured':
      return { label: 'not configured', className: NEUTRAL, title: null };
    case 'not_installed':
      return {
        label: model.optional ? 'optional · not installed' : 'not installed',
        className: NEUTRAL,
        title: model.optional
          ? 'Optional model that is not installed; the pipeline runs without it.'
          : null,
      };
    default:
      return {
        label: 'unavailable',
        className: 'bg-danger/20 text-danger border-danger/40',
        title: null,
      };
  }
}

/** A model that isn't installed isn't "in use by the pipeline", even when
 *  the served `is_region_protected` flag still guards its name. */
export function isInstalled(model: Pick<ModelInfo, 'status'>): boolean {
  return model.status !== 'not_installed';
}
