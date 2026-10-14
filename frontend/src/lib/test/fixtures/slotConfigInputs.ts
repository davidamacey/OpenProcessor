/**
 * Shared fixtures for the slot-config parser tests: a minimal valid slot,
 * a fully-loaded valid slot for the per-field rejection cases to mutate,
 * and the rejection assertion every invalid case uses.
 */

import { expect } from 'vitest';
import { parseSlotConfig } from '$lib/annotations/config/parseSlotConfig';

/** A minimal, fully valid slot — every required field present, no
 *  optional field set. Mutate a clone of this for the "accepts" cases. */
export function minimalSlot(): Record<string, unknown> {
  return {
    key: 'widget_tag',
    bind: { className: 'widget' },
    label: { singular: 'tag', plural: 'tags', title: 'Tag' },
    capabilities: {},
    endpoints: {},
  };
}

/** A fully-loaded valid slot exercising every capability, for the
 *  per-field rejection tests to mutate. */
export function fullSlot(): Record<string, unknown> {
  return {
    key: 'widget_tag',
    bind: { className: 'widget' },
    label: { singular: 'tag', plural: 'tags', title: 'Tag' },
    capabilities: {
      subBox: {
        bboxField: 'tag_bbox_norm',
        storedFrame: 'source',
        frameField: 'tag_bbox_frame',
        scoreField: 'tag_score',
        visibleField: 'tag_visible',
        thumbnail: {
          path: '/crops/{cropId}/region_thumbnail?box_id={boxId}&size={size}',
          aspect: '1 / 1',
          defaultSize: 160,
        },
        ring: 'default',
        editor: { thumbSize: 512, viewPadding: 2.0, nudgeStep: 0.001953125 },
      },
      text: {
        valueField: 'tag_text',
        rawField: 'tag_text_raw',
        sourceField: 'tag_text_source',
        confidenceField: 'tag_text_confidence',
        label: 'Tag text',
        placeholder: '000123',
        transform: 'uppercase',
        monospace: true,
        pattern: { source: '^[0-9]{6}$', flags: '' },
        maxLength: 6,
      },
      provenance: {
        detectorField: 'tag_detector',
        detectorVersionField: 'tag_detector_version',
        chainField: 'tag_detector_chain',
        showChainOnCard: true,
      },
      lifecycle: {
        statusField: 'tag_status',
        verifiedField: 'tag_verified',
        states: [
          { value: 'detected', label: 'detected', humanWritable: true, role: 'proposed' },
          {
            value: 'no_tag',
            label: 'no tag visible',
            humanWritable: true,
            role: 'absent',
          },
          {
            value: 'false_positive',
            label: 'false positive',
            humanWritable: true,
            role: 'falsePositive',
            dim: true,
            badge: 'false pos',
          },
        ],
        confirmState: 'detected',
        rejectState: 'no_tag',
        falsePositiveState: 'false_positive',
      },
      queue: {
        endpointId: 'widget_tags',
        urlId: 'widget_tags',
        tabLabel: 'Widget tags',
        browsePath: '/widget_tags',
        keymap: {
          confirm: ['enter'],
          reject: ['r'],
          markFalsePositive: ['f'],
          editBox: ['e'],
          back: ['arrowleft'],
        },
        textFilter: { param: 'text', label: 'Tag', placeholder: 'e.g. 000123' },
        alwaysVisible: true,
      },
      trainingCohorts: {
        cohorts: [
          {
            id: 'validated_tags',
            label: 'Validated tags',
            description: 'Human-confirmed tags',
            query: {
              kind: 'endpoint',
              path: '/widgets/training_candidates',
              params: { class_id: '{classId}' },
            },
            rowKind: 'slot',
            reviewTarget: 'slotQueue',
          },
        ],
      },
    },
    endpoints: {
      setBox: '/crops/{cropId}/tag',
      clearBox: '/crops/{cropId}/tag',
      patchMeta: '/crops/{cropId}/tag_meta',
      batchStatus: '/widget_tags/batch_status',
    },
    stats: {
      key: 'widget_tags',
      panelTitle: 'Widget tags',
      coverageTitle: 'Widget tag coverage',
    },
  };
}

export function expectRejected(
  result: ReturnType<typeof parseSlotConfig>,
  match: RegExp,
) {
  expect(result.slot).toBeUndefined();
  expect(result.errors.some((e) => match.test(e))).toBe(true);
}
