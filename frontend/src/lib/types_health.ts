/**
 * Health and dashboard wire types: global and project health, the served
 * region profile and the stats summary.
 */
import type { ClassThresholds } from '$lib/types_classes';

export interface StatsSummary {
  /** Set when `/stats/dataset` failed; the totals below are then zeros,
   *  not real counts. `per_class` comes from `/stats/classes` and stays. */
  dataset_error: string | null;
  total_crops: number;
  validated_crops: number;
  test_holdout_crops: number;
  per_class: Array<{
    class_id: number;
    class_name: string;
    count: number;
    validated_count: number;
    /** Server-computed adequacy tier (`block`/`warn`/`ok`) — see
     *  `RegistryClass.adequacy`. */
    adequacy: string;
    /** Server-computed YOLO augmentation target for this class. */
    aug_target: number;
    /** `aug_target - validated_count`, served directly. */
    aug_gap: number;
    /** See `RegistryClass.trainable`. */
    trainable: number;
    /** See `RegistryClass.trainable_gap` (shortfall against the per-class
     *  hard minimum, not against `aug_target`). */
    trainable_gap: number;
  }>;
  /** Served alongside `per_class` on `/stats/classes` — same shape as
   *  `ClassesResponse.thresholds`. Absent only when `/stats/classes`
   *  itself failed to load. */
  thresholds?: ClassThresholds;
}

/** `GET {globalApi()}/health` (P1 projects cutover) — unscoped, has no
 *  project bound. Feeds only the top-bar API status chip; every
 *  project-scoped fact (region profile, queue counts, …) comes from the
 *  scoped `ApiHealth` below. */
/** The backend's `vlm_status()` on `GET /health` (the contract types it as
 *  an open object; it serves no endpoint name, question A-6). */
export interface VlmHealth {
  reachable: boolean;
  model?: string | null;
  last_error?: string | null;
  detail?: string | null;
}

export interface GlobalHealth {
  status: 'ok' | 'degraded' | 'down';
  triton?: { reachable: boolean; detail?: string };
  opensearch?: { reachable: boolean; indexes?: Record<string, boolean> };
  vlm?: VlmHealth;
  version?: string;
  api_version?: string;
}

/** `GET {scoped()}/health` — today's project-scoped shape (P1 projects
 *  cutover) plus the bound `project` slug. `degraded` means a
 *  non-critical dependency (e.g. the VLM) is down; labeling still
 *  works. */
export interface ApiHealth {
  status: 'ok' | 'degraded' | 'down';
  triton?: { reachable: boolean; detail?: string };
  opensearch?: { reachable: boolean; indexes?: Record<string, boolean> };
  vlm?: VlmHealth;
  registry?: { path?: string; exists?: boolean; mtime?: string | null };
  /** The backend's active region profile, or `null` when none is
   *  configured (then every region route answers 409). The only signal
   *  region features key on. */
  region_profile: ServedRegionProfile | null;
  /** The project this scoped health was read from. */
  project: string;
}

/** `RegionProfileSummary` on `GET {API_PREFIX}/health` and
 *  `GET {API_PREFIX}/regions/vocabulary`. */
export interface ServedRegionProfile {
  /** Profile id; the region slot's key. */
  name: string;
  /** The region noun shown to operators (tab label, gallery copy). May be
   *  empty when the profile doesn't set one. */
  display_name: string;
  /** The singular form of the region noun, for a singular-context label
   *  ("Confirm <Region>", "<Region> score"). May be empty — falls back
   *  to the generic "Region" (OpenProcessor #36 item 10, 2026-09-25). */
  display_name_singular: string;
  /** The class whose items ARE regions (e.g. the class a region export
   *  writes). May be empty. */
  region_class_name: string;
  /** How region text is read (`vlm` / `ocr` / `vlm_then_ocr` / `both`),
   *  or `'none'` for a text-free profile — a profile that detects/segments a region but never
   *  reads text off it. */
  text_reader: string;
  /** False for a text-free profile: no region text is read, stored or
   *  editable — `region_text*` stays null and a `region_meta` PATCH
   *  carrying `region_text` 422s. The one gate for the
   *  region slot's text capability (`servedRegionSlot.ts`). */
  reads_text: boolean;
  /** Whether the OCR text-hint re-pass (after a segmenter miss) is
   *  enabled for this profile. Informational only today — no UI reads it
   *  yet (see CLAUDE.md). */
  text_hint_enabled: boolean;
  /** Request-size guards on region box writes — never a labeling rule.
   *  `max_boxes_per_write` gates the Add-box action (human box lists are
   *  otherwise unbounded). */
  limits: {
    max_boxes_per_write: number;
  };
}
