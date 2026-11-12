/**
 * What the ingest policy's `detect.class_resolution` does, in words. The values are the contract's
 * enum (`DetectFilter.class_resolution`); the form help and the Ingest page read this one table.
 */
import type { ClassResolution } from '$lib/types_detector';

export const RESOLUTION_HELP: Record<ClassResolution, string> = {
  proposal: 'The detector label is kept as a proposal; no class is assigned.',
  by_name:
    'Gives the box the class named like the detector label, adding it to the registry when it is missing. It is a machine label, reviewable and still clustered.',
};

/** The short form for a line of facts: `Class: ...`. */
export const RESOLUTION_SHORT: Record<ClassResolution, string> = {
  proposal: 'proposals only',
  by_name: "the detector's class as a machine label",
};
