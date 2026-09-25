/**
 * Display placeholder for a served `snake_case` id the backend doesn't
 * label yet: underscores become spaces, the first letter is capitalised,
 * and reader/model acronyms stay upper-case ("vlm_preferred" → "VLM
 * preferred", not "Vlm Preferred" — visual audit 2026-09-24, R6).
 *
 * A stand-in, not a label table: replace a call site with the served
 * label as soon as the backend serves one.
 */
const ACRONYMS = new Set(['vlm', 'ocr', 'id', 'nms', 'iou']);

export function humanizeId(id: string): string {
  const words = id
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((w) => (ACRONYMS.has(w.toLowerCase()) ? w.toUpperCase() : w.toLowerCase()));
  if (words.length === 0) return id;
  const [first, ...rest] = words;
  return [first.charAt(0).toUpperCase() + first.slice(1), ...rest].join(' ');
}
