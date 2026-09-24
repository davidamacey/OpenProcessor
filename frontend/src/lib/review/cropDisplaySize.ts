/**
 * DQ-M5 (docs/design/data-quality-pass-2026-09-24.md): the review crop
 * panel's `<img>` used `h-full w-full object-contain` (phase-A p9's "fit
 * width" fix) so a small/tall crop — a 98×106 motorcycle-plate crop in the
 * repro — was stretched to fill the whole panel height: 758×820 at
 * 1600×1000, 918×993 at 1920×1080, 598×647 at 1280×720. That pushed
 * Reason, Proposed and Confirm/Skip/Discard below the fold at every
 * measured width, because nothing capped how far a tiny crop could be
 * blown up.
 *
 * `capCropDisplayStyle` is the pure part of the fix: given the crop's
 * natural pixel size (from the `<img>`'s own `naturalWidth`/`naturalHeight`,
 * via Svelte's `bind:naturalWidth`/`bind:naturalHeight`) and the panel's
 * available box, it returns an inline style that caps the *rendered* size
 * at `capFactor`× the natural size (never below 1×, i.e. never downscale
 * past fit) — so a 98px-tall crop tops out around 400px even inside a
 * much taller panel, leaving the rest of the panel for the metadata block
 * below it. The container CSS (`max-h-[46%]` on the crop panel in
 * review/+page.svelte) supplies the other half of the fix: a hard ceiling
 * on the panel's own height so the metadata block always gets its share
 * regardless of the image's natural size.
 */

export const CROP_UPSCALE_CAP = 4;

/**
 * Returns a CSS `style` attribute value (empty string until natural size
 * is known, e.g. before the image has loaded) that bounds the crop's
 * rendered box to `min(available, natural * capFactor)` per axis, while
 * still letting `object-contain` handle the aspect ratio within that box.
 */
export function capCropDisplayStyle(
  naturalWidth: number,
  naturalHeight: number,
  capFactor: number = CROP_UPSCALE_CAP,
): string {
  if (!(naturalWidth > 0) || !(naturalHeight > 0)) return '';
  const capW = Math.round(naturalWidth * capFactor);
  const capH = Math.round(naturalHeight * capFactor);
  return `max-width:min(100%, ${capW}px);max-height:min(100%, ${capH}px);`;
}
