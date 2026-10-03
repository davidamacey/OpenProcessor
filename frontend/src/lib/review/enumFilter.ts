import type { ReviewFilterSpec } from '$lib/api';

/**
 * The value a served enum filter's `<select>` should show. Prefers the
 * operator's pick, then the tab's served default; when neither names one of
 * the spec's own options (the backend applies its own default for an omitted
 * param, so "unset" has no option of its own) it shows the first served
 * option instead of a blank one. Never invents an option or label.
 */
export function enumFilterSelection(
  spec: Pick<ReviewFilterSpec, 'options'>,
  chosen: string | undefined,
  servedDefault: unknown,
): string {
  const candidates = [chosen, servedDefault == null ? undefined : String(servedDefault)];
  for (const c of candidates) {
    if (c != null && spec.options.some((o) => o.value === c)) return c;
  }
  return spec.options[0]?.value ?? '';
}
