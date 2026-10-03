import type { ReviewFilterSpec } from '$lib/api';

/**
 * The value a served enum filter's `<select>` should show: the operator's
 * pick when it names one of the spec's own options (an explicit `''` is the
 * served "Any" option), else the spec's served `default`. When neither names
 * an option the select shows `''`, never the first option as a stand-in.
 */
export function enumFilterSelection(
  spec: Pick<ReviewFilterSpec, 'options' | 'default'>,
  chosen: string | undefined,
): string {
  const served = spec.default == null ? '' : String(spec.default);
  for (const c of [chosen, served]) {
    if (c != null && spec.options.some((o) => o.value === c)) return c;
  }
  return '';
}
