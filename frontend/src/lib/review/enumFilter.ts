import type { ReviewFilterSpec } from '$lib/api';

type EnumSpec = Pick<ReviewFilterSpec, 'options' | 'default'>;

/**
 * The value the backend applies when the param is omitted, when it names one
 * of the spec's own options: the spec's served `default` first, then the
 * tab's `filter_defaults` entry. `null` when nothing served names one.
 */
export function enumServedDefault(spec: EnumSpec, tabDefault: unknown): string | null {
  const candidates = [spec.default, tabDefault];
  for (const c of candidates) {
    if (c != null && spec.options.some((o) => o.value === String(c))) return String(c);
  }
  return null;
}

/**
 * The value a served enum filter's `<select>` should show: the operator's
 * pick, else the served default, else `''` (the "any" placeholder: nothing
 * is sent). Never guesses a value the queue request does not apply, and
 * never invents an option or label.
 */
export function enumFilterSelection(
  spec: EnumSpec,
  chosen: string | undefined,
  servedDefault: unknown,
): string {
  if (chosen != null && spec.options.some((o) => o.value === chosen)) return chosen;
  return enumServedDefault(spec, servedDefault) ?? '';
}
