const RULES = new Intl.PluralRules('en');

/** "1 item", "24 items": the count with the singular or plural noun. */
export function plural(n: number, singular: string, pluralForm = `${singular}s`): string {
  return `${n.toLocaleString()} ${RULES.select(n) === 'one' ? singular : pluralForm}`;
}
