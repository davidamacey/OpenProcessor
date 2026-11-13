/** "Mixed: 19 person, 5 backpack" from the served detector-class facet of the open cluster;
 *  `null` when the facet names fewer than two classes. */
export function mixedBreakdown(
  classes: { value: string; count: number }[],
): string | null {
  const present = classes.filter((f) => f.count > 0);
  if (present.length < 2) return null;
  return `Mixed: ${present.map((f) => `${f.count.toLocaleString()} ${f.value}`).join(', ')}.`;
}
