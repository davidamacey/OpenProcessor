/** The class names the picker ranks first: the item's own proposal, then the served hints. */
export function pickerRecent(
  proposed: string | null | undefined,
  hints: readonly string[],
): string[] {
  return [...new Set([...(proposed ? [proposed] : []), ...hints])];
}
