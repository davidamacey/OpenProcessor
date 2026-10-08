/** A served pack/profile ref as `name@revision`, or "draft" for an
 *  unsaved draft. Pure formatting of served fields. */
export function refText(ref: {
  draft: boolean;
  name: string | null;
  revision: number | null;
}): string {
  if (ref.draft) return 'draft';
  const name = ref.name ?? '—';
  return ref.revision == null ? name : `${name}@${ref.revision}`;
}
