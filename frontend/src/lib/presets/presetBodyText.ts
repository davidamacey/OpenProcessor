import type { PresetBody } from '$lib/types_presets';

/** A preset body as indented JSON text. */
export const bodyToText = (body: PresetBody): string => JSON.stringify(body, null, 2);

/** The body a text holds, or why it is not one. Only the JSON syntax and shape (an object) are checked
 *  here; every rule on the fields is the server's. */
export function parseBodyText(text: string): { body: PresetBody } | { error: string } {
  try {
    const v: unknown = JSON.parse(text);
    if (v === null || typeof v !== 'object' || Array.isArray(v)) {
      return { error: 'The body must be a JSON object.' };
    }
    return { body: v as PresetBody };
  } catch (e) {
    return { error: (e as Error).message };
  }
}
