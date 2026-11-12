/**
 * One preset document's editor: the preset binding of the shared `ConfigEditor` (the draft, Save with
 * `expected_revision`, the served 409 with "reload" and "keep my edits", the served 422 reasons).
 * The body is edited as JSON text. Only the JSON syntax is checked here; every rule on the fields is
 * the server's, shown when Save is refused.
 */
import { getPreset, putPreset } from '$lib/api_presets';
import { ConfigEditor } from '$lib/config/configEditor.svelte';
import type { PresetBody, PresetDoc } from '$lib/types_presets';
import { bodyToText, parseBodyText } from './presetBodyText';
import { PresetActive, type PresetActiveDeps } from './presetActive.svelte';
import type { PresetKind } from './presetKinds';

export interface PresetEditorDeps extends PresetActiveDeps {
  getPreset: typeof getPreset;
  putPreset: typeof putPreset;
}

export class PresetEditor extends ConfigEditor<
  PresetBody,
  PresetDoc,
  null,
  PresetActive
> {
  /** The body as the operator sees and edits it. */
  bodyText = $state('');
  /** Why `bodyText` is not a JSON object; null while it is one. */
  bodyError = $state<string | null>(null);
  readonly kind: PresetKind;

  constructor(kind: PresetKind, name: string, deps: Partial<PresetEditorDeps> = {}) {
    super(
      name,
      {
        getDoc: (n) => (deps.getPreset ?? getPreset)(kind.id, n),
        update: (n, req) =>
          (deps.putPreset ?? putPreset)(kind.id, n, {
            body: req.body,
            description: req.description ?? '',
            expected_revision: req.expected_revision,
          }),
      },
      new PresetActive(kind, deps),
    );
    this.kind = kind;
  }

  protected override adopt(doc: PresetDoc): void {
    super.adopt(doc);
    this.bodyText = bodyToText(doc.body);
    this.bodyError = null;
  }

  override get dirty(): boolean {
    return super.dirty || this.bodyError !== null;
  }

  override get canSave(): boolean {
    return super.canSave && this.bodyError === null;
  }

  setBodyText(text: string): void {
    if (!this.editable) return;
    this.bodyText = text;
    const parsed = parseBodyText(text);
    if ('error' in parsed) {
      this.bodyError = parsed.error;
      return;
    }
    this.draftBody = parsed.body;
    this.bodyError = null;
  }

  /** Rewrites the text with two-space indentation (when it parses). */
  format(): void {
    if (this.bodyError === null) this.bodyText = bodyToText(this.draftBody);
  }
}

export function createPresetEditor(
  kind: PresetKind,
  name: string,
  deps: Partial<PresetEditorDeps> = {},
): PresetEditor {
  return new PresetEditor(kind, name, deps);
}
