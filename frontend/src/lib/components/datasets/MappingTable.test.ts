/**
 * The class-mapping step, mounted (§7.12 item 1, W10.5): one row per
 * served class with its served suggestion label, the highlight driven by
 * the served `resolved`, and controls that write names/ids the operator
 * picked — never a dataset index.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import MappingTable from './MappingTable.svelte';
import { ImportWizard } from '$lib/datasets/importWizardController.svelte';
import { formatsFixture, previewFixture } from '$lib/test/fixtures/datasetImport';
import type { RegistryClass } from '$lib/types';
import type { DatasetPreview } from '$lib/types_import';

const CLASSES = [
  { id: 2, name: 'widget' },
  { id: 5, name: 'gadget' },
] as RegistryClass[];

let target: HTMLDivElement;
let instance: Record<string, unknown> | undefined;

function render(preview: DatasetPreview) {
  const wizard = new ImportWizard({ previewDataset: vi.fn().mockResolvedValue(preview) });
  wizard.preview = preview;
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(MappingTable, {
    target,
    props: { wizard, formats: formatsFixture(), classes: CLASSES },
  });
  flushSync();
  return wizard;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

function row(name: string): HTMLElement {
  return target.querySelector(`[data-dataset-class="${name}"]`) as HTMLElement;
}

describe('MappingTable', () => {
  it('renders one row per served class with the served match label', () => {
    render(previewFixture());
    expect(target.querySelectorAll('[data-testid="mapping-row"]')).toHaveLength(2);
    expect(
      row('Widget').querySelector('[data-testid="suggestion-chip"]')?.textContent,
    ).toContain('Same name, different case');
    expect(row('sprocket').textContent).toContain('No match');
    expect(row('sprocket').textContent).toContain('Create class: sprocket');
  });

  it('highlights rows the served preview does not resolve, and shows resolved targets', () => {
    const p = previewFixture();
    p.classes[0]!.resolved = {
      dataset_class: 'Widget',
      kind: 'item',
      class_id: 2,
      class_name: 'widget',
    };
    render(p);
    expect(row('Widget').dataset.unmapped).toBeUndefined();
    expect(
      row('Widget').querySelector('[data-testid="resolved"]')?.textContent,
    ).toContain('widget');
    expect(row('sprocket').dataset.unmapped).toBe('true');
    expect(row('sprocket').textContent).toContain('not mapped yet');
  });

  it('"map" shows a class picker that writes the picked registry id', () => {
    const w = render(previewFixture());
    const action = row('Widget').querySelector(
      'select[aria-label="Action for Widget"]',
    ) as HTMLSelectElement;
    expect(
      row('Widget').querySelector('select[aria-label="Class for Widget"]'),
    ).toBeNull();
    action.value = 'map';
    action.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    const picker = row('Widget').querySelector(
      'select[aria-label="Class for Widget"]',
    ) as HTMLSelectElement;
    expect([...picker.options].map((o) => o.textContent)).toEqual([
      'Pick a class…',
      'widget',
      'gadget',
    ]);
    picker.value = '5';
    picker.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(w.mappingEntries()).toEqual([
      { dataset_class: 'Widget', action: 'map', class_id: 5 },
    ]);
  });

  it('"create" shows a name input; "Use suggestion" copies the served suggestion', () => {
    const w = render(previewFixture());
    const btn = [...row('sprocket').querySelectorAll('button')].find(
      (b) => b.textContent === 'Use suggestion',
    )!;
    btn.click();
    flushSync();
    const input = row('sprocket').querySelector(
      'input[aria-label="New class name for sprocket"]',
    ) as HTMLInputElement;
    expect(input.value).toBe('sprocket');
    input.value = 'sprocket_b';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(w.mappingEntries()).toEqual([
      { dataset_class: 'sprocket', action: 'create', new_class_name: 'sprocket_b' },
    ]);
  });

  it('the dataset id is only a label; the index hint shows only when the served issue fires', () => {
    render(previewFixture());
    expect(row('Widget').textContent).toContain('dataset id 0');
    expect(row('Widget').querySelector('[data-testid="index-hint"]')).toBeNull();
    if (instance) unmount(instance);
    instance = undefined;
    target.remove();
    const p = previewFixture();
    p.issues.push({
      code: 'class_index_name_mismatch',
      severity: 'info',
      blocking: false,
      bypassable: false,
      message: 'The old index path would have mislabeled 1 class.',
      count: 1,
      samples: [],
    });
    render(p);
    expect(
      row('Widget').querySelector('[data-testid="index-hint"]')?.textContent,
    ).toContain('gadget');
  });
});
