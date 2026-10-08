/**
 * The VLM form adapter: a schema row as the profile editor's row (nothing
 * invented: `applies_when` is null, absent facts get the editor's neutral
 * values), both `choices_from` lists resolved from what was served (the
 * `secret_refs` list from the registry, `vlm_catalog` from the catalog)
 * and never mixed up, a missing list = a text input, the empty choice
 * first in the picker, and a cleared pick becoming the served
 * `empty_choice.id`.
 */
import { describe, expect, it } from 'vitest';
import { selectOptions } from '$lib/profiles/profileFields';
import { catalogFixture, listFixture, schemaFixture } from '$lib/test/fixtures/vlm';
import {
  bodyFromDefaults,
  normalizeFieldValue,
  vlmChoices,
  vlmFieldAsProfileField,
  vlmSchemaAsProfileSchema,
} from './vlmFields';

const row = (field: string) => schemaFixture().fields.find((f) => f.field === field)!;

describe('vlmFieldAsProfileField', () => {
  it('maps the served row and adds only applies_when: null', () => {
    const f = vlmFieldAsProfileField(row('max_images_per_call'));
    expect(f).toMatchObject({
      field: 'max_images_per_call',
      label: 'Images per call',
      group: 'limits',
      type: 'int',
      default: 8,
      min: 1,
      max: 32,
      advanced: false,
      applies_when: null,
      choices_from: null,
      empty_choice: null,
      enum: null,
      help: '',
    });
  });

  it('carries enum choices as {id, label} and the served empty_choice', () => {
    expect(vlmFieldAsProfileField(row('json_mode')).enum).toEqual([
      { id: 'auto', label: 'Automatic' },
      { id: 'on', label: 'Always on' },
      { id: 'off', label: 'Off' },
    ]);
    expect(vlmFieldAsProfileField(row('api_key_ref')).empty_choice).toEqual({
      id: null,
      label: 'No key',
    });
  });

  it('keeps the served groups and field order', () => {
    const s = vlmSchemaAsProfileSchema(schemaFixture());
    expect(s.groups.map((g) => g.id)).toEqual(['connection', 'limits']);
    expect(s.fields.map((f) => f.field)).toEqual(
      schemaFixture().fields.map((f) => f.field),
    );
  });
});

describe('vlmChoices', () => {
  it('secret_refs come from the registry list, never the catalog', () => {
    const c = vlmChoices(row('api_key_ref'), listFixture(), catalogFixture());
    expect(c).toEqual([
      { id: 'CLOUD_VLM_KEY', label: 'CLOUD_VLM_KEY' },
      { id: 'MISSING_KEY', label: 'MISSING_KEY' },
    ]);
  });

  it('vlm_catalog comes from the catalog entries, never the secret refs', () => {
    const c = vlmChoices(row('catalog_id'), listFixture(), catalogFixture());
    expect(c).toEqual([
      { id: 'vision-7b', label: 'Vision 7B' },
      { id: 'vision-30b', label: 'Vision 30B' },
    ]);
  });

  it('a list that has not been served is null (the row is a text input)', () => {
    expect(vlmChoices(row('api_key_ref'), null, catalogFixture())).toBeNull();
    expect(vlmChoices(row('catalog_id'), listFixture(), null)).toBeNull();
  });

  it('a row with no choices_from has none', () => {
    expect(vlmChoices(row('model'), listFixture(), catalogFixture())).toBeNull();
  });

  it('a null-id choice is not repeated (the empty_choice carries it)', () => {
    const list = listFixture();
    list.secret_refs.push({
      ref: '',
      present: false,
      choice: { id: null, label: 'No key' },
    });
    expect(vlmChoices(row('api_key_ref'), list, null)!.map((c) => c.id)).toEqual([
      'CLOUD_VLM_KEY',
      'MISSING_KEY',
    ]);
  });

  it('the picker lists the empty choice first, then the served list', () => {
    const f = vlmFieldAsProfileField(row('api_key_ref'));
    const opts = selectOptions(
      f,
      vlmChoices(row('api_key_ref'), listFixture(), null)!,
      null,
    );
    expect(opts.map((o) => o.label)).toEqual(['No key', 'CLOUD_VLM_KEY', 'MISSING_KEY']);
  });
});

describe('bodyFromDefaults / normalizeFieldValue', () => {
  it('seeds a new body from each served default', () => {
    expect(bodyFromDefaults(schemaFixture())).toMatchObject({
      base_url: '',
      api_key_ref: null,
      allow_external: false,
      json_mode: 'auto',
      max_images_per_call: 8,
      timeout_s: 240,
    });
  });

  it("a cleared pick is the row's served empty_choice id (null), other values pass", () => {
    expect(normalizeFieldValue(schemaFixture(), 'api_key_ref', '')).toBeNull();
    expect(normalizeFieldValue(schemaFixture(), 'api_key_ref', 'CLOUD_VLM_KEY')).toBe(
      'CLOUD_VLM_KEY',
    );
    // A row with no empty_choice keeps an empty string.
    expect(normalizeFieldValue(schemaFixture(), 'model', '')).toBe('');
    expect(normalizeFieldValue(null, 'api_key_ref', '')).toBe('');
  });
});
