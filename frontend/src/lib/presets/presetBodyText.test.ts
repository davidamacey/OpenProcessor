import { describe, expect, it } from 'vitest';
import { bodyToText, parseBodyText } from './presetBodyText';

describe('parseBodyText', () => {
  it('round-trips a body through its indented text', () => {
    const body = { k: 3, nested: { a: [1, 2] } };
    expect(parseBodyText(bodyToText(body))).toEqual({ body });
  });

  it('names the JSON syntax error', () => {
    const r = parseBodyText('{"k": ');
    expect('error' in r && r.error.length > 0).toBe(true);
  });

  it.each(['[1]', 'null', '"x"', '3'])('refuses %s: the body is an object', (text) => {
    expect(parseBodyText(text)).toEqual({ error: 'The body must be a JSON object.' });
  });
});
