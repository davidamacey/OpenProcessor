import { describe, expect, it } from 'vitest';
import { extractBalanced, extractFunction, normalize, stripComments } from './sourceScan';

describe('stripComments', () => {
  it('removes block, HTML and line comments', () => {
    const src = `/* block */\n<!-- html -->\nconst x = 1; // line\nconst url = 'http://x';`;
    const out = stripComments(src);
    expect(out).not.toContain('block');
    expect(out).not.toContain('html');
    expect(out).not.toContain('// line');
    // A `://` inside a real string literal must survive.
    expect(out).toContain("'http://x'");
  });
});

describe('normalize', () => {
  it('collapses whitespace so reformatting does not change the result', () => {
    const a = 'function foo() {\n  return 1;\n}';
    const b = 'function foo() {\n    return 1;\n}\n\n';
    expect(normalize(a)).toBe(normalize(b));
  });

  it('strips comments before collapsing', () => {
    expect(normalize('const x = 1; // comment\n')).toBe('const x = 1;');
  });
});

describe('extractFunction', () => {
  it('extracts a function body regardless of indentation width', () => {
    const twoSpace = 'function foo() {\n  return 1;\n}';
    const fourSpace = 'function foo() {\n    return 1;\n}';
    expect(extractFunction(twoSpace, 'foo')).toContain('return 1;');
    expect(extractFunction(fourSpace, 'foo')).toContain('return 1;');
  });

  it('survives a closing brace that is not alone on its own line', () => {
    const wrapped = 'function foo() { return 1; }';
    expect(extractFunction(wrapped, 'foo')).toBe('function foo() { return 1; }');
  });

  it('balances nested braces correctly', () => {
    const src = 'function foo() {\n  if (true) {\n    return 1;\n  }\n  return 2;\n}';
    const extracted = extractFunction(src, 'foo');
    expect(extracted).toContain('return 2;');
    expect(extracted?.trim().endsWith('}')).toBe(true);
  });

  it('matches an async function', () => {
    const src = 'async function bar() {\n  await x();\n}';
    expect(extractFunction(src, 'bar')).toContain('await x();');
  });

  it('returns null when the function is not present', () => {
    expect(extractFunction('const x = 1;', 'missing')).toBeNull();
  });

  it('does not bleed into a later, unrelated function', () => {
    const src = 'function foo() {\n  return 1;\n}\nfunction bar() {\n  return 2;\n}';
    const extracted = extractFunction(src, 'foo');
    expect(extracted).not.toContain('return 2;');
  });
});

describe('extractBalanced', () => {
  it('extracts a non-function-declaration call argument regardless of indentation', () => {
    const src = 'obj.register(\n  async (x) => {\n    doThing(x);\n  },\n);';
    const extracted = extractBalanced(src, /obj\.register\(\s*async \(x\) => \{/);
    expect(extracted).toContain('doThing(x);');
    // Stops at the matching close brace — the trailing `,);` closing the
    // outer call is deliberately NOT included.
    expect(extracted?.trim().endsWith('}')).toBe(true);
    expect(extracted).not.toContain(',);');
  });

  it('returns null when the pattern does not match', () => {
    expect(extractBalanced('const x = 1;', /missing\(/)).toBeNull();
  });
});
