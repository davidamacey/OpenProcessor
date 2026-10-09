import { describe, expect, it } from 'vitest';
import { DOCS_ENTRY, resourceViews, safeHref } from './resourceLinks';
import type { ResourceLink } from './curationSettings';

const link = (o: Partial<ResourceLink>): ResourceLink => ({
  id: 'grafana',
  label: 'Grafana',
  url: 'http://h:3000',
  kind: 'service',
  status: 'configured',
  hint: 'help',
  reachable: null,
  ...o,
});

describe('safeHref', () => {
  it('accepts absolute http(s) URLs and single-slash root-relative paths', () => {
    expect(safeHref('http://h:3000/x')).toBe('http://h:3000/x');
    expect(safeHref('https://h/')).toBe('https://h/');
    expect(safeHref('/docs')).toBe('/docs');
    expect(safeHref('/openapi.json')).toBe('/openapi.json');
  });

  it('rejects script, data, protocol-relative and bare values', () => {
    for (const bad of [
      'javascript:alert(1)',
      'data:text/html,x',
      '//evil.example/x',
      '/\\evil.example',
      'docs',
      'ftp://h/x',
      '',
      null,
      undefined,
    ]) {
      expect(safeHref(bad)).toBeNull();
    }
  });
});

describe('resourceViews', () => {
  it('puts the client-owned Documentation entry first, then the served list in order', () => {
    const v = resourceViews([
      link({ id: 'swagger', label: 'Swagger', url: '/docs', kind: 'docs' }),
      link({}),
    ]);
    expect(v.map((x) => [x.id, x.href])).toEqual([
      [DOCS_ENTRY.id, '/OpenProcessor/docs/cropwright/getting-started/introduction'],
      ['swagger', '/docs'],
      ['grafana', 'http://h:3000'],
    ]);
  });

  it('with no served list is only Documentation, never a guess', () => {
    expect(resourceViews([]).map((x) => x.id)).toEqual([DOCS_ENTRY.id]);
  });

  it('a null url is a muted not-configured row carrying the served hint', () => {
    const [, g] = resourceViews([
      link({ url: null, status: 'not_configured', hint: 'set OP_GRAFANA_URL' }),
    ]);
    expect(g).toMatchObject({
      href: null,
      note: 'not configured',
      hint: 'set OP_GRAFANA_URL',
    });
  });

  it('an unsafe url yields no href', () => {
    const [, g] = resourceViews([link({ url: 'javascript:alert(1)' })]);
    expect(g!.href).toBeNull();
  });

  it('flags not running only for reachable === false', () => {
    const v = resourceViews([
      link({ id: 'a', reachable: false }),
      link({ id: 'b', reachable: true }),
      link({ id: 'c', reachable: null }),
    ]);
    expect(v.slice(1).map((x) => x.notRunning)).toEqual([true, false, false]);
  });
});
