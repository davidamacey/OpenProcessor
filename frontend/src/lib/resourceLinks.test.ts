import { describe, expect, it } from 'vitest';
import { resourceLinks, monitoringResourceLinks } from './resourceLinks';

const NONE = { grafana: null, prometheus: null, opensearch_dashboards: null };

describe('resourceLinks', () => {
  it('always offers the same-origin docs and API reference paths', () => {
    expect(resourceLinks(NONE, null).map((l) => l.href)).toEqual([
      '/cropwright/',
      '/docs',
      '/redoc',
      '/openapi.json',
    ]);
  });

  it('adds served monitoring links and MLflow after them, in order', () => {
    const links = resourceLinks(
      { ...NONE, grafana: 'http://g:3000', opensearch_dashboards: 'https://os/' },
      'http://mlf:5000',
    );
    expect(links.slice(4).map((l) => [l.key, l.href])).toEqual([
      ['grafana', 'http://g:3000'],
      ['opensearch_dashboards', 'https://os/'],
      ['mlflow', 'http://mlf:5000'],
    ]);
  });

  it('drops null and non-http(s) served URLs', () => {
    expect(
      monitoringResourceLinks({
        ...NONE,
        grafana: 'javascript:alert(1)',
        prometheus: '',
      }),
    ).toEqual([]);
    expect(monitoringResourceLinks(undefined)).toEqual([]);
  });
});
