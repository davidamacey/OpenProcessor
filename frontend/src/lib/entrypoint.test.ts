/**
 * docker-entrypoint.sh validates every value it substitutes into sed and JS
 * string literals. The script is run against a temp tree (its two absolute
 * target paths rewritten), so no container is needed.
 */
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const script = readFileSync(path.resolve(process.cwd(), 'docker-entrypoint.sh'), 'utf-8');

function run(env: Record<string, string>) {
  const dir = mkdtempSync(path.join(tmpdir(), 'entrypoint-'));
  const html = path.join(dir, 'html');
  mkdirSync(html);
  writeFileSync(
    path.join(html, 'a.js'),
    'const B="__RUNTIME__";const P="__API_PREFIX__";',
  );
  const conf = path.join(dir, 'default.conf');
  writeFileSync(
    conf,
    'location ~ ^__API_PREFIX__/ { proxy_pass __API_UPSTREAM__; } set $d __DOCS_UPSTREAM__;',
  );
  const patched = script
    .replaceAll('/usr/share/nginx/html', html)
    .replaceAll('/etc/nginx/conf.d/default.conf', conf);
  const r = spawnSync('sh', ['-c', patched], {
    env: { PATH: process.env.PATH ?? '', ...env },
    encoding: 'utf-8',
  });
  return {
    status: r.status,
    stderr: r.stderr,
    js: readFileSync(path.join(html, 'a.js'), 'utf-8'),
    conf: readFileSync(conf, 'utf-8'),
  };
}

describe('docker-entrypoint.sh input validation', () => {
  it('control: the default (empty URL, default prefix) substitutes', () => {
    const r = run({});
    expect(r.status).toBe(0);
    expect(r.js).toBe('const B="";const P="/curation";');
  });

  it('control: an explicit http URL and prefix substitute', () => {
    const r = run({
      PUBLIC_TRITON_API_URL: 'http://h:4603',
      PUBLIC_API_PREFIX: 'api/v1/',
    });
    expect(r.status).toBe(0);
    expect(r.js).toBe('const B="http://h:4603";const P="/api/v1";');
  });

  it('substitutes DOCS_UPSTREAM into the nginx conf, defaulting to http://docs:8080', () => {
    expect(run({}).conf).toContain('set $d http://docs:8080;');
    expect(run({ DOCS_UPSTREAM: 'http://d:9' }).conf).toContain('set $d http://d:9;');
  });

  it('nginx.conf carries every placeholder the entrypoint substitutes', () => {
    const conf = readFileSync(path.resolve(process.cwd(), 'nginx.conf'), 'utf-8');
    expect(conf).toContain('set $docs_upstream __DOCS_UPSTREAM__;');
    for (const loc of [
      '= /docs',
      '= /redoc',
      '= /openapi.json',
      '= /docs/oauth2-redirect',
      '^~ /docs-assets/',
      '^~ /OpenProcessor/',
    ]) {
      expect(conf).toContain(`location ${loc} {`);
    }
  });

  // The docs embed their own architecture diagrams in same-origin iframes, so the proxied docs
  // location must allow same-origin framing; the app and API locations keep the strict DENY.
  it('lets the proxied docs frame their own diagrams (SAMEORIGIN) and denies framing elsewhere', () => {
    const conf = readFileSync(path.resolve(process.cwd(), 'nginx.conf'), 'utf-8');
    const docs = conf.match(/location \^~ \/OpenProcessor\/ \{([^}]*)\}/);
    expect(docs).not.toBeNull();
    expect(docs![1]).toContain('add_header X-Frame-Options "SAMEORIGIN" always;');
    expect(docs![1]).toContain('add_header X-Content-Type-Options "nosniff" always;');
    expect(
      readFileSync(path.resolve(process.cwd(), 'nginx-security-headers.conf'), 'utf-8'),
    ).toContain('add_header X-Frame-Options "DENY" always;');
  });

  // The backend builds service links from the host the client used, so every
  // proxied location must carry the original host (with its port) through.
  it('forwards the original host and scheme to the API on every API and docs location', () => {
    const conf = readFileSync(path.resolve(process.cwd(), 'nginx.conf'), 'utf-8');
    const headers = [
      'proxy_set_header Host $http_host;',
      'proxy_set_header X-Forwarded-Host $http_host;',
      'proxy_set_header X-Forwarded-Proto $scheme;',
      'proxy_set_header X-Real-IP $remote_addr;',
      'proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;',
    ];
    const blocks = [...conf.matchAll(/location ([^{]+)\{([^}]*)\}/g)].filter((m) =>
      m[2]!.includes('proxy_pass $api_upstream'),
    );
    // ingest, dataset uploads, general API, five docs paths
    expect(blocks.length).toBe(8);
    for (const [, loc, body] of blocks) {
      for (const h of headers) expect(body, `${loc} lacks ${h}`).toContain(h);
    }
    expect(conf).not.toContain('proxy_set_header Host $host;\n        proxy_pass $api');
  });

  for (const bad of ['ftp://d', 'http://d|x', 'http://d"x', 'http://d&x']) {
    it(`rejects DOCS_UPSTREAM=${bad} with a clear message`, () => {
      const r = run({ DOCS_UPSTREAM: bad });
      expect(r.status).not.toBe(0);
      expect(r.stderr).toContain('DOCS_UPSTREAM');
    });
  }

  for (const bad of ['http://h:1/a&b', 'http://h|x', 'ftp://h', 'http://h"x']) {
    it(`rejects PUBLIC_TRITON_API_URL=${bad} with a clear message`, () => {
      const r = run({ PUBLIC_TRITON_API_URL: bad });
      expect(r.status).not.toBe(0);
      expect(r.stderr).toContain('PUBLIC_TRITON_API_URL');
    });
  }

  for (const bad of ['/a&b', '/a|b', '/a"b', '/a b']) {
    it(`rejects PUBLIC_API_PREFIX=${bad} with a clear message`, () => {
      const r = run({ PUBLIC_API_PREFIX: bad });
      expect(r.status).not.toBe(0);
      expect(r.stderr).toContain('PUBLIC_API_PREFIX');
    });
  }
});
