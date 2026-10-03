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
  writeFileSync(conf, 'location ~ ^__API_PREFIX__/ { proxy_pass __API_UPSTREAM__; }');
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
