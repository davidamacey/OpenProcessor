/**
 * MapLibre starts its GeoJSON worker as a module worker from a `.mjs` file. Browsers refuse a
 * module script served as application/octet-stream, which is what the stock nginx mime.types
 * gives `.mjs`: the worker then fails to load and the map draws no points. The served nginx
 * config must therefore give `.mjs` a JavaScript type.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const nginxConf = readFileSync(path.resolve(process.cwd(), 'nginx.conf'), 'utf-8');

describe('nginx.conf static types', () => {
  it('serves .mjs as JavaScript', () => {
    const block = /location\s+~\*\s+\\\.mjs\$\s*\{([^}]*)\}/.exec(nginxConf);
    expect(block, 'a location for .mjs files').not.toBeNull();
    expect(block![1]).toMatch(/default_type\s+text\/javascript;/);
  });
});
