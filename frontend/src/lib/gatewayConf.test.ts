/**
 * Gateway mode (OP_GATEWAY_SUBPATHS): the entrypoint renders the sub-path
 * locations for the four monitoring UIs into nginx's gateway.d, and removes
 * them otherwise. Runs the real docker-entrypoint.sh against the real
 * nginx.conf + nginx-gateway.conf in a temp tree, so no container is needed.
 */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const root = process.cwd();
const script = readFileSync(path.resolve(root, 'docker-entrypoint.sh'), 'utf-8');
const nginxConf = readFileSync(path.resolve(root, 'nginx.conf'), 'utf-8');
const gatewayConf = readFileSync(path.resolve(root, 'nginx-gateway.conf'), 'utf-8');

function render(env: Record<string, string>) {
  const dir = mkdtempSync(path.join(tmpdir(), 'gateway-'));
  const html = path.join(dir, 'html');
  const gatewayDir = path.join(dir, 'gateway.d');
  const snippets = path.join(dir, 'snippets');
  mkdirSync(html);
  mkdirSync(gatewayDir);
  mkdirSync(snippets);
  const conf = path.join(dir, 'default.conf');
  writeFileSync(conf, nginxConf);
  writeFileSync(path.join(snippets, 'gateway-subpaths.conf'), gatewayConf);
  const patched = script
    .replaceAll('/usr/share/nginx/html', html)
    .replaceAll('/etc/nginx/conf.d/default.conf', conf)
    .replaceAll('/etc/nginx/gateway.d', gatewayDir)
    .replaceAll(
      '/etc/nginx/snippets/gateway-subpaths.conf',
      path.join(snippets, 'gateway-subpaths.conf'),
    );
  const r = spawnSync('sh', ['-c', patched], {
    env: { PATH: process.env.PATH ?? '', ...env },
    encoding: 'utf-8',
  });
  const out = path.join(gatewayDir, 'gateway-subpaths.conf');
  return {
    status: r.status,
    stderr: r.stderr,
    conf: readFileSync(conf, 'utf-8'),
    gateway: existsSync(out) ? readFileSync(out, 'utf-8') : null,
  };
}

function locations(text: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const m of text.matchAll(/^\s*location ([^{]+)\{([^}]*)\}/gm))
    out.set(m[1]!.trim(), m[2]!);
  return out;
}

const FORWARD = [
  'proxy_set_header Host $http_host;',
  'proxy_set_header X-Forwarded-Host $http_host;',
  'proxy_set_header X-Forwarded-Proto $forwarded_proto;',
  'proxy_set_header X-Real-IP $remote_addr;',
  'proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;',
];

describe('gateway sub-path locations', () => {
  it('nginx.conf includes the gateway directory', () => {
    expect(nginxConf).toContain('include /etc/nginx/gateway.d/*.conf;');
  });

  for (const off of [
    {},
    { OP_GATEWAY_SUBPATHS: 'false' },
    { OP_GATEWAY_SUBPATHS: '' },
  ] as Record<string, string>[]) {
    it(`answers 404 on the four sub-paths and proxies nothing when disabled (${JSON.stringify(off)})`, () => {
      const r = render(off);
      expect(r.status).toBe(0);
      const locs = locations(r.gateway ?? '');
      for (const sub of ['grafana', 'prometheus', 'dashboards', 'mlflow']) {
        expect(locs.get(`^~ /${sub}/`), sub).toContain('return 404;');
        expect(locs.get(`= /${sub}`), sub).toContain('return 404;');
      }
      expect(r.gateway).not.toContain('proxy_pass');
    });
  }

  for (const on of ['true', 'TRUE', '1', 'yes', 'on']) {
    it(`renders the locations when OP_GATEWAY_SUBPATHS=${on}`, () => {
      const r = render({ OP_GATEWAY_SUBPATHS: on });
      expect(r.status).toBe(0);
      expect(r.gateway).not.toBeNull();
      expect(r.gateway).not.toMatch(/__[A-Z_]+__/);
    });
  }

  it('proxies each UI under its sub-path to the compose service by default', () => {
    const g = render({ OP_GATEWAY_SUBPATHS: 'true' }).gateway!;
    expect(g).toContain('set $grafana_upstream http://grafana:3000;');
    expect(g).toContain('set $prometheus_upstream http://prometheus:9090;');
    expect(g).toContain('set $dashboards_upstream http://opensearch-dashboards:5601;');
    expect(g).toContain('set $mlflow_upstream http://curation-mlflow:5000;');
    const locs = locations(g);
    for (const [loc, upstream] of [
      ['^~ /grafana/', '$grafana_upstream'],
      ['^~ /prometheus/', '$prometheus_upstream'],
      ['^~ /dashboards/', '$dashboards_upstream'],
      ['^~ /mlflow/', '$mlflow_upstream'],
    ] as const) {
      expect(locs.get(loc), loc).toContain(`proxy_pass ${upstream}`);
    }
    for (const bare of ['grafana', 'prometheus', 'dashboards', 'mlflow']) {
      expect(locs.get(`= /${bare}`), bare).toContain(`return 301 /${bare}/;`);
    }
  });

  it('keeps an https X-Forwarded-Proto from an outer TLS proxy instead of overwriting it', () => {
    expect(nginxConf).toMatch(
      /map \$http_x_forwarded_proto \$forwarded_proto \{[^}]*default \$scheme;[^}]*http\s+http;[^}]*https\s+https;/,
    );
    const g = render({ OP_GATEWAY_SUBPATHS: 'true' }).gateway!;
    expect(g).not.toContain('X-Forwarded-Proto $scheme');
  });

  it('forwards Host and X-Forwarded-* on every UI location', () => {
    const g = render({ OP_GATEWAY_SUBPATHS: 'true' }).gateway!;
    const proxied = [...locations(g)].filter(([, body]) => body.includes('proxy_pass'));
    expect(proxied.length).toBeGreaterThanOrEqual(5);
    for (const [loc, body] of proxied) {
      for (const h of FORWARD) {
        // MLflow's DNS-rebinding guard gets its own upstream Host (see below).
        if (loc.includes('/mlflow/') && h === 'proxy_set_header Host $http_host;')
          continue;
        expect(body, `${loc} lacks ${h}`).toContain(h);
      }
    }
  });

  it('upgrades websockets for Grafana Live only, with HTTP/1.1', () => {
    const g = render({ OP_GATEWAY_SUBPATHS: 'true' }).gateway!;
    const locs = locations(g);
    const live = locs.get('^~ /grafana/api/live/');
    expect(live).toBeDefined();
    expect(live).toContain('proxy_http_version 1.1;');
    expect(live).toContain('proxy_set_header Upgrade $http_upgrade;');
    expect(live).toContain('proxy_set_header Connection "upgrade";');
    expect(live).toContain('proxy_pass $grafana_upstream');
    expect(locs.get('^~ /grafana/')).not.toContain('Upgrade');
  });

  it('strips the /mlflow prefix and sends the upstream Host so the allowed-hosts guard passes', () => {
    const mlflow = locations(render({ OP_GATEWAY_SUBPATHS: 'true' }).gateway!).get(
      '^~ /mlflow/',
    )!;
    expect(mlflow).toContain('rewrite ^/mlflow/(.*)$ /$1 break;');
    expect(mlflow).toContain('proxy_set_header Host $proxy_host;');
    expect(mlflow).toContain('proxy_set_header X-Forwarded-Host $http_host;');
  });

  it('keeps the prefix for Grafana, Prometheus and Dashboards (they serve from it)', () => {
    const locs = locations(render({ OP_GATEWAY_SUBPATHS: 'true' }).gateway!);
    for (const loc of ['^~ /grafana/', '^~ /prometheus/', '^~ /dashboards/']) {
      expect(locs.get(loc), loc).not.toContain('rewrite');
    }
  });

  it('honours per-service upstream overrides', () => {
    const g = render({
      OP_GATEWAY_SUBPATHS: 'true',
      GRAFANA_UPSTREAM: 'http://gf:3001',
      MLFLOW_UPSTREAM: 'http://ml:5001',
    }).gateway!;
    expect(g).toContain('set $grafana_upstream http://gf:3001;');
    expect(g).toContain('set $mlflow_upstream http://ml:5001;');
  });

  // Fail closed: a UI location may only reach a single-label compose service
  // name over plain http, never an IP, an FQDN or another scheme.
  for (const bad of [
    'http://10.0.0.5:3000',
    'http://example.com:3000',
    'https://grafana:3000',
    'http://grafana',
    'http://grafana:3000/x',
    'http://127.0.0.1:3000',
    'http://grafana:3000;x',
    'http://[::1]:3000',
  ]) {
    it(`refuses GRAFANA_UPSTREAM=${bad} in gateway mode`, () => {
      const r = render({ OP_GATEWAY_SUBPATHS: 'true', GRAFANA_UPSTREAM: bad });
      expect(r.status).not.toBe(0);
      expect(r.stderr).toContain('GRAFANA_UPSTREAM');
      expect(r.gateway).toBeNull();
    });
  }

  it('every rendered upstream is a bare compose service name', () => {
    const g = render({ OP_GATEWAY_SUBPATHS: 'true' }).gateway!;
    const upstreams = [...g.matchAll(/set \$\w+_upstream (\S+);/g)].map((m) => m[1]!);
    expect(upstreams).toHaveLength(4);
    for (const u of upstreams) expect(u).toMatch(/^http:\/\/[a-z][a-z0-9-]*:\d{1,5}$/);
  });

  it('rejects an unparseable OP_GATEWAY_SUBPATHS rather than guessing', () => {
    const r = render({ OP_GATEWAY_SUBPATHS: 'maybe' });
    expect(r.status).not.toBe(0);
    expect(r.stderr).toContain('OP_GATEWAY_SUBPATHS');
  });

  it('redirects are relative: nginx listens on 8080, the published port differs', () => {
    expect(nginxConf).toContain('absolute_redirect off;');
  });

  it('leaves the API docs locations at the root', () => {
    const locs = locations(nginxConf);
    for (const loc of [
      '= /docs',
      '= /redoc',
      '= /openapi.json',
      '= /docs/oauth2-redirect',
      '^~ /docs-assets/',
    ]) {
      expect(locs.get(loc), loc).toContain('proxy_pass $api_upstream');
    }
  });

  it('no hardcoded published host ports (4604-4609) in the gateway config', () => {
    expect(gatewayConf).not.toMatch(/\b46\d\d\b/);
  });
});
