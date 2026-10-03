#!/usr/bin/env node
// Per-file Stryker driver. @stryker-mutator/vitest-runner 10.0.0 does not work
// with vitest 5 (it runs 0 tests per mutant, so every mutant survives), so each
// mutant runs through Stryker's command runner as
// `vitest related <file> --run --bail 1`: a mutant is killed when a test that
// imports the mutated file fails. The command differs per file, hence one
// Stryker run (and generated config) per file.
// Usage: node scripts/mutation.mjs [file ...]   (default: stryker.config.json `mutate`)
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';

const base = JSON.parse(readFileSync('stryker.config.json', 'utf8'));
const args = process.argv.slice(2);
const files = args.length ? args : base.mutate;
const summary = [];
let failed = false;

for (const file of files) {
  const slug = file.replace(/^src\/lib\//, '').replace(/[\\/]/g, '__');
  const reportDir = `reports/mutation/${slug}`;
  mkdirSync(reportDir, { recursive: true });
  const cfg = {
    ...base,
    testRunner: 'command',
    coverageAnalysis: 'off',
    commandRunner: { command: `npx vitest related ${file} --run --bail 1` },
    mutate: [file],
    incremental: false,
    htmlReporter: { fileName: `${reportDir}/index.html` },
    jsonReporter: { fileName: `${reportDir}/mutation.json` },
  };
  delete cfg.$schema;
  delete cfg.vitest;
  delete cfg.incrementalFile;
  const cfgPath = `.stryker-tmp-config-${slug}.json`;
  writeFileSync(cfgPath, JSON.stringify(cfg));
  const started = Date.now();
  const r = spawnSync('npx', ['stryker', 'run', cfgPath], { stdio: 'inherit' });
  const minutes = ((Date.now() - started) / 60000).toFixed(1);
  rmSync(cfgPath, { force: true });
  summary.push(`${file}: exit ${r.status}, ${minutes} min`);
  if (r.status !== 0) failed = true;
}

console.log('\nMutation summary');
for (const line of summary) console.log(`  ${line}`);
process.exit(failed ? 1 : 0);
