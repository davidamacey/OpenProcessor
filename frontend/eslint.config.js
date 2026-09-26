import js from '@eslint/js';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import ts from 'typescript-eslint';
import svelteConfig from './svelte.config.js';

/** Svelte 5 rune globals — visible in .svelte.ts modules too. */
const runes = {
  $state: 'readonly',
  $derived: 'readonly',
  $effect: 'readonly',
  $props: 'readonly',
  $bindable: 'readonly',
  $inspect: 'readonly',
  $host: 'readonly',
};

export default ts.config(
  {
    ignores: [
      'build/',
      '.svelte-kit/',
      'node_modules/',
      'diagnostics/',
      'static/',
      'scripts/',
      // The stubbed e2e suite's own gitignored per-project venv
      // (scripts/run-e2e.mjs) — vendors Playwright's driver JS. Once it
      // exists on disk (after running `npm run test:e2e` once), `npm run
      // lint` would otherwise scan thousands of unrelated vendored files.
      'e2e/.venv/',
      // Local agent worktrees, gitignored evidence and Stryker's sandbox copies.
      '.claude/',
      'artifacts_local/',
      '.stryker-tmp/',
      'reports/',
      // Vendored, generated verbatim from OpenProcessor's contracts/ — see
      // contracts/openprocessor/SOURCE.md. Not ours to lint or format.
      'contracts/openprocessor/',
      // Docusaurus docs site — its own package.json/tsconfig/eslint story
      // (or none), not part of this SvelteKit app's lint surface.
      'docs-site/',
    ],
  },
  js.configs.recommended,
  ...ts.configs.recommended,
  ...svelte.configs.recommended,
  {
    languageOptions: {
      globals: { ...globals.browser, ...globals.node, ...runes },
    },
    rules: {
      // The codebase leans on `catch (e) { toast((e as Error).message) }` and
      // narrows API payloads by hand; `any` is not the tool of choice here,
      // but unused-vars should stay advisory for intentional _-prefixed args.
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_', caughtErrors: 'none' },
      ],
      // New in eslint-plugin-svelte 3's recommended config (PR #15 major
      // bump). Both are real, pre-existing debt across ~20 files (raw
      // <a href>/goto()/replaceState() calls that predate SvelteKit's
      // resolve() helper, and a handful of native Set/Map instances in
      // reactive scope that should be SvelteSet/SvelteMap) — not something
      // to silently fix as a drive-by inside a dependency bump. Downgraded
      // to warn for now, mirroring transcribe-app/frontend's
      // eslint.config.js, which hit the exact same two rules on the same
      // bump; ratchet back to 'error' as each call site is migrated.
      'svelte/no-navigation-without-resolve': 'warn',
      'svelte/prefer-svelte-reactivity': 'warn',
    },
  },
  {
    // eslint-plugin-svelte 3 + typescript-eslint 8's parser needs the
    // project's own svelte.config.js (for preprocessors) to parse both
    // .svelte files and .svelte.ts/.svelte.js rune modules — without it,
    // typescript-eslint's parser chokes on Svelte 5 rune syntax in a
    // `.svelte.ts` file ("Parsing error: Unexpected token {").
    files: ['**/*.svelte', '**/*.svelte.ts', '**/*.svelte.js'],
    languageOptions: { parserOptions: { parser: ts.parser, svelteConfig } },
  },
);
