import js from '@eslint/js';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import ts from 'typescript-eslint';

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
  ...svelte.configs['flat/recommended'],
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
    },
  },
  {
    files: ['**/*.svelte'],
    languageOptions: { parserOptions: { parser: ts.parser } },
  },
);
