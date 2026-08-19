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
