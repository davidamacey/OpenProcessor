import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// SvelteKit 3 reads its options from the sveltekit() plugin in vite.config.ts;
// eslint imports this same object so the parser sees the same preprocessors.
// The `$lib`, `$components`, `$stores` and `$contracts` import aliases are not
// kit options (`kit.alias` is deprecated): vite.config.ts resolves them for
// vite and vitest, tsconfig.json `paths` for the type checker.
export default {
  preprocess: vitePreprocess(),
  adapter: adapter({
    pages: 'build',
    assets: 'build',
    fallback: 'index.html',
    precompress: false,
    strict: true,
  }),
  // Poll _app/version.json so the `updated` store flips after a deploy;
  // SvelteKit then does a full-page load on the next failed navigation.
  version: { pollInterval: 60_000 },
};
