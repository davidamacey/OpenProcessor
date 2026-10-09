import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// SvelteKit 3 reads its options from the sveltekit() plugin in vite.config.ts;
// eslint imports this same object so the parser sees the same preprocessors.
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
  alias: {
    $lib: 'src/lib',
    $components: 'src/lib/components',
    $stores: 'src/lib/stores',
    // The API contract the backend generates; one copy, in this repository.
    $contracts: '../contracts',
  },
};
