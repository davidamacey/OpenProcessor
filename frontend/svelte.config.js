import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

const config = {
  preprocess: vitePreprocess(),
  kit: {
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
  },
};

export default config;
