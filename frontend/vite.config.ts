import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
// vitest/config re-exports vite's defineConfig with the `test` block typed.
import { defineConfig } from 'vitest/config';

import svelteKitOptions from './sveltekit.options.js';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit(svelteKitOptions)],
  // Vite's default envPrefix is 'VITE_' only, which silently drops
  // PUBLIC_TRITON_API_URL (api.ts reads import.meta.env.PUBLIC_TRITON_API_URL
  // directly) even though CLAUDE.md documents it as the way to point local
  // dev at a remote OpenProcessor.
  envPrefix: ['VITE_', 'PUBLIC_'],
  server: { port: 5173, host: '0.0.0.0' },
  preview: { port: 5181, host: '0.0.0.0' },
  // Vitest otherwise resolves Svelte's server build
  // (svelte/src/index-server.js), which throws
  // `lifecycle_function_unavailable` for `mount`/`unmount` — component
  // mount tests need the browser build under jsdom instead (test-audit
  // -2026-09-24.md §2.4). Scoped to `process.env.VITEST` so `vite build`
  // and `vite dev` still resolve the server/client builds normally.
  resolve: process.env.VITEST ? { conditions: ['browser'] } : undefined,
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts'],
    setupFiles: ['src/lib/test/setup.ts'],
  },
});
