import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
// vitest/config re-exports vite's defineConfig with the `test` block typed.
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  // Vite's default envPrefix is 'VITE_' only, which silently drops
  // PUBLIC_TRITON_API_URL (api.ts reads import.meta.env.PUBLIC_TRITON_API_URL
  // directly) even though CLAUDE.md documents it as the way to point local
  // dev at a remote openprocessor.
  envPrefix: ['VITE_', 'PUBLIC_'],
  server: { port: 5173, host: '0.0.0.0' },
  preview: { port: 5181, host: '0.0.0.0' },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts'],
  },
});
