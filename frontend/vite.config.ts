import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
// vitest/config re-exports vite's defineConfig with the `test` block typed.
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  server: { port: 5173, host: '0.0.0.0' },
  preview: { port: 5181, host: '0.0.0.0' },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts'],
  },
});
