import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  root: fileURLToPath(new URL('../..', import.meta.url)),
  plugins: [svelte()],
  optimizeDeps: { entries: ['tests/ui/index.html'] },
  server: { strictPort: true },
});
