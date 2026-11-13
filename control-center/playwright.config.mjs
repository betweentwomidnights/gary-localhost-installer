import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/ui',
  outputDir: '../tmp/ui-test-results',
  projects: [
    { name: 'chromium', use: { browserName: 'chromium' } },
    { name: 'webkit', use: { browserName: 'webkit' } },
  ],
  use: { baseURL: 'http://127.0.0.1:1421', viewport: { width: 1024, height: 768 }, screenshot: 'only-on-failure' },
  webServer: {
    command: 'npx vite --config tests/ui/vite.config.mjs --host 127.0.0.1 --port 1421',
    url: 'http://127.0.0.1:1421/tests/ui/',
    reuseExistingServer: false,
  },
});
