import { test, expect } from '@playwright/test';

test('LM-only install identifies the missing audio dependencies visibly', async ({ page }) => {
  await page.goto('/tests/ui/?captioning');
  const button = page.getByRole('button', { name: 'auto-label all', exact: true });
  await expect(button).toBeDisabled();
  await expect(button).toHaveAttribute('title', /audio model.*shared VAE/);
  await expect(page.getByRole('status')).toContainText('audio model (DiT)');
  await expect(page.getByRole('status')).toContainText('shared VAE');
  await expect(page.getByRole('status')).not.toContainText('Download the ACE-Step 1.7B');
});

test('complete prerequisites enable auto-labeling', async ({ page }) => {
  await page.goto('/tests/ui/?captioning&caption-ready');
  await expect(page.getByRole('button', { name: 'auto-label all', exact: true })).toBeEnabled();
  await expect(page.getByRole('status')).toHaveCount(0);
});

test('a failed inventory check is not reported as a missing model', async ({ page }) => {
  await page.goto('/tests/ui/?captioning&availability-error');
  await expect(page.getByRole('status')).toContainText('Cannot check auto-label setup: Inventory unavailable');
});
