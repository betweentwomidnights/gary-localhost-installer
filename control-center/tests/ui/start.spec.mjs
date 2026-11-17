import { test, expect } from '@playwright/test';

for (const service of ['sa3', 'carey', 'gary', 'stable-audio', 'foundation', 'melodyflow']) {
  test(`${service}: the only downloaded precision needs Start without a Use click`, async ({ page }) => {
    await page.goto(`/tests/ui/?service=${service}&fresh&row`);
    await expect(page.getByRole('combobox', { name: /precision/ }).first()).toHaveValue(service === 'carey' ? 'BF16' : 'F16');
    await expect(page.getByRole('button', { name: 'use', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'start', exact: true })).toBeEnabled();
    expect(await page.evaluate(() => window.testCalls.filter(({ command }) => ['activate_native_runtime', 'start_service'].includes(command)))).toEqual([]);
    await page.getByRole('button', { name: 'start', exact: true }).click();
    await expect(page.getByRole('button', { name: 'stop', exact: true })).toBeVisible();
    expect(await page.evaluate(() => window.testCalls.filter(({ command }) => command === 'start_service'))).toHaveLength(1);
  });
}

test('multiple downloaded precisions require an explicit choice before first start', async ({ page }) => {
  await page.goto('/tests/ui/?service=sa3&fresh&row&all');
  await expect(page.getByRole('button', { name: 'start', exact: true })).toBeDisabled();
  await expect(page.locator('.service-row')).toContainText('multiple generation precisions');
  await page.getByRole('combobox', { name: 'generation tier precision', exact: true }).selectOption('Q4_K_M');
  await page.getByRole('region', { name: 'generation tier', exact: true }).getByRole('button', { name: 'use', exact: true }).click();
  await expect(page.getByRole('button', { name: 'start', exact: true })).toBeEnabled();
  expect(await page.evaluate(() => window.testCalls.find(({ command }) => command === 'activate_native_runtime').args.encoding)).toBe('Q4_K_M');
});

test('a missing shared component explains the disabled Start button', async ({ page }) => {
  await page.goto('/tests/ui/?service=sa3&fresh&row&missing-shared');
  await expect(page.getByRole('button', { name: 'start', exact: true })).toBeDisabled();
  await expect(page.locator('.service-row')).toContainText('required shared components');
});

test('pending activation prevents duplicate starts and failures remain visible', async ({ page }) => {
  await page.goto('/tests/ui/?service=sa3&fresh&row&hold&fail-start');
  await page.getByRole('button', { name: 'start', exact: true }).click();
  await expect(page.getByRole('button', { name: 'starting...', exact: true })).toBeDisabled();
  await expect.poll(() => page.evaluate(() => typeof window.finishStart)).toBe('function');
  await page.evaluate(() => window.finishStart());
  await expect(page.getByRole('alert')).toContainText('Native model verification failed');
  await expect(page.getByRole('button', { name: 'stop', exact: true })).toHaveCount(0);
});

test('an existing Python environment keeps its usual Start action', async ({ page }) => {
  await page.goto('/tests/ui/?service=sa3&legacy&row');
  await expect(page.getByRole('button', { name: 'start', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'start', exact: true }).click();
  expect(await page.evaluate(() => window.testCalls.some(({ command }) => command === 'activate_native_runtime'))).toBe(false);
});
