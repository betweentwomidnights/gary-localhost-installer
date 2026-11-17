import { test, expect } from '@playwright/test';

const services = ['sa3', 'yuey', 'stable-audio', 'foundation', 'melodyflow', 'carey', 'gary'];
const writes = (calls) => calls.filter(({ command }) => ['select_native_model', 'save_app_settings', 'restart_service', 'download_model'].includes(command));

for (const service of services) {
  test(`${service}: precision browsing does not select or download a model`, async ({ page }) => {
    await page.goto(`/tests/ui/?service=${service}`);
    const picker = page.getByRole('combobox', { name: /precision/ }).first();
    await expect(picker).toHaveValue(service === 'carey' ? 'BF16' : 'F16');
    if (service === 'melodyflow') {
      await expect(picker.locator('option')).toHaveCount(1);
      return;
    }
    await picker.selectOption('Q4_K_M');
    await expect(picker).toHaveValue('Q4_K_M');
    expect(writes(await page.evaluate(() => window.testCalls))).toEqual([]);
    const row = picker.locator('xpath=ancestor::div[contains(@class,"model-row")]');
    await row.getByRole('button', { name: 'Download', exact: true }).click();
    const calls = writes(await page.evaluate(() => window.testCalls));
    expect(calls).toHaveLength(1);
    expect(calls[0].command).toBe('download_model');
    expect(calls[0].args.modelId).toContain('Q4_K_M');
    await expect(picker).toBeDisabled();
    await expect(row).toContainText(/Downloading/);
  });
}

for (const service of ['sa3', 'yuey', 'foundation', 'stable-audio', 'carey', 'gary']) {
  test(`${service}: use activates the displayed installed precision explicitly`, async ({ page }) => {
    await page.goto(`/tests/ui/?service=${service}&all&running`);
    const picker = page.getByRole('combobox', { name: /precision/ }).first();
    await picker.selectOption('Q4_K_M');
    expect(writes(await page.evaluate(() => window.testCalls))).toEqual([]);
    await picker.locator('xpath=ancestor::div[contains(@class,"model-row")]').getByRole('button', { name: 'use', exact: true }).click();
    await expect(page.getByRole('status').last()).toContainText(service === 'yuey' ? 'restarting' : /next time/);
    const calls = writes(await page.evaluate(() => window.testCalls));
    if (service === 'yuey') {
      expect(calls.map(({ command }) => command)).toEqual(['save_app_settings', 'restart_service']);
      expect(calls[0].args.settings.yueyEncoding).toBe('Q4_K_M');
    } else {
      expect(calls.map(({ command }) => command)).toEqual(['select_native_model']);
      expect(calls[0].args.encoding).toBe('Q4_K_M');
    }
  });
}

test('SA3 training precision stays independent of generation', async ({ page }) => {
  await page.goto('/tests/ui/?service=sa3');
  await page.getByRole('combobox', { name: 'LoRA training base precision', exact: true }).selectOption('Q4_K_M');
  await expect(page.getByRole('combobox', { name: 'generation tier precision', exact: true })).toHaveValue('F16');
  expect(writes(await page.evaluate(() => window.testCalls))).toEqual([]);
});

test('SA3 cannot use a precision without shared components', async ({ page }) => {
  await page.goto('/tests/ui/?service=sa3&all&missing-shared');
  const picker = page.getByRole('combobox', { name: 'generation tier precision', exact: true });
  await picker.selectOption('Q4_K_M');
  await expect(picker.locator('xpath=ancestor::div[contains(@class,"model-row")]').getByRole('button', { name: 'use', exact: true })).toBeDisabled();
});

test('failed copies can be retried at the displayed precision', async ({ page }) => {
  await page.goto('/tests/ui/?service=sa3&failed');
  const picker = page.getByRole('combobox', { name: 'generation tier precision', exact: true });
  await picker.selectOption('Q4_K_M');
  await picker.locator('xpath=ancestor::div[contains(@class,"model-row")]').getByRole('button', { name: 'Retry', exact: true }).click();
  expect(writes(await page.evaluate(() => window.testCalls))[0].args.modelId).toBe('sa3-native::medium-Q4_K_M');
});

test('download buttons have white text and a solid red background', async ({ page }) => {
  await page.goto('/tests/ui/?service=sa3');
  await page.getByRole('combobox', { name: 'generation tier precision', exact: true }).selectOption('Q4_K_M');
  const button = page.getByRole('region', { name: 'generation tier', exact: true }).getByRole('button', { name: 'Download', exact: true });
  expect(await button.evaluate((element) => {
    const style = getComputedStyle(element);
    return [style.color, style.backgroundColor, getComputedStyle(element.closest('.model-row')).opacity];
  })).toEqual(['rgb(255, 255, 255)', 'rgb(187, 0, 0)', '1']);
});

test('storage remove buttons fit on one line at a narrow width', async ({ page }) => {
  await page.setViewportSize({ width: 800, height: 600 });
  await page.goto('/tests/ui/?controls');
  const rows = page.locator('.env-row');
  await expect(rows).toHaveCount(2);
  for (const row of await rows.all()) {
    const button = row.getByRole('button');
    expect(await button.evaluate((element) => ({ wrapping: getComputedStyle(element).whiteSpace, overflow: element.scrollWidth > element.clientWidth }))).toEqual({ wrapping: 'nowrap', overflow: false });
    const rowBox = await row.boundingBox();
    const buttonBox = await button.boundingBox();
    expect(buttonBox.x + buttonBox.width).toBeLessThanOrEqual(rowBox.x + rowBox.width + 1);
  }
});

test('native runtime actions and Yuey options have readable colors', async ({ page }) => {
  await page.goto('/tests/ui/?controls');
  await page.getByRole('dialog').press('Escape');
  const reinstall = page.getByRole('button', { name: 'reinstall runtime', exact: true });
  await expect(reinstall).toBeVisible();
  expect(await reinstall.evaluate((element) => [getComputedStyle(element).color, getComputedStyle(element).backgroundColor])).toEqual(['rgb(255, 255, 255)', 'rgb(187, 0, 0)']);
  const method = page.getByRole('combobox', { name: /instrumental method/ });
  await expect(method).toHaveValue('transfer');
  expect(await method.locator('option[value="transfer"]').evaluate((element) => [getComputedStyle(element).color, getComputedStyle(element).backgroundColor])).toEqual(['rgb(224, 224, 224)', 'rgb(34, 34, 34)']);
});
