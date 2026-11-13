import { test } from 'node:test';
import assert from 'node:assert/strict';
import { shownPrecision } from '../src/lib/modelPrecision.ts';

test('a stale saved or browsed precision cannot hide the remaining model copies', () => {
  assert.equal(shownPrecision([{ value: 'F16' }, { value: 'Q4_K_M', status: 'downloaded' }], 'Q8_0', 'BF16'), 'Q4_K_M');
  assert.equal(shownPrecision([], 'Q8_0', 'F16'), undefined);
});

test('a download is visible after reopening even when another precision is selected', () => {
  assert.equal(shownPrecision([{ value: 'F16', status: 'downloaded' }, { value: 'Q4_K_M', status: 'downloading' }], 'F16', 'F16'), 'Q4_K_M');
});

test('recommendations do not replace an installed or explicitly browsed copy', () => {
  const options = [{ value: 'F16' }, { value: 'Q4_K_M', status: 'downloaded' }];
  assert.equal(shownPrecision(options, null, null, 'F16'), 'Q4_K_M');
  assert.equal(shownPrecision(options, 'F16', 'Q4_K_M'), 'F16');
});
