/**
 * The endpoint form's provider, JSON mode and temperature rows: the
 * provider select carries the served options, the provider's served
 * defaults show under it, and an emptied temperature is sent as null
 * ("none": the server leaves it out).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { bodyFixture, schemaFixture } from '$lib/test/fixtures/vlm';
import type { VlmEndpointBody } from '$lib/types_vlm';
import VlmEndpointForm from './VlmEndpointForm.svelte';

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | null = null;

function render(body: VlmEndpointBody, onchange = vi.fn()) {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(VlmEndpointForm, {
    target,
    props: {
      schema: schemaFixture(),
      body,
      report: null,
      list: null,
      catalog: null,
      onchange,
    },
  });
  flushSync();
  const adv = target.querySelector<HTMLInputElement>('[data-testid="show-advanced"]')!;
  adv.click();
  flushSync();
  return onchange;
}

afterEach(() => {
  if (instance) unmount(instance);
  instance = null;
  target?.remove();
});

describe('VlmEndpointForm provider rows', () => {
  it('offers the served providers and shows the selected one defaults', () => {
    render(bodyFixture({ provider: 'anthropic' }));
    const opts = [...target.querySelectorAll('option')].map((o) => o.value);
    expect(opts).toContain('anthropic');
    expect(
      target.querySelector('[data-testid="vlm-provider-defaults"]')?.textContent,
    ).toMatch(/JSON mode off.*temperature\s+none \(not sent\)/s);
  });

  it('sends null when the temperature is cleared', () => {
    const onchange = render(bodyFixture({ temperature: 0.5 }));
    const input = target.querySelector<HTMLInputElement>(
      'input[type="number"][step="any"][max="2"]',
    )!;
    input.value = '';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(onchange).toHaveBeenCalledWith('temperature', null);
  });
});
