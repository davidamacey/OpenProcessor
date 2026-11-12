import { describe, expect, it, vi } from 'vitest';
import { mountUi, snip } from '$lib/test/mountUi';
import Badge from './Badge.svelte';
import Chip from './Chip.svelte';
import Kbd from './Kbd.svelte';

const text = snip('<b>tag</b>');

describe('Chip', () => {
  it('is a static neutral span by default', () => {
    const el = mountUi(Chip, { children: text }).firstElementChild!;
    expect(el.tagName).toBe('SPAN');
    expect(el.className).toBe('chip ');
  });

  it('takes its tone from the chip-* classes', () => {
    const el = mountUi(Chip, { tone: 'warn', children: text }).firstElementChild!;
    expect(el.className).toContain('chip-warn');
  });

  it('with onclick is a button, and pressed turns it accent with aria-pressed', () => {
    const onclick = vi.fn();
    const b = mountUi(Chip, { onclick, pressed: true, children: text }).querySelector(
      'button',
    )!;
    expect(b.getAttribute('aria-pressed')).toBe('true');
    expect(b.className).toContain('chip-accent');
    b.click();
    expect(onclick).toHaveBeenCalledOnce();
  });

  it('onremove adds a named close button', () => {
    const onremove = vi.fn();
    const el = mountUi(Chip, { onremove, removeLabel: 'Remove car', children: text });
    const close = el.querySelector(
      'button[aria-label="Remove car"]',
    ) as HTMLButtonElement;
    close.click();
    expect(onremove).toHaveBeenCalledOnce();
  });
});

describe('Badge', () => {
  it('is a round count pill, neutral or toned', () => {
    const n = mountUi(Badge, { children: snip('<i>12</i>') }).firstElementChild!;
    expect(n.className).toContain('rounded-full');
    expect(n.className).toContain('bg-surface-2');
    const t = mountUi(Badge, {
      tone: 'danger',
      children: snip('<i>3</i>'),
    }).firstElementChild!;
    expect(t.className).toContain('chip-danger');
  });
});

describe('Kbd', () => {
  it('renders a kbd element', () => {
    const el = mountUi(Kbd, { children: snip('<i>Esc</i>') });
    expect(el.querySelector('kbd')!.textContent).toBe('Esc');
  });
});
