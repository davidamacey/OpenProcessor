import { describe, expect, it, vi } from 'vitest';
import { mountUi, snip } from '$lib/test/mountUi';
import Button from './Button.svelte';

const btn = (props: Record<string, unknown>) =>
  mountUi(Button, { children: snip('<span>Save</span>'), ...props }).querySelector(
    'button',
  )!;

describe('Button', () => {
  it('is the md secondary control by default and never submits a form implicitly', () => {
    const b = btn({});
    expect(b.className).toContain('btn');
    expect(b.className).not.toContain('btn-primary');
    expect(b.getAttribute('type')).toBe('button');
  });

  it('maps the variants and the sm size onto the control-tier classes', () => {
    expect(btn({ variant: 'primary' }).className).toContain('btn-primary');
    expect(btn({ variant: 'danger' }).className).toContain('btn-danger');
    expect(btn({ variant: 'ghost' }).className).toContain('btn-ghost');
    expect(btn({ size: 'sm' }).className).toContain('btn-sm');
  });

  it('loading disables it, marks it busy and shows a spinner instead of the icon', () => {
    const onclick = vi.fn();
    const b = btn({ loading: true, icon: 'check', onclick });
    expect(b.disabled).toBe(true);
    expect(b.getAttribute('aria-busy')).toBe('true');
    expect(b.querySelector('svg')!.getAttribute('class')).toContain('animate-spin');
    b.click();
    expect(onclick).not.toHaveBeenCalled();
  });

  it('with an icon and no text is a square icon button that keeps its aria-label', () => {
    const b = mountUi(Button, { icon: 'x', 'aria-label': 'Close' }).querySelector(
      'button',
    )!;
    expect(b.className).toContain('btn-icon');
    expect(b.getAttribute('aria-label')).toBe('Close');
    expect(b.querySelector('svg')).not.toBeNull();
  });

  it('calls onclick when enabled', () => {
    const onclick = vi.fn();
    btn({ onclick }).click();
    expect(onclick).toHaveBeenCalledOnce();
  });
});
