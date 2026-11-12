import { describe, expect, it } from 'vitest';
import { mountUi, snip } from '$lib/test/mountUi';
import Card from './Card.svelte';
import PageHeader from './PageHeader.svelte';
import PageShell from './PageShell.svelte';
import Section from './Section.svelte';

describe('PageShell', () => {
  it('uses one width per size: 6xl by default, 7xl wide, none for full', () => {
    const w = (width?: 'default' | 'wide' | 'full') =>
      mountUi(PageShell, { width, children: snip('<p>x</p>') }).firstElementChild!;
    expect(w().className).toContain('max-w-6xl');
    expect(w('wide').className).toContain('max-w-7xl');
    expect(w('full').className).not.toMatch(/max-w-/);
    expect(w().className).toContain('px-4');
    expect(w().className).toContain('md:px-gutter');
  });
});

describe('PageHeader', () => {
  it('renders the title as the page h1 with its description and actions', () => {
    const el = mountUi(PageHeader, {
      title: 'Quality',
      description: 'Audit and duplicates',
      actions: snip('<button>Run</button>'),
    });
    const h1 = el.querySelector('h1')!;
    expect(h1.textContent).toBe('Quality');
    expect(h1.className).toContain('text-h1');
    expect(el.textContent).toContain('Audit and duplicates');
    expect(el.querySelector('button')!.textContent).toBe('Run');
  });

  it('has no description paragraph or action row when none is given', () => {
    const el = mountUi(PageHeader, { title: 'Train' });
    expect(el.querySelector('p')).toBeNull();
    expect(el.querySelectorAll('header > div').length).toBe(1);
  });
});

describe('Card', () => {
  it('is a padded surface, and a selected card gets a ring and a check, not colour alone', () => {
    const plain = mountUi(Card, { children: snip('<p>body</p>') }).firstElementChild!;
    expect(plain.className).toContain('surface');
    expect(plain.className).toContain('p-4');
    expect(plain.querySelector('svg')).toBeNull();
    const sel = mountUi(Card, {
      selected: true,
      children: snip('<p>body</p>'),
    }).firstElementChild!;
    expect(sel.className).toContain('ring-accent');
    expect(sel.querySelector('svg[aria-label="Selected"]')).not.toBeNull();
  });
});

describe('Section', () => {
  it('is a region named by its h2', () => {
    const el = mountUi(Section, { title: 'Pipeline', children: snip('<p>rows</p>') });
    const section = el.querySelector('section')!;
    const h2 = el.querySelector('h2')!;
    expect(section.getAttribute('aria-labelledby')).toBe(h2.id);
    expect(h2.textContent).toBe('Pipeline');
  });
});
