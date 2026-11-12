import { describe, expect, it } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import SidebarNav from './SidebarNav.svelte';

const groups = [
  {
    heading: 'Project',
    items: [
      { id: 'general', label: 'General', href: '/p/default/settings/general' },
      { id: 'cluster', label: 'Cluster', href: '/p/default/settings/cluster', count: 2 },
    ],
  },
  {
    heading: 'Personal',
    items: [{ id: 'keys', label: 'Keyboard', href: '/p/default/settings/keys' }],
  },
];

describe('SidebarNav', () => {
  it('lists every group and marks the current page', () => {
    const el = mountUi(SidebarNav, {
      groups,
      current: 'cluster',
      label: 'Settings sections',
    });
    const nav = el.querySelector('nav[aria-label="Settings sections"]')!;
    expect(nav.textContent).toContain('Project');
    expect(nav.textContent).toContain('Personal');
    const links = nav.querySelectorAll('a');
    expect(links.length).toBe(3);
    const current = nav.querySelectorAll('a[aria-current="page"]');
    expect(current.length).toBe(1);
    expect(current[0]!.textContent).toContain('Cluster');
    expect(current[0]!.className).toContain('border-accent');
  });

  it('has a select twin for narrow screens, with the current page chosen', () => {
    const el = mountUi(SidebarNav, {
      groups,
      current: 'keys',
      label: 'Settings sections',
    });
    const select = el.querySelector('select')!;
    expect(select.className).toContain('md:hidden');
    expect(select.value).toBe('keys');
    expect(select.querySelectorAll('optgroup').length).toBe(2);
    expect(el.querySelector('nav')!.className).toContain('hidden');
  });
});
