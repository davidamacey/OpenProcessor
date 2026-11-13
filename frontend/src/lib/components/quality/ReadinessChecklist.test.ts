import { describe, expect, it } from 'vitest';
import { mountUi } from '$lib/test/mountUi';
import { projectsStore } from '$stores/projects.svelte';
import { testProject } from '$lib/test/fixtures/projects';
import type { ReadinessCheck } from '$lib/types_quality';
import ReadinessChecklist from './ReadinessChecklist.svelte';

const check = (over: Partial<ReadinessCheck>): ReadinessCheck => ({
  id: 'c',
  label: 'Test set',
  severity: 'block',
  message: '2 classes have under 5 test examples.',
  count: 2,
  link: { section: 'export', query: null },
  classes: [],
  ...over,
});

function render(checks: ReadinessCheck[]): HTMLElement {
  projectsStore.current = testProject({ slug: 'demo' });
  return mountUi(ReadinessChecklist, { checks });
}

describe('ReadinessChecklist', () => {
  it('names the severity in words beside the icon', () => {
    const el = render([
      check({ id: 'a' }),
      check({ id: 'b', severity: 'warn' }),
      check({ id: 'c', severity: 'ok', link: null }),
    ]);
    const words = [...el.querySelectorAll('[data-testid="readiness-severity"]')].map(
      (e) => e.textContent,
    );
    expect(words).toEqual(['Blocking', 'Warning', 'OK']);
    expect(el.querySelectorAll('svg').length).toBeGreaterThanOrEqual(3);
  });

  it('renders the served label, message and the served classes', () => {
    const el = render([
      check({
        classes: [
          { class_name: 'bottle', count: 0 },
          { class_name: 'book', count: 1131 },
        ],
      }),
    ]);
    expect(el.textContent).toContain('Test set');
    expect(el.textContent).toContain('2 classes have under 5 test examples.');
    expect(el.textContent).toContain('bottle 0 · book 1,131');
  });

  it('opens the served section and query, with no link when none is served', () => {
    const el = render([
      check({ id: 'w', link: { section: 'workspace', query: 'view=review&queue=x' } }),
      check({ id: 'n', link: null, severity: 'ok' }),
    ]);
    const rows = el.querySelectorAll('[data-testid="readiness-row"]');
    expect(rows[0]!.querySelector('a')!.getAttribute('href')).toBe(
      '/p/demo/workspace?view=review&queue=x',
    );
    expect(rows[1]!.querySelector('a')).toBeNull();
  });
});
