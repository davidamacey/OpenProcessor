import { afterEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import type { WorkflowStep } from '$lib/types_workflow';
import NextStepsCard from './NextStepsCard.svelte';

let instance: unknown;
let target: HTMLDivElement;

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target?.remove();
});

const STEPS = [
  {
    id: 'ingest',
    label: 'Ingest images',
    detail: '120 images',
    done: true,
    section: 'ingest',
  },
  {
    id: 'classes',
    label: 'Create classes from the detector',
    detail: 'No classes yet',
    done: false,
    section: 'classes',
  },
  {
    id: 'cluster',
    label: 'Cluster',
    detail: 'No partition yet',
    done: false,
    section: 'workspace',
    query: 'view=clusters',
  },
  {
    id: 'near_dups',
    label: 'Find near-duplicates',
    detail: 'Never run',
    done: false,
    section: 'quality/near-duplicates',
  },
] satisfies WorkflowStep[];

function render(): HTMLDivElement {
  target = document.createElement('div');
  document.body.appendChild(target);
  instance = mount(NextStepsCard, {
    target,
    props: { steps: STEPS, next: STEPS[1], slug: 'demo' },
  } as never);
  flushSync();
  return target;
}

describe('NextStepsCard', () => {
  it('renders served steps in order', () => {
    const rows = [...render().querySelectorAll('[data-testid="next-step"]')];
    expect(rows.map((r) => r.getAttribute('data-step-id'))).toEqual([
      'ingest',
      'classes',
      'cluster',
      'near_dups',
    ]);
    expect(rows[1]!.textContent).toContain('No classes yet');
  });

  it('each link is the served section', () => {
    const links = [...render().querySelectorAll('a')].map((a) => a.getAttribute('href'));
    expect(links).toEqual([
      '/p/demo/ingest',
      '/p/demo/classes',
      '/p/demo/workspace?view=clusters',
      '/p/demo/quality/near-duplicates',
    ]);
  });

  it('done steps are marked, not hidden', () => {
    const root = render();
    const done = root.querySelector('[data-step-id="ingest"]')!;
    expect(done.getAttribute('data-done')).toBe('true');
    expect(done.querySelector('[aria-label="Done"]')).not.toBeNull();
    expect(
      root.querySelector('[data-step-id="classes"]')!.getAttribute('data-done'),
    ).toBe('false');
    expect(
      root.querySelector('[data-testid="next-step-current"]')!.textContent,
    ).toContain('Create classes');
  });

  it('a served action is a button on the step that offers it, and nowhere else', () => {
    const onaction = vi.fn();
    target = document.createElement('div');
    document.body.appendChild(target);
    const withAction = STEPS.map((s) =>
      s.id === 'classes' ? { ...s, action: 'adopt_detector_classes' as const } : s,
    );
    instance = mount(NextStepsCard, {
      target,
      props: { steps: withAction, next: withAction[1], slug: 'demo', onaction },
    } as never);
    flushSync();
    const buttons = target.querySelectorAll('button');
    expect(buttons.length).toBe(1);
    expect(buttons[0]!.closest('[data-step-id]')!.getAttribute('data-step-id')).toBe(
      'classes',
    );
    expect(buttons[0]!.classList.contains('btn-primary')).toBe(true);
    buttons[0]!.click();
    expect(onaction).toHaveBeenCalledWith(withAction[1]);
  });
});
