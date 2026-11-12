import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import { mountUi, snip } from '$lib/test/mountUi';
import EmptyState from './EmptyState.svelte';
import ErrorState from './ErrorState.svelte';
import Skeleton from './Skeleton.svelte';

describe('EmptyState', () => {
  it('says what is missing and offers one next step', () => {
    const el = mountUi(EmptyState, {
      title: 'No clusters yet',
      description: 'Cluster the items to group them.',
      action: snip('<button>Cluster items</button>'),
    });
    expect(el.textContent).toContain('No clusters yet');
    expect(el.textContent).toContain('Cluster the items');
    expect(el.querySelectorAll('button').length).toBe(1);
    expect(el.querySelector('svg')).not.toBeNull();
  });
});

describe('ErrorState', () => {
  it('is an alert with the served message, the request id and a working Retry', () => {
    const onretry = vi.fn();
    const el = mountUi(ErrorState, {
      message: 'Internal server error',
      requestId: 'abc-123',
      onretry,
    });
    expect(el.querySelector('[role="alert"]')).not.toBeNull();
    expect(el.querySelector('[data-testid="error-state-message"]')!.textContent).toBe(
      'Internal server error',
    );
    expect(el.querySelector('[data-testid="error-state-request-id"]')!.textContent).toBe(
      'abc-123',
    );
    (el.querySelector('button') as HTMLButtonElement).click();
    expect(onretry).toHaveBeenCalledOnce();
  });

  it('reads the message and request id from an ApiError', () => {
    const error = new ApiError(500, '/x', { detail: 'boom' }, undefined, 'req-9');
    const el = mountUi(ErrorState, { error });
    expect(el.textContent).toContain('boom');
    expect(el.querySelector('[data-testid="error-state-request-id"]')!.textContent).toBe(
      'req-9',
    );
    expect(el.querySelector('button')).toBeNull();
  });
});

describe('Skeleton', () => {
  it('is announced once as loading and draws one bar per line, the last shorter', () => {
    const el = mountUi(Skeleton, { lines: 3 });
    expect(el.querySelector('[role="status"]')!.getAttribute('aria-label')).toBe(
      'Loading',
    );
    const bars = el.querySelectorAll('[aria-hidden="true"] > div');
    expect(bars.length).toBe(3);
    expect(bars[2]!.className).toContain('w-2/3');
    expect(bars[0]!.className).toContain('w-full');
  });
});
