/**
 * F-64 (fresh-start findings 2026-09-25): the promote modal showed only
 * "API 422 …". It now renders the served gate detail and offers force
 * only when the server says `force_allowed`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import PromoteModal from './PromoteModal.svelte';
import { ApiError, getPromoteJob, promoteTrainJob } from '$lib/api';
import type { PromoteJobStatus } from '$lib/types_train';
import { defaultTritonName } from '$lib/promote';
import { toastStore } from '$stores/toast.svelte';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return { ...actual, promoteTrainJob: vi.fn(), getPromoteJob: vi.fn() };
});

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
  vi.mocked(promoteTrainJob).mockReset();
  vi.mocked(getPromoteJob).mockReset();
  vi.useRealTimers();
});

function pj(
  status: PromoteJobStatus['status'],
  extra: Partial<PromoteJobStatus> = {},
): PromoteJobStatus {
  return {
    promote_id: 'p1',
    job_id: 'run-1',
    triton_name: 'run_1',
    status,
    poll_after_s: ['done', 'failed'].includes(status) ? null : 1,
    ...extra,
  };
}

function gate422(forceAllowed: boolean): ApiError {
  return new ApiError(422, '/curation/train/promote/run-1', {
    detail: {
      message: 'promote gate failed: 2 check(s)',
      failures: [
        { code: 'map50_below_threshold', message: 'mAP50 0.641 < 0.65' },
        {
          code: 'class_precision_below_threshold',
          message: 'precision 0.412 < 0.5',
          class_name: 'widget_b',
        },
      ],
      force_allowed: forceAllowed,
      override: 'Re-submit with force=true to promote anyway.',
      thresholds: { map50: 0.65 },
    },
  });
}

function submitButton(): HTMLButtonElement {
  return target.querySelector('button[type="submit"]') as HTMLButtonElement;
}

async function openAndSubmit(): Promise<void> {
  instance = mount(PromoteModal, {
    target,
    props: { open: true, jobId: 'run-1', defaultName: 'run_1', onclose: () => {} },
  });
  flushSync();
  submitButton().click();
  await vi.waitFor(() => {
    flushSync();
    expect(target.querySelector('[data-testid="promote-gate"]')).not.toBeNull();
  });
}

describe('PromoteModal gate failure', () => {
  it('renders the served message, every failure (with class) and the override hint', async () => {
    vi.mocked(promoteTrainJob).mockRejectedValue(gate422(true));
    await openAndSubmit();
    const text = target.querySelector('[data-testid="promote-gate"]')!.textContent ?? '';
    expect(text).toContain('promote gate failed: 2 check(s)');
    expect(text).toContain('mAP50 0.641 < 0.65');
    expect(text).toContain('widget_b');
    expect(text).toContain('precision 0.412 < 0.5');
    expect(text).toContain('Re-submit with force=true to promote anyway.');
    expect(target.textContent).not.toContain('API 422');
  });

  it('does not repeat the class prefix when the served message already carries it', async () => {
    vi.mocked(promoteTrainJob).mockRejectedValue(
      new ApiError(422, '/curation/train/promote/run-1', {
        detail: {
          message: 'promote gate failed: 1 check(s)',
          failures: [
            {
              code: 'class_precision_below_threshold',
              message: 'object: precision 0.412 < 0.5',
              class_name: 'object',
            },
          ],
          force_allowed: false,
        },
      }),
    );
    await openAndSubmit();
    const li = target.querySelector('[data-testid="promote-gate-failures"] li')!;
    expect(li.textContent?.replace(/\s+/g, ' ').trim()).toBe(
      'object: precision 0.412 < 0.5',
    );
  });

  it('force_allowed: offers force, and a checked force re-submits with force=true', async () => {
    vi.mocked(promoteTrainJob).mockRejectedValueOnce(gate422(true));
    await openAndSubmit();
    const force = target.querySelector(
      '[data-testid="promote-force"]',
    ) as HTMLInputElement;
    expect(force).not.toBeNull();
    force.click();
    flushSync();
    vi.mocked(promoteTrainJob).mockResolvedValueOnce(pj('queued'));
    submitButton().click();
    await vi.waitFor(() => expect(promoteTrainJob).toHaveBeenCalledTimes(2));
    expect(vi.mocked(promoteTrainJob).mock.calls[0]![1].force).toBeUndefined();
    expect(vi.mocked(promoteTrainJob).mock.calls[1]![1].force).toBe(true);
  });

  it('force not allowed: no force control', async () => {
    vi.mocked(promoteTrainJob).mockRejectedValue(gate422(false));
    await openAndSubmit();
    expect(target.querySelector('[data-testid="promote-force"]')).toBeNull();
  });
});

function fullClassRemapMissingGate(): ApiError {
  // OpenProcessor 7e758390, tests/curation/test_promote_full_class_remap.py
  // ::test_full_class_promote_without_remap_refuses_when_registry_has_a_gap
  // -- the EXISTING PromoteGateFailedDetail shape, no class_name on the
  // failure (this gate isn't about one class), force_allowed true.
  const message =
    "job 'gap-no-remap-job' is a full-class run with no resolvable class_remap and its " +
    'pinned registry has a gap or deprecated class -- the identity map ' +
    '(labels.txt line i = registry class i) is not provably correct for a ' +
    'dense-id-trained model; refusing to promote (pass force=true to bypass -- ' +
    'logged distinctly)';
  return new ApiError(422, '/curation/train/promote/gap-no-remap-job', {
    detail: {
      message,
      failures: [{ code: 'class_remap_missing_full_class', message }],
      force_allowed: true,
      override: 'pass force=true in the request body',
    },
  });
}

describe('PromoteModal gate failure — full-class remap missing (OpenProcessor 7e758390)', () => {
  it('renders the message, the class_remap_missing_full_class failure, the override hint, and offers force', async () => {
    vi.mocked(promoteTrainJob).mockRejectedValue(fullClassRemapMissingGate());
    await openAndSubmit();
    const text = target.querySelector('[data-testid="promote-gate"]')!.textContent ?? '';
    expect(text).toContain('is a full-class run with no resolvable class_remap');
    expect(text).toContain('pass force=true in the request body');
    expect(target.querySelector('[data-testid="promote-force"]')).not.toBeNull();
    expect(target.textContent).not.toContain('API 422');
  });
});

describe('defaultTritonName', () => {
  it('is the Triton-safe job id with no version suffix', () => {
    expect(defaultTritonName('2026-09-25T17-01-25_yolo26s')).toBe(
      '2026-09-25T17-01-25_yolo26s',
    );
    expect(defaultTritonName('a:b.c')).toBe('a_b_c');
    expect(defaultTritonName('x'.repeat(80))).toHaveLength(64);
    expect(defaultTritonName('2026-09-25T17-01-25_yolo26s')).not.toMatch(/_v\d+$/);
  });
});

describe('PromoteModal success toast', () => {
  const RES = {
    job_id: 'run-1',
    triton_name: 'run_1',
    onnx_path: '/m/model.onnx',
    config_path: '/m/config.pbtxt',
    labels_path: '/m/labels.txt',
    triton_loaded: true,
    cold_start_expected_on_first_inference: false,
  };

  async function promoteWith(extra: Record<string, unknown>): Promise<string> {
    const success = vi.spyOn(toastStore, 'success');
    vi.mocked(promoteTrainJob).mockResolvedValue(
      pj('done', { result: { ...RES, ...extra } }),
    );
    instance = mount(PromoteModal, {
      target,
      props: { open: true, jobId: 'run-1', defaultName: 'run_1', onclose: () => {} },
    });
    flushSync();
    submitButton().click();
    vi.mocked(getPromoteJob).mockResolvedValue(
      pj('done', { result: { ...RES, ...extra } }),
    );
    await vi.waitFor(() => expect(success).toHaveBeenCalled(), { timeout: 3000 });
    const msg = String(success.mock.calls[0][0]);
    success.mockRestore();
    return msg;
  }

  it('warns the first prediction is slow when the server expects a cold start', async () => {
    const msg = await promoteWith({ cold_start_expected_on_first_inference: true });
    expect(msg).toContain('Promoted run_1');
    expect(msg).toMatch(/first prediction will be slow/i);
  });

  it('says nothing about a cold start when the server says none', async () => {
    expect(await promoteWith({ cold_start_expected_on_first_inference: false })).toBe(
      'Promoted run_1 → Triton',
    );
  });
});

describe('PromoteModal background job (#87)', () => {
  function mountOpen(extra: Record<string, unknown> = {}) {
    const onclose = vi.fn();
    const onpromoted = vi.fn();
    instance = mount(PromoteModal, {
      target,
      props: {
        open: true,
        jobId: 'run-1',
        defaultName: 'run_1',
        onclose,
        onpromoted,
        ...extra,
      },
    });
    flushSync();
    return { onclose, onpromoted };
  }
  const phaseOf = (name: string) =>
    target.querySelector(`[data-testid="promote-phase-${name}"]`);
  const current = () =>
    target.querySelector('[aria-current="step"]')?.getAttribute('data-testid');

  it('walks the served phases then closes with the result on done', async () => {
    const { onclose, onpromoted } = mountOpen();
    vi.mocked(promoteTrainJob).mockResolvedValue(pj('queued'));
    const polls = [pj('exporting'), pj('loading'), pj('building'), pj('warming')];
    const RES = {
      job_id: 'run-1',
      triton_name: 'run_1',
      cold_start_expected_on_first_inference: false,
    };
    vi.mocked(getPromoteJob).mockImplementation(async () =>
      polls.length ? polls.shift()! : pj('done', { result: RES as never }),
    );
    submitButton().click();
    await vi.waitFor(() => {
      flushSync();
      expect(current()).toBe('promote-phase-queued');
    });
    const seen: string[] = [];
    await vi.waitFor(
      () => {
        flushSync();
        const c = current();
        if (c && seen[seen.length - 1] !== c) seen.push(c);
        expect(onpromoted).toHaveBeenCalled();
      },
      { timeout: 8000, interval: 100 },
    );
    expect(seen).toEqual([
      'promote-phase-queued',
      'promote-phase-exporting',
      'promote-phase-loading',
      'promote-phase-building',
      'promote-phase-warming',
    ]);
    expect(onclose).toHaveBeenCalled();
  }, 15000);

  it('failed: shows the served error with its status and offers edit-and-retry', async () => {
    mountOpen();
    vi.mocked(promoteTrainJob).mockResolvedValue(pj('queued'));
    vi.mocked(getPromoteJob).mockResolvedValue(
      pj('failed', { error: 'Triton refused the load', error_status: 502 }),
    );
    submitButton().click();
    await vi.waitFor(
      () => {
        flushSync();
        expect(target.querySelector('[data-testid="promote-failed"]')).not.toBeNull();
      },
      { timeout: 4000 },
    );
    expect(target.querySelector('[data-testid="promote-failed"]')!.textContent).toContain(
      'Triton refused the load (502)',
    );
    const retry = Array.from(target.querySelectorAll('button')).find((b) =>
      /edit and retry/i.test(b.textContent ?? ''),
    )!;
    retry.click();
    flushSync();
    expect(target.querySelector('[data-testid="promote-progress"]')).toBeNull();
    expect(submitButton()).not.toBeNull();
  });

  it('attachJob with an active promote shows progress instead of the form', () => {
    vi.mocked(getPromoteJob).mockResolvedValue(pj('building'));
    mountOpen({ attachJob: pj('building') });
    expect(target.querySelector('[data-testid="promote-progress"]')).not.toBeNull();
    expect(current()).toBe('promote-phase-building');
    expect(phaseOf('exporting')!.textContent).toContain('✓');
    expect(target.querySelector('button[type="submit"]')).toBeNull();
  });
});
