<script lang="ts">
  /**
   * Auto-scrolling log tail viewer.
   *
   * Polls the supplied `fetcher` on a 2s interval (matches the design
   * doc §8 cadence) while `active=true`. Stops polling on terminal
   * states. Auto-scrolls to bottom unless the user has scrolled up
   * (sticky-bottom behaviour — same heuristic as `tail -f`).
   */
  import type { LogTailResponse } from '$lib/types_train';

  interface Props {
    /** Job we're tailing. Switching this resets the buffer. */
    jobId: string;
    /** Stop polling when false; resume when flips back true. */
    active?: boolean;
    /** Lines per request. Backend caps at 5000. */
    lines?: number;
    /** Poll interval in ms. */
    intervalMs?: number;
    /** Fetcher injected so the page can supply its own AbortSignal handling. */
    fetcher: (jobId: string, lines: number) => Promise<LogTailResponse>;
  }

  let { jobId, active = true, lines = 200, intervalMs = 2000, fetcher }: Props = $props();

  let buffer = $state<string[]>([]);
  let error = $state<string | null>(null);
  let scroller: HTMLDivElement | null = $state(null);
  let stickyBottom = $state<boolean>(true);

  let timer: ReturnType<typeof setInterval> | null = null;
  let inFlight: AbortController | null = null;

  async function tick(): Promise<void> {
    if (inFlight) inFlight.abort();
    inFlight = new AbortController();
    try {
      const res = await fetcher(jobId, lines);
      buffer = res.lines ?? [];
      error = null;
      if (stickyBottom) queueMicrotask(scrollToBottom);
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      error = (e as Error).message;
    }
  }

  function scrollToBottom(): void {
    if (!scroller) return;
    scroller.scrollTop = scroller.scrollHeight;
  }

  function onScroll(): void {
    if (!scroller) return;
    // Within 32px of bottom counts as "still tailing". Anything above
    // pauses auto-scroll so the user can read prior lines without the
    // viewport snapping back on each poll.
    const slack = 32;
    stickyBottom =
      scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= slack;
  }

  // Reset buffer when job changes — otherwise the user briefly sees the
  // previous run's tail under the new heading.
  let lastJobId = $state<string>('');
  $effect(() => {
    if (jobId !== lastJobId) {
      buffer = [];
      lastJobId = jobId;
      stickyBottom = true;
    }
  });

  $effect(() => {
    if (!active || !jobId) {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      return;
    }
    void tick();
    timer = setInterval(() => void tick(), intervalMs);
    return () => {
      if (timer) clearInterval(timer);
      timer = null;
      inFlight?.abort();
      inFlight = null;
    };
  });
</script>

<div class="flex flex-col rounded-md border border-zinc-800 bg-zinc-950">
  <div
    class="flex items-center justify-between gap-3 border-b border-zinc-800 px-3 py-1.5"
  >
    <h3 class="text-xs font-semibold tracking-wide text-zinc-400 uppercase">
      Live log <span class="font-mono text-[10px] text-zinc-500">tail {lines}</span>
    </h3>
    <div class="flex items-center gap-2 text-[11px] text-zinc-500">
      {#if !stickyBottom}
        <button
          type="button"
          class="rounded border border-zinc-700 px-1.5 py-0.5 text-blue-300 hover:bg-zinc-900"
          onclick={() => {
            stickyBottom = true;
            scrollToBottom();
          }}
        >
          jump to bottom
        </button>
      {:else if active}
        <span class="font-mono text-green-400">tailing</span>
      {:else}
        <span class="font-mono">paused</span>
      {/if}
    </div>
  </div>
  {#if error}
    <p class="px-3 py-2 text-xs text-red-300">log fetch failed: {error}</p>
  {/if}
  <div
    bind:this={scroller}
    onscroll={onScroll}
    class="max-h-[24rem] min-h-[12rem] overflow-auto px-3 py-2 font-mono text-[11px] leading-relaxed text-zinc-300"
  >
    {#if buffer.length === 0}
      <p class="text-zinc-600">(no log lines yet)</p>
    {:else}
      {#each buffer as line, i (i)}
        <div class="whitespace-pre">{line}</div>
      {/each}
    {/if}
  </div>
</div>
