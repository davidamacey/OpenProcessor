/**
 * The subscriber count behind a polling store: the first `acquire()` starts the run, the last
 * release stops it.
 *
 * The root layout's mount/release/remount during app boot (an SPA-boot quirk: `acquire()` fires
 * repeatedly with the count back at 0 each time) used to tear down and immediately restart
 * polling, aborting an in-flight fetch and refiring a new one. The stop is therefore deferred
 * one task, and a same-tick-or-soon re-acquire cancels it instead of running it.
 */
export class RefCountedRun {
  #count = 0;
  #stopHandle: ReturnType<typeof setTimeout> | null = null;
  readonly #isRunning: () => boolean;
  readonly #start: () => void;
  readonly #stop: () => void;

  constructor(hooks: { isRunning: () => boolean; start: () => void; stop: () => void }) {
    this.#isRunning = hooks.isRunning;
    this.#start = hooks.start;
    this.#stop = hooks.stop;
  }

  /** At least one subscriber holds the run. */
  get held(): boolean {
    return this.#count > 0;
  }

  /** Returns the release function for this subscriber. */
  acquire(): () => void {
    if (this.#stopHandle) {
      clearTimeout(this.#stopHandle);
      this.#stopHandle = null;
    }
    this.#count += 1;
    if (this.#count === 1 && !this.#isRunning()) this.#start();
    return () => this.#release();
  }

  #release(): void {
    this.#count = Math.max(0, this.#count - 1);
    if (this.#count === 0 && !this.#stopHandle) {
      this.#stopHandle = setTimeout(() => {
        this.#stopHandle = null;
        if (this.#count === 0) this.#stop();
      }, 0);
    }
  }
}
