//! A pool of OS threads, each owning one piece of state (an nvJPEG lane, a CUDA stream), fed
//! from one queue (plan 17 section 2.1).
//!
//! CPU-bound and blocking work runs here, never on the network runtime; its result returns
//! through a tokio oneshot channel, as tokio's documentation advises for CPU-bound work
//! (<https://docs.rs/tokio/1.54.1/tokio/index.html>, "CPU-bound tasks and blocking code"). The
//! queue is crossbeam's multi-consumer channel (<https://docs.rs/crossbeam-channel/0.5.17>); it is
//! unbounded because the photos that can reach it are bounded upstream by admission.
//!
//! Two rules of the plan live here, once:
//! * a panic in a job fails that job ([`JobError::Panicked`]) and the thread goes on with its
//!   state, so a lane is never lost for good (`std::panic::catch_unwind`; the release profile
//!   keeps `panic = "unwind"`; plan 17 section 1.3);
//! * each hand-off checks the photo's [`CancelToken`] and whether anyone still waits for the
//!   answer; a job that is not wanted is dropped unrun, which drops whatever it owns (budget
//!   permits, region slices) instead of starting work (plan 17 section 2.2, "Cancellation").

use std::any::Any;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};
use tokio::sync::oneshot;

/// A photo's cancelled flag, set when its caller is gone or its deadline passed.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A token that is not cancelled.
    pub fn new() -> CancelToken {
        CancelToken::default()
    }

    /// Cancel every job submitted with this token (or a clone) that has not started.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether [`CancelToken::cancel`] was called on this token or a clone.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Why a job gave no value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobError {
    /// Cancelled, or nobody waited for it any more, before it started.
    Cancelled,
    /// It panicked; the message is the panic's payload when it is a string.
    Panicked(String),
    /// The pool is shut down.
    Closed,
}

impl fmt::Display for JobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JobError::Cancelled => write!(f, "cancelled before it started"),
            JobError::Panicked(m) => write!(f, "panicked: {m}"),
            JobError::Closed => write!(f, "the pool is shut down"),
        }
    }
}

impl std::error::Error for JobError {}

/// Why a pool did not start.
#[derive(Debug)]
pub enum PoolError<E> {
    /// A thread's state could not be built.
    Init(E),
    /// Building a thread's state panicked.
    Panicked(String),
    /// The operating system refused a thread.
    Spawn(std::io::Error),
}

impl<E: fmt::Display> fmt::Display for PoolError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolError::Init(e) => write!(f, "a pool thread's state failed: {e}"),
            PoolError::Panicked(m) => write!(f, "a pool thread's state panicked: {m}"),
            PoolError::Spawn(e) => write!(f, "a pool thread could not start: {e}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for PoolError<E> {}

/// The answer of one job; await it ([`JobHandle::wait`]) or block on it
/// ([`JobHandle::wait_blocking`], never on the network runtime). Dropping it tells the pool the
/// answer is no longer wanted.
#[derive(Debug)]
pub struct JobHandle<T> {
    rx: oneshot::Receiver<Result<T, JobError>>,
}

impl<T> JobHandle<T> {
    /// Wait for the job's value from async code.
    ///
    /// # Errors
    ///
    /// [`JobError`] when the job was cancelled, panicked, or the pool shut down before it answered.
    pub async fn wait(self) -> Result<T, JobError> {
        self.rx.await.unwrap_or(Err(JobError::Closed))
    }

    /// Wait for the job's value on a thread that may block (not inside an async task).
    ///
    /// # Errors
    ///
    /// [`JobError`] when the job was cancelled, panicked, or the pool shut down before it answered.
    pub fn wait_blocking(self) -> Result<T, JobError> {
        self.rx.blocking_recv().unwrap_or(Err(JobError::Closed))
    }
}

type Job<S> = Box<dyn FnOnce(&mut S) + Send>;

/// Threads, each owning a state built on that thread, running jobs from one queue.
pub struct WorkerPool<S> {
    tx: Option<Sender<Job<S>>>,
    threads: Vec<JoinHandle<()>>,
}

impl<S> fmt::Debug for WorkerPool<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerPool")
            .field("threads", &self.threads.len())
            .finish()
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "a non-string panic payload".to_string()
    }
}

impl<S: 'static> WorkerPool<S> {
    /// Start `count` threads named `{name}-{i}`; each builds its state with `init(i)` on itself
    /// (nvJPEG states and CUDA bindings are per thread). Returns once every state is built, or
    /// the first failure after stopping the threads already started.
    ///
    /// # Errors
    ///
    /// [`PoolError`] carrying the `init` error of the first thread whose state could not be built,
    /// or the failure to start a thread.
    pub fn new<E, F>(name: &str, count: usize, init: F) -> Result<WorkerPool<S>, PoolError<E>>
    where
        E: Send + 'static,
        F: Fn(usize) -> Result<S, E> + Send + Sync + 'static,
    {
        let (tx, rx) = crossbeam_channel::unbounded::<Job<S>>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), PoolError<E>>>();
        let init = Arc::new(init);
        let mut pool = WorkerPool {
            tx: Some(tx),
            threads: Vec::with_capacity(count),
        };
        for i in 0..count {
            let (rx, ready, init) = (rx.clone(), ready_tx.clone(), init.clone());
            let spawned = std::thread::Builder::new()
                .name(format!("{name}-{i}"))
                .spawn(move || run_thread(i, &*init, &rx, &ready));
            match spawned {
                Ok(handle) => pool.threads.push(handle),
                Err(e) => return Err(PoolError::Spawn(e)),
            }
        }
        drop(ready_tx);
        for _ in 0..count {
            match ready_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(PoolError::Panicked("a pool thread ended at start".into())),
            }
        }
        Ok(pool)
    }

    /// Queue `job` for the next free thread. It runs only if `cancel` is not set and the handle
    /// is still held when a thread takes it.
    ///
    /// # Errors
    ///
    /// [`JobError::Closed`] when the pool is shut down.
    pub fn submit<T, F>(&self, cancel: &CancelToken, job: F) -> Result<JobHandle<T>, JobError>
    where
        T: Send + 'static,
        F: FnOnce(&mut S) -> T + Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let cancel = cancel.clone();
        let wrapped: Job<S> = Box::new(move |state: &mut S| {
            if cancel.is_cancelled() || tx.is_closed() {
                drop(job);
                let _ = tx.send(Err(JobError::Cancelled));
                return;
            }
            let out = catch_unwind(AssertUnwindSafe(|| job(state)))
                .map_err(|p| JobError::Panicked(panic_message(p.as_ref())));
            let _ = tx.send(out);
        });
        match &self.tx {
            Some(q) => q.send(wrapped).map_err(|_| JobError::Closed)?,
            None => return Err(JobError::Closed),
        }
        Ok(JobHandle { rx })
    }

    /// The number of threads.
    pub fn threads(&self) -> usize {
        self.threads.len()
    }
}

fn run_thread<S, E>(
    index: usize,
    init: &(dyn Fn(usize) -> Result<S, E> + Send + Sync),
    rx: &Receiver<Job<S>>,
    ready: &std::sync::mpsc::Sender<Result<(), PoolError<E>>>,
) {
    let mut state = match catch_unwind(AssertUnwindSafe(|| init(index))) {
        Ok(Ok(state)) => {
            let _ = ready.send(Ok(()));
            state
        }
        Ok(Err(e)) => {
            let _ = ready.send(Err(PoolError::Init(e)));
            return;
        }
        Err(p) => {
            let _ = ready.send(Err(PoolError::Panicked(panic_message(p.as_ref()))));
            return;
        }
    };
    // Every job catches its own panic (`submit`), so this loop ends only when the queue closes.
    for job in rx.iter() {
        job(&mut state);
    }
}

impl<S> Drop for WorkerPool<S> {
    /// Close the queue, let the threads finish what they hold (every wait inside a job is
    /// bounded) and drop their states on their own threads.
    fn drop(&mut self) {
        self.tx = None;
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn pool(threads: usize) -> WorkerPool<usize> {
        WorkerPool::new("test", threads, |i| Ok::<usize, String>(i * 100)).unwrap()
    }

    #[test]
    fn jobs_run_on_their_threads_state() {
        let p = pool(1);
        let h = p.submit(&CancelToken::new(), |s: &mut usize| {
            *s += 1;
            *s
        });
        assert_eq!(h.unwrap().wait_blocking(), Ok(1));
        let h = p.submit(&CancelToken::new(), |s: &mut usize| *s);
        assert_eq!(
            h.unwrap().wait_blocking(),
            Ok(1),
            "the state persists between jobs"
        );
    }

    #[test]
    fn a_panic_fails_its_job_and_the_thread_serves_the_next() {
        let p = pool(1);
        let first = p.submit(&CancelToken::new(), |s: &mut usize| {
            *s = 7;
            std::thread::current().id()
        });
        let id = first.unwrap().wait_blocking().unwrap();
        let boom = p.submit(&CancelToken::new(), |_s: &mut usize| -> usize {
            panic!("decode blew up")
        });
        assert_eq!(
            boom.unwrap().wait_blocking(),
            Err(JobError::Panicked("decode blew up".into()))
        );
        let after = p.submit(&CancelToken::new(), |s: &mut usize| {
            (*s, std::thread::current().id())
        });
        assert_eq!(
            after.unwrap().wait_blocking(),
            Ok((7, id)),
            "same thread, same state"
        );
    }

    struct Held(Arc<AtomicUsize>);
    impl Drop for Held {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn a_cancelled_job_is_dropped_unrun_and_releases_what_it_owns() {
        let p = pool(1);
        let released = Arc::new(AtomicUsize::new(0));
        let ran = Arc::new(AtomicBool::new(false));
        let token = CancelToken::new();
        token.cancel();
        let (held, flag) = (Held(released.clone()), ran.clone());
        let h = p.submit(&token, move |_s: &mut usize| {
            let _keep = held;
            flag.store(true, Ordering::SeqCst);
        });
        assert_eq!(h.unwrap().wait_blocking(), Err(JobError::Cancelled));
        assert!(!ran.load(Ordering::SeqCst));
        assert_eq!(
            released.load(Ordering::SeqCst),
            1,
            "the permit it carried was released"
        );
    }

    #[test]
    fn a_job_nobody_waits_for_is_not_started() {
        let p = pool(1);
        // Hold the only thread until the second job's handle is gone.
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let blocker = p.submit(&CancelToken::new(), move |_s: &mut usize| {
            let _ = go_rx.recv();
        });
        let ran = Arc::new(AtomicBool::new(false));
        let flag = ran.clone();
        let orphan = p.submit(&CancelToken::new(), move |_s: &mut usize| {
            flag.store(true, Ordering::SeqCst)
        });
        drop(orphan);
        go_tx.send(()).unwrap();
        blocker.unwrap().wait_blocking().unwrap();
        // A later job on the same single thread runs after the orphan was taken off the queue.
        p.submit(&CancelToken::new(), |_s: &mut usize| ())
            .unwrap()
            .wait_blocking()
            .unwrap();
        assert!(!ran.load(Ordering::SeqCst));
    }

    #[test]
    fn cancelling_while_queued_stops_the_job_at_the_hand_off() {
        let p = pool(1);
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let blocker = p.submit(&CancelToken::new(), move |_s: &mut usize| {
            let _ = go_rx.recv();
        });
        let token = CancelToken::new();
        let queued = p.submit(&token, |_s: &mut usize| 1).unwrap();
        token.cancel();
        go_tx.send(()).unwrap();
        blocker.unwrap().wait_blocking().unwrap();
        assert_eq!(queued.wait_blocking(), Err(JobError::Cancelled));
    }

    #[test]
    fn a_failed_state_stops_the_start() {
        let r = WorkerPool::<usize>::new("test", 3, |i| {
            if i == 2 {
                Err("no lane".to_string())
            } else {
                Ok(i)
            }
        });
        assert!(matches!(r, Err(PoolError::Init(m)) if m == "no lane"));
    }

    #[tokio::test]
    async fn the_answer_can_be_awaited() {
        let p = pool(2);
        let h = p
            .submit(&CancelToken::new(), |s: &mut usize| *s + 1)
            .unwrap();
        let v = h.wait().await.unwrap();
        assert!(v == 1 || v == 101);
    }
}
