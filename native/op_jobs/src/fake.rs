//! An in-memory [`JobStore`] for the runner's tests: the job tables' behaviour as the statements
//! give it, on tokio's clock. A move the trigger would refuse panics, so a wrong move of the runner
//! is a test failure here, not only in the database tier.

use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::Value;
use tokio::time::Instant;
use uuid::Uuid;

use crate::state::{AttemptOutcome, JobStatus, PauseReason, transition_allowed};
use crate::store::{
    ClaimRequest, ClaimedJob, Conclusion, Control, JobId, JobStore, Lease, StoreError,
};

#[derive(Debug, Clone)]
pub(crate) struct FakeJob {
    pub id: Uuid,
    pub kind: String,
    pub queue: String,
    pub status: JobStatus,
    pub params: Value,
    pub checkpoint: Option<Value>,
    pub progress: Option<Value>,
    pub result: Option<Value>,
    pub error_code: Option<String>,
    pub attempt: i32,
    pub retry_count: i32,
    pub max_attempts: i32,
    pub auto_resume: bool,
    pub fencing_token: i64,
    pub owner: Option<String>,
    pub lease_expires: Option<Instant>,
    pub cancel_requested: bool,
    pub pause_requested: bool,
    pub pause_reason: Option<PauseReason>,
    pub group: Option<(String, String)>,
    pub seq: u64,
    pub attempts: Vec<Option<String>>,
}

#[derive(Default)]
pub(crate) struct FakeStore {
    jobs: Mutex<Vec<FakeJob>>,
    next_seq: Mutex<u64>,
    pub down: AtomicBool,
    /// Make this many `conclude` calls fail as an outage before one succeeds.
    pub conclude_failures: AtomicUsize,
    pub progress_writes: AtomicUsize,
    pub heartbeats: AtomicUsize,
}

fn control(job: &FakeJob) -> Control {
    Control {
        status: job.status,
        cancel_requested: job.cancel_requested,
        pause_requested: job.pause_requested,
        pause_reason: job.pause_reason,
    }
}

impl FakeStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn check_up(&self) -> Result<(), StoreError> {
        if self.down.load(Ordering::Relaxed) {
            return Err(StoreError::Other("the database is down".into()));
        }
        Ok(())
    }

    fn set_status(job: &mut FakeJob, to: JobStatus) {
        assert!(
            transition_allowed(job.status, to),
            "the trigger refuses {} -> {to}",
            job.status
        );
        job.status = to;
    }

    pub fn insert(
        &self,
        kind: &str,
        params: Value,
        group: Option<(&str, &str)>,
    ) -> Result<JobId, String> {
        let spec = crate::kinds::kind(kind).ok_or("unknown kind")?;
        let mut jobs = self.jobs.lock().unwrap();
        let group = group.map(|(g, s)| (g.to_owned(), s.to_owned()));
        let busy = group.as_ref().is_some_and(|g| {
            jobs.iter()
                .any(|j| j.group.as_ref() == Some(g) && j.status.is_active())
        });
        if busy {
            return Err("busy".into());
        }
        let mut seq = self.next_seq.lock().unwrap();
        *seq += 1;
        let id = Uuid::from_u128(u128::from(*seq));
        jobs.push(FakeJob {
            id,
            kind: kind.to_owned(),
            queue: spec.queue.clone(),
            status: JobStatus::Queued,
            params,
            checkpoint: None,
            progress: None,
            result: None,
            error_code: None,
            attempt: 0,
            retry_count: 0,
            max_attempts: spec.max_attempts,
            auto_resume: spec.auto_resume,
            fencing_token: 0,
            owner: None,
            lease_expires: None,
            cancel_requested: false,
            pause_requested: false,
            pause_reason: None,
            group,
            seq: *seq,
            attempts: Vec::new(),
        });
        Ok(JobId(id))
    }

    pub fn job(&self, id: JobId) -> FakeJob {
        self.jobs
            .lock()
            .unwrap()
            .iter()
            .find(|j| j.id == id.0)
            .cloned()
            .unwrap()
    }

    pub fn status(&self, id: JobId) -> JobStatus {
        self.job(id).status
    }

    fn with<T>(&self, id: JobId, f: impl FnOnce(&mut FakeJob) -> T) -> T {
        let mut jobs = self.jobs.lock().unwrap();
        f(jobs.iter_mut().find(|j| j.id == id.0).unwrap())
    }

    pub fn running_count(&self) -> usize {
        self.jobs
            .lock()
            .unwrap()
            .iter()
            .filter(|j| j.status == JobStatus::Running)
            .count()
    }

    /// The route's cancel.
    pub fn cancel(&self, id: JobId) -> JobStatus {
        self.with(id, |j| {
            j.cancel_requested = true;
            let to = if matches!(
                j.status,
                JobStatus::Queued | JobStatus::Paused | JobStatus::Interrupted
            ) {
                JobStatus::Cancelled
            } else {
                JobStatus::Cancelling
            };
            Self::set_status(j, to);
            j.status
        })
    }

    /// The route's pause.
    pub fn request_pause_route(&self, id: JobId) -> JobStatus {
        self.with(id, |j| {
            j.pause_requested = true;
            j.pause_reason.get_or_insert(PauseReason::User);
            if j.status == JobStatus::Queued {
                Self::set_status(j, JobStatus::Paused);
            }
            j.status
        })
    }

    /// The route's resume.
    pub fn resume(&self, id: JobId) -> Result<(), &'static str> {
        let mut jobs = self.jobs.lock().unwrap();
        let (group, status) = {
            let j = jobs.iter().find(|j| j.id == id.0).unwrap();
            (j.group.clone(), j.status)
        };
        if !matches!(status, JobStatus::Paused | JobStatus::Interrupted) {
            return Err("not resumable");
        }
        let busy = group.as_ref().is_some_and(|g| {
            jobs.iter()
                .any(|j| j.id != id.0 && j.group.as_ref() == Some(g) && j.status.is_active())
        });
        if busy {
            return Err("busy");
        }
        let mut seq = self.next_seq.lock().unwrap();
        *seq += 1;
        let j = jobs.iter_mut().find(|j| j.id == id.0).unwrap();
        Self::set_status(j, JobStatus::Queued);
        j.pause_requested = false;
        j.pause_reason = None;
        j.seq = *seq;
        Ok(())
    }

    /// Reaper rule a, for every lease past its time; returns the jobs it ended.
    pub fn reap_expired(&self) -> Vec<JobId> {
        let now = Instant::now();
        let mut jobs = self.jobs.lock().unwrap();
        let mut ended = Vec::new();
        for j in jobs.iter_mut() {
            let expired = matches!(j.status, JobStatus::Running | JobStatus::Cancelling)
                && j.lease_expires.is_some_and(|at| at < now);
            if !expired {
                continue;
            }
            if j.status == JobStatus::Running {
                j.retry_count += 1;
                Self::set_status(j, JobStatus::Interrupted);
            } else {
                Self::set_status(j, JobStatus::Cancelled);
            }
            j.owner = None;
            j.lease_expires = None;
            if let Some(last) = j.attempts.last_mut() {
                *last = Some("lost".into());
            }
            ended.push(JobId(j.id));
        }
        ended
    }

    /// Test hook: the lease of a running job ends in the past.
    pub fn expire_lease(&self, id: JobId) {
        self.with(id, |j| {
            j.lease_expires = Some(Instant::now() - Duration::from_secs(1))
        });
    }

    /// Reaper rule b with no settle time: `(requeued, failed)`.
    pub fn resume_interrupted(&self) -> (usize, usize) {
        let mut jobs = self.jobs.lock().unwrap();
        let mut seq = self.next_seq.lock().unwrap();
        let mut counts = (0, 0);
        for i in 0..jobs.len() {
            let j = &jobs[i];
            if j.status != JobStatus::Interrupted || !j.auto_resume {
                continue;
            }
            if j.retry_count >= j.max_attempts {
                Self::set_status(&mut jobs[i], JobStatus::Failed);
                jobs[i].error_code = Some("attempts_exhausted".into());
                counts.1 += 1;
                continue;
            }
            let group = j.group.clone();
            let blocked = group.is_some_and(|g| {
                jobs.iter().any(|o| {
                    o.id != jobs[i].id && o.group.as_ref() == Some(&g) && o.status.is_active()
                })
            });
            if blocked {
                continue;
            }
            *seq += 1;
            let j = &mut jobs[i];
            Self::set_status(j, JobStatus::Queued);
            j.pause_requested = false;
            j.pause_reason = None;
            j.seq = *seq;
            counts.0 += 1;
        }
        counts
    }

    /// A takeover without waiting: the job is queued again and claimed by a new owner, as after
    /// rule a and a resume.
    pub fn lose_and_requeue(&self, id: JobId) {
        self.with(id, |j| {
            j.lease_expires = Some(Instant::now() - Duration::from_secs(1));
        });
        self.reap_expired();
        self.resume(id).unwrap();
    }
}

impl JobStore for FakeStore {
    async fn claim(&self, request: &ClaimRequest) -> Result<Vec<ClaimedJob>, StoreError> {
        self.check_up()?;
        let wanted: BTreeSet<&str> = request.kinds.iter().map(String::as_str).collect();
        let mut jobs = self.jobs.lock().unwrap();
        let mut picked: Vec<usize> = jobs
            .iter()
            .enumerate()
            .filter(|(_, j)| {
                j.status == JobStatus::Queued
                    && wanted.contains(j.kind.as_str())
                    && !j.cancel_requested
                    && j.retry_count < j.max_attempts
            })
            .map(|(i, _)| i)
            .collect();
        picked.sort_by_key(|&i| jobs[i].seq);
        picked.truncate(request.limit);
        let mut out = Vec::new();
        for i in picked {
            let j = &mut jobs[i];
            Self::set_status(j, JobStatus::Running);
            j.attempt += 1;
            j.fencing_token += 1;
            j.owner = Some(request.owner.clone());
            j.lease_expires = Some(Instant::now() + request.lease_ttl);
            j.attempts.push(None);
            out.push(ClaimedJob {
                lease: Lease {
                    job_id: JobId(j.id),
                    project_id: None,
                    fencing_token: j.fencing_token,
                    attempt: j.attempt,
                },
                kind: j.kind.clone(),
                queue: j.queue.clone(),
                params: j.params.clone(),
                checkpoint: j.checkpoint.clone(),
            });
        }
        Ok(out)
    }

    async fn heartbeat(&self, lease: &Lease, ttl: Duration) -> Result<Option<Control>, StoreError> {
        self.check_up()?;
        self.heartbeats.fetch_add(1, Ordering::Relaxed);
        Ok(self.with(lease.job_id, |j| {
            let live = matches!(j.status, JobStatus::Running | JobStatus::Cancelling);
            (live && j.fencing_token == lease.fencing_token).then(|| {
                j.lease_expires = Some(Instant::now() + ttl);
                control(j)
            })
        }))
    }

    async fn progress(
        &self,
        lease: &Lease,
        snapshot: &Value,
    ) -> Result<Option<Control>, StoreError> {
        self.check_up()?;
        Ok(self.with(lease.job_id, |j| {
            let live = matches!(j.status, JobStatus::Running | JobStatus::Cancelling);
            (live && j.fencing_token == lease.fencing_token).then(|| {
                self.progress_writes.fetch_add(1, Ordering::Relaxed);
                j.progress = Some(snapshot.clone());
                control(j)
            })
        }))
    }

    async fn checkpoint(
        &self,
        lease: &Lease,
        cursor: Option<&Value>,
    ) -> Result<Option<Control>, StoreError> {
        self.check_up()?;
        Ok(self.with(lease.job_id, |j| {
            let live = matches!(j.status, JobStatus::Running | JobStatus::Cancelling);
            (live && j.fencing_token == lease.fencing_token).then(|| {
                j.checkpoint = cursor.cloned();
                control(j)
            })
        }))
    }

    async fn request_pause(&self, lease: &Lease, reason: PauseReason) -> Result<bool, StoreError> {
        self.check_up()?;
        Ok(self.with(lease.job_id, |j| {
            let live = j.status == JobStatus::Running && j.fencing_token == lease.fencing_token;
            if live {
                j.pause_requested = true;
                j.pause_reason.get_or_insert(reason);
            }
            live
        }))
    }

    async fn conclude(
        &self,
        lease: &Lease,
        conclusion: &Conclusion,
    ) -> Result<Option<JobStatus>, StoreError> {
        self.check_up()?;
        if self
            .conclude_failures
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
        {
            return Err(StoreError::Other("the database is restarting".into()));
        }
        let mut seq = self.next_seq.lock().unwrap();
        *seq += 1;
        let seq_now = *seq;
        drop(seq);
        Ok(self.with(lease.job_id, |j| {
            let live = matches!(j.status, JobStatus::Running | JobStatus::Cancelling);
            if !(live && j.fencing_token == lease.fencing_token) {
                return None;
            }
            let was_cancelling = j.status == JobStatus::Cancelling;
            let outcome = match conclusion {
                Conclusion::Finish {
                    status,
                    outcome,
                    result,
                    error_code,
                    ..
                } => {
                    let to = if was_cancelling {
                        JobStatus::Cancelled
                    } else {
                        *status
                    };
                    Self::set_status(j, to);
                    j.result = result.clone().map(Value::Object);
                    j.error_code.clone_from(error_code);
                    if was_cancelling && *status != JobStatus::Cancelled {
                        AttemptOutcome::Cancelled
                    } else {
                        *outcome
                    }
                }
                Conclusion::Interrupt => {
                    Self::set_status(
                        j,
                        if was_cancelling {
                            JobStatus::Cancelled
                        } else {
                            JobStatus::Interrupted
                        },
                    );
                    AttemptOutcome::Interrupted
                }
                Conclusion::Pause { reason } => {
                    if was_cancelling {
                        return None;
                    }
                    Self::set_status(j, JobStatus::Paused);
                    if j.pause_reason.is_none() {
                        j.pause_reason = Some(reason.unwrap_or(PauseReason::User));
                    }
                    AttemptOutcome::Paused
                }
                Conclusion::Yield => {
                    if was_cancelling {
                        return None;
                    }
                    Self::set_status(j, JobStatus::Queued);
                    j.seq = seq_now;
                    AttemptOutcome::Yielded
                }
            };
            if outcome.spends_retry() {
                j.retry_count += 1;
            }
            j.owner = None;
            j.lease_expires = None;
            if let Some(last) = j.attempts.last_mut() {
                *last = Some(outcome.as_str().to_owned());
            }
            Some(j.status)
        }))
    }
}
