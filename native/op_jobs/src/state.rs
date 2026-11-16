//! The job state machine (plan 02b section 1): the statuses, the allowed transitions, and the
//! vocabularies of pause reasons and attempt outcomes.
//!
//! The database trigger `jobs_status_transition_guard` is the authority and is generated from
//! `src/domain/job_states.py`; `contracts/json/job_kinds.json` carries the same table, and a test
//! holds [`TRANSITIONS`] equal to it, so a Rust writer never attempts a move the trigger refuses.

use std::fmt;

/// A row's `status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum JobStatus {
    /// Waiting for a runner.
    Queued,
    /// Held by a runner under a lease.
    Running,
    /// Stopped at a boundary, waiting for a person or for its cause to clear.
    Paused,
    /// A cancel was asked of a running job; the runner has not stopped it yet.
    Cancelling,
    /// Finished.
    Completed,
    /// Finished, with errors in its result.
    CompletedWithErrors,
    /// Finished without a result.
    Failed,
    /// Stopped for good on request.
    Cancelled,
    /// The runner stopped or its lease expired; recoverable.
    Interrupted,
}

impl JobStatus {
    /// Every status.
    pub const ALL: [Self; 9] = [
        Self::Queued,
        Self::Running,
        Self::Paused,
        Self::Cancelling,
        Self::Completed,
        Self::CompletedWithErrors,
        Self::Failed,
        Self::Cancelled,
        Self::Interrupted,
    ];

    /// The text stored in `jobs.status`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Cancelling => "cancelling",
            Self::Completed => "completed",
            Self::CompletedWithErrors => "completed_with_errors",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
        }
    }

    /// The status a stored text names.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.as_str() == text)
    }

    /// No further transition: `finished_at` is set exactly for these.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::CompletedWithErrors | Self::Failed | Self::Cancelled
        )
    }

    /// Counted by the exclusive-group index. `interrupted` is deliberately not active: a new start
    /// is allowed while an interrupted job waits to be resumed.
    #[must_use]
    pub const fn is_active(self) -> bool {
        matches!(
            self,
            Self::Queued | Self::Running | Self::Paused | Self::Cancelling
        )
    }
}

impl fmt::Display for JobStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

use JobStatus::{
    Cancelled, Cancelling, Completed, CompletedWithErrors, Failed, Interrupted, Paused, Queued,
    Running,
};

/// Every move the trigger allows.
pub const TRANSITIONS: [(JobStatus, JobStatus); 18] = [
    (Queued, Running),
    (Queued, Cancelled),
    (Queued, Failed),
    (Queued, Paused),
    (Running, Completed),
    (Running, CompletedWithErrors),
    (Running, Failed),
    (Running, Cancelling),
    (Running, Paused),
    (Running, Queued),
    (Running, Interrupted),
    (Cancelling, Interrupted),
    (Cancelling, Cancelled),
    (Paused, Queued),
    (Interrupted, Queued),
    (Paused, Cancelled),
    (Interrupted, Cancelled),
    (Interrupted, Failed),
];

/// Whether the trigger lets a row move from `from` to `to`. A row that stays where it is passes
/// (the trigger fires only when the status changes).
#[must_use]
pub fn transition_allowed(from: JobStatus, to: JobStatus) -> bool {
    from == to || TRANSITIONS.contains(&(from, to))
}

/// Why a job is `paused` (`jobs.pause_reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauseReason {
    /// A person asked.
    User,
    /// The handler stopped for an audit.
    Audit,
    /// The handler stopped at its budget.
    Budget,
    /// The project is paused; the reaper resumes the job when it is not.
    ProjectPaused,
    /// The VLM is not available; the reaper resumes the job when it is.
    VlmUnavailable,
}

impl PauseReason {
    /// The text stored in `jobs.pause_reason`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Audit => "audit",
            Self::Budget => "budget",
            Self::ProjectPaused => "project_paused",
            Self::VlmUnavailable => "vlm_unavailable",
        }
    }

    /// The reason a stored text names.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        [
            Self::User,
            Self::Audit,
            Self::Budget,
            Self::ProjectPaused,
            Self::VlmUnavailable,
        ]
        .into_iter()
        .find(|r| r.as_str() == text)
    }
}

/// How one attempt ended (`job_attempts.outcome`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// The handler returned a result.
    Completed,
    /// The handler returned a result with errors in it.
    CompletedWithErrors,
    /// The handler failed.
    Failed,
    /// A cancel was honoured.
    Cancelled,
    /// The handler stopped at a boundary for a pause.
    Paused,
    /// The worker shut down.
    Interrupted,
    /// The lease expired and the reaper took the job.
    Lost,
    /// The soft time limit was reached.
    TimeLimit,
    /// A sliced job gave way at a boundary.
    Yielded,
}

impl AttemptOutcome {
    /// The text stored in `job_attempts.outcome`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::CompletedWithErrors => "completed_with_errors",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Paused => "paused",
            Self::Interrupted => "interrupted",
            Self::Lost => "lost",
            Self::TimeLimit => "time_limit",
            Self::Yielded => "yielded",
        }
    }

    /// Whether the attempt spends a try of the retry budget (`retry_count` grows). A pause, a
    /// resume or a yield never does (research gap G-1).
    #[must_use]
    pub const fn spends_retry(self) -> bool {
        matches!(self, Self::Interrupted | Self::Lost | Self::TimeLimit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinds::contract;

    #[test]
    fn the_transition_table_is_the_one_the_trigger_is_built_from() {
        let from_contract: Vec<(JobStatus, JobStatus)> = contract()
            .job_states
            .transitions
            .iter()
            .map(|[a, b]| (JobStatus::parse(a).unwrap(), JobStatus::parse(b).unwrap()))
            .collect();
        let mut ours = TRANSITIONS.to_vec();
        let mut theirs = from_contract;
        ours.sort();
        theirs.sort();
        assert_eq!(ours, theirs);
    }

    #[test]
    fn every_pair_outside_the_table_is_refused_and_every_pair_in_it_accepted() {
        let table = &contract().job_states.transitions;
        for from in JobStatus::ALL {
            for to in JobStatus::ALL {
                let listed = table
                    .iter()
                    .any(|[a, b]| a == from.as_str() && b == to.as_str());
                assert_eq!(
                    transition_allowed(from, to),
                    listed || from == to,
                    "{from} -> {to}"
                );
            }
        }
    }

    #[test]
    fn a_running_job_cannot_be_cancelled_directly_and_a_cancelling_one_cannot_complete() {
        assert!(!transition_allowed(Running, Cancelled));
        assert!(!transition_allowed(Cancelling, Completed));
        assert!(!transition_allowed(Cancelling, Failed));
        assert!(!transition_allowed(Completed, Queued));
    }

    #[test]
    fn terminal_and_active_sets_are_the_contracts() {
        let states = &contract().job_states;
        for status in JobStatus::ALL {
            assert_eq!(
                status.is_terminal(),
                states.terminal.iter().any(|t| t == status.as_str()),
                "{status}"
            );
            assert_eq!(
                status.is_active(),
                states.active.iter().any(|t| t == status.as_str()),
                "{status}"
            );
        }
        assert_eq!(
            JobStatus::ALL.map(JobStatus::as_str).to_vec(),
            states.statuses
        );
    }

    #[test]
    fn the_vocabularies_are_the_contracts() {
        let c = contract();
        let reasons: Vec<&str> = [
            PauseReason::User,
            PauseReason::Audit,
            PauseReason::Budget,
            PauseReason::ProjectPaused,
            PauseReason::VlmUnavailable,
        ]
        .map(PauseReason::as_str)
        .to_vec();
        assert_eq!(reasons, c.pause_reasons);
        let outcomes = [
            AttemptOutcome::Completed,
            AttemptOutcome::CompletedWithErrors,
            AttemptOutcome::Failed,
            AttemptOutcome::Cancelled,
            AttemptOutcome::Paused,
            AttemptOutcome::Interrupted,
            AttemptOutcome::Lost,
            AttemptOutcome::TimeLimit,
            AttemptOutcome::Yielded,
        ];
        assert_eq!(
            outcomes.map(AttemptOutcome::as_str).to_vec(),
            c.attempt_outcomes
        );
        let mut spending: Vec<&str> = outcomes
            .into_iter()
            .filter(|o| o.spends_retry())
            .map(AttemptOutcome::as_str)
            .collect();
        spending.sort_unstable();
        assert_eq!(spending, c.retry_outcomes);
    }
}
