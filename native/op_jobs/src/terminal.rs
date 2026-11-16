//! How a run ends: the one fenced write that moves the row out of `running` (plan 02b section 3.1,
//! step 7), decided here from what the handler returned and what the lease watch learned.
//!
//! [`plan_conclusion`] is pure. The outcome map is the Python wrapper's (`_write_terminal`):
//!
//! | The run ended with | The row goes to |
//! |---|---|
//! | the worker shutting down (a drain that ran out, or `Interrupted`) | `interrupted` (`cancelling` goes `cancelled`); spends a retry |
//! | the soft time limit | `failed`, `time_limit`; spends a retry |
//! | a panic or a failure | `failed`, with the handler's code or `handler_error` |
//! | a cancel (returned `Cancelled`, or any result after a cancel request) | `cancelled`, the result kept |
//! | a pause (returned `Paused`, or `Yield` with a pause requested) | `paused` for a pausable kind, else `failed` `job_not_pausable` |
//! | `Yield` | `queued` for a kind with a slice, else `failed` `yield_not_allowed`; no retry spent |
//! | a result | `completed` or `completed_with_errors` |

use op_db::{DbError, Scope, Statement, Tx};
use serde_json::Value;

use crate::events::{StateRow, emit_state, state_columns};
use crate::handler::Outcome;
use crate::kinds::KindSpec;
use crate::state::{AttemptOutcome, JobStatus};
use crate::store::{Conclusion, Lease};

/// How the handler's run ended, before the lease watch is consulted.
#[derive(Debug, Clone, PartialEq)]
pub enum RunEnd {
    /// The handler returned.
    Returned(Outcome),
    /// The handler panicked; the text is the panic message.
    Panicked(String),
    /// The soft time limit aborted the handler.
    TimeLimit,
    /// The worker's drain period ran out and the handler was aborted.
    DrainExpired,
}

/// What the lease watch knew when the run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Flags {
    /// A cancel was requested.
    pub cancel_requested: bool,
    /// A pause was requested.
    pub pause_requested: bool,
}

fn truncated(text: &str, limit: usize) -> String {
    text.chars().take(limit).collect()
}

fn fail(code: &str, message: &str, outcome: AttemptOutcome, limit: usize) -> Conclusion {
    Conclusion::Finish {
        status: JobStatus::Failed,
        outcome,
        result: None,
        error_code: Some(code.to_owned()),
        error_message: Some(truncated(message, limit)),
    }
}

fn cancelled(result: Option<serde_json::Map<String, serde_json::Value>>) -> Conclusion {
    Conclusion::Finish {
        status: JobStatus::Cancelled,
        outcome: AttemptOutcome::Cancelled,
        result,
        error_code: None,
        error_message: None,
    }
}

/// The write that ends a run. `message_limit` is the contract's cut for error messages.
#[must_use]
pub fn plan_conclusion(
    kind: &KindSpec,
    end: &RunEnd,
    flags: Flags,
    message_limit: usize,
) -> Conclusion {
    let returned = match end {
        RunEnd::DrainExpired => return Conclusion::Interrupt,
        RunEnd::TimeLimit => {
            return fail(
                "time_limit",
                "the soft time limit was reached",
                AttemptOutcome::TimeLimit,
                message_limit,
            );
        }
        RunEnd::Panicked(message) => {
            return fail(
                "handler_error",
                &format!("panic: {message}"),
                AttemptOutcome::Failed,
                message_limit,
            );
        }
        RunEnd::Returned(outcome) => outcome,
    };
    match returned {
        Outcome::Interrupted => Conclusion::Interrupt,
        Outcome::Failed { code, message } => {
            fail(code, message, AttemptOutcome::Failed, message_limit)
        }
        Outcome::Cancelled if flags.cancel_requested => cancelled(None),
        Outcome::Cancelled => fail(
            "cancel_not_requested",
            "the handler stopped as cancelled but no cancel was requested",
            AttemptOutcome::Failed,
            message_limit,
        ),
        Outcome::Completed(result) | Outcome::CompletedWithErrors(result)
            if flags.cancel_requested =>
        {
            cancelled(Some(result.clone()))
        }
        Outcome::Yield | Outcome::Paused(_) if flags.cancel_requested => cancelled(None),
        Outcome::Paused(reason) => pause_or_fail(kind, *reason, message_limit),
        Outcome::Yield if flags.pause_requested => pause_or_fail(kind, None, message_limit),
        Outcome::Yield if kind.slice_s.is_none() => fail(
            "yield_not_allowed",
            &format!("{} has no slice_s", kind.name),
            AttemptOutcome::Failed,
            message_limit,
        ),
        Outcome::Yield => Conclusion::Yield,
        Outcome::Completed(result) => Conclusion::Finish {
            status: JobStatus::Completed,
            outcome: AttemptOutcome::Completed,
            result: Some(result.clone()),
            error_code: None,
            error_message: None,
        },
        Outcome::CompletedWithErrors(result) => Conclusion::Finish {
            status: JobStatus::CompletedWithErrors,
            outcome: AttemptOutcome::CompletedWithErrors,
            result: Some(result.clone()),
            error_code: None,
            error_message: None,
        },
    }
}

fn pause_or_fail(
    kind: &KindSpec,
    reason: Option<crate::state::PauseReason>,
    message_limit: usize,
) -> Conclusion {
    if kind.pausable {
        Conclusion::Pause { reason }
    } else {
        fail(
            "job_not_pausable",
            &format!("{} cannot pause", kind.name),
            AttemptOutcome::Failed,
            message_limit,
        )
    }
}

/// The fenced moves. Each names the token and the live statuses, releases the lease, and adds
/// `$bump` (1 for an outcome that spends a retry) to `retry_count`.
pub(crate) const FINISH: Statement = Statement::new(
    "jobs.finish",
    concat!(
        "UPDATE jobs j SET status = CASE WHEN j.status = 'cancelling' THEN 'cancelled' \
           ELSE $3::text END, result = $4, error_code = $5, error_message = $6, \
           finished_at = now(), owner = NULL, lease_expires_at = NULL, updated_at = now(), \
           retry_count = j.retry_count + $7::int \
         WHERE j.id = $1 AND j.fencing_token = $2 AND j.status IN ('running', 'cancelling') \
         RETURNING ",
        state_columns!()
    ),
);

pub(crate) const INTERRUPT: Statement = Statement::new(
    "jobs.interrupt",
    concat!(
        "UPDATE jobs j SET status = CASE WHEN j.status = 'cancelling' THEN 'cancelled' \
           ELSE 'interrupted' END, \
           finished_at = CASE WHEN j.status = 'cancelling' THEN now() END, \
           owner = NULL, lease_expires_at = NULL, updated_at = now(), \
           retry_count = j.retry_count + $3::int \
         WHERE j.id = $1 AND j.fencing_token = $2 AND j.status IN ('running', 'cancelling') \
         RETURNING ",
        state_columns!()
    ),
);

pub(crate) const PAUSE_RUNNING: Statement = Statement::new(
    "jobs.pause_running",
    concat!(
        "UPDATE jobs j SET status = 'paused', \
           pause_reason = coalesce(j.pause_reason, $3::text, 'user'), \
           owner = NULL, lease_expires_at = NULL, updated_at = now() \
         WHERE j.id = $1 AND j.fencing_token = $2 AND j.status = 'running' RETURNING ",
        state_columns!()
    ),
);

pub(crate) const QUEUE_YIELD: Statement = Statement::new(
    "jobs.queue_yield",
    concat!(
        "UPDATE jobs j SET status = 'queued', queued_at = now(), last_dispatched_at = NULL, \
           owner = NULL, lease_expires_at = NULL, updated_at = now() \
         WHERE j.id = $1 AND j.fencing_token = $2 AND j.status = 'running' RETURNING ",
        state_columns!()
    ),
);

pub(crate) const END_ATTEMPT: Statement = Statement::new(
    "jobs.end_attempt",
    "UPDATE job_attempts SET ended_at = now(), outcome = $3, error_code = $4, error_message = $5 \
     WHERE job_id = $1 AND attempt = $2",
);

/// Every statement of this module.
pub(crate) const ALL: &[Statement] = &[FINISH, INTERRUPT, PAUSE_RUNNING, QUEUE_YIELD, END_ATTEMPT];

/// Write `conclusion` for `lease` in `tx` (the caller commits), with the attempt's end and the
/// `job.state` event. The status the row has afterwards, or `None` for `Lost`: the token no longer
/// holds a live row, nothing was written.
pub(crate) async fn conclude<S: Scope>(
    tx: &mut Tx<S>,
    lease: &Lease,
    conclusion: &Conclusion,
) -> Result<Option<JobStatus>, DbError> {
    let id = lease.job_id.0;
    let token = lease.fencing_token;
    let (row, outcome, code, message) = match conclusion {
        Conclusion::Finish {
            status,
            outcome,
            result,
            error_code,
            error_message,
        } => {
            let result = result.as_ref().map(|m| Value::Object(m.clone()));
            let bump = i32::from(outcome.spends_retry());
            let row = tx
                .query_opt(
                    &FINISH,
                    &[
                        &id,
                        &token,
                        &status.as_str(),
                        &result,
                        error_code,
                        error_message,
                        &bump,
                    ],
                )
                .await?;
            (row, *outcome, error_code.clone(), error_message.clone())
        }
        Conclusion::Interrupt => {
            let bump = i32::from(AttemptOutcome::Interrupted.spends_retry());
            let row = tx.query_opt(&INTERRUPT, &[&id, &token, &bump]).await?;
            (row, AttemptOutcome::Interrupted, None, None)
        }
        Conclusion::Pause { reason } => {
            let reason = reason.map(crate::state::PauseReason::as_str);
            let row = tx
                .query_opt(&PAUSE_RUNNING, &[&id, &token, &reason])
                .await?;
            (row, AttemptOutcome::Paused, None, None)
        }
        Conclusion::Yield => {
            let row = tx.query_opt(&QUEUE_YIELD, &[&id, &token]).await?;
            (row, AttemptOutcome::Yielded, None, None)
        }
    };
    let Some(row) = row else {
        return Ok(None);
    };
    let state = StateRow::read(&row)?;
    // A job being cancelled can only end `cancelled`: the attempt records that, not what the
    // handler returned.
    let outcome = match (conclusion, state.status) {
        (Conclusion::Finish { status, .. }, JobStatus::Cancelled)
            if *status != JobStatus::Cancelled =>
        {
            AttemptOutcome::Cancelled
        }
        _ => outcome,
    };
    tx.execute(
        &END_ATTEMPT,
        &[&id, &lease.attempt, &outcome.as_str(), &code, &message],
    )
    .await?;
    emit_state(tx, &state).await?;
    if state.status == JobStatus::Queued {
        crate::control::notify_queued(tx, &state.queue).await?;
    }
    Ok(Some(state.status))
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::kinds::kind;
    use crate::state::PauseReason;

    fn spec(name: &str) -> &'static KindSpec {
        kind(name).unwrap()
    }

    fn result() -> Map<String, Value> {
        json!({"n": 3}).as_object().unwrap().clone()
    }

    fn plan(kind_name: &str, end: RunEnd, flags: Flags) -> Conclusion {
        plan_conclusion(spec(kind_name), &end, flags, 2000)
    }

    const NONE: Flags = Flags {
        cancel_requested: false,
        pause_requested: false,
    };
    const CANCEL: Flags = Flags {
        cancel_requested: true,
        pause_requested: false,
    };
    const PAUSE: Flags = Flags {
        cancel_requested: false,
        pause_requested: true,
    };

    fn finish(status: JobStatus, outcome: AttemptOutcome) -> (JobStatus, AttemptOutcome) {
        (status, outcome)
    }

    fn status_of(conclusion: &Conclusion) -> Option<(JobStatus, AttemptOutcome)> {
        match conclusion {
            Conclusion::Finish {
                status, outcome, ..
            } => Some(finish(*status, *outcome)),
            _ => None,
        }
    }

    #[test]
    fn a_returned_result_completes_and_one_with_errors_says_so() {
        let done = plan("noop", RunEnd::Returned(Outcome::Completed(result())), NONE);
        assert_eq!(
            status_of(&done),
            Some((JobStatus::Completed, AttemptOutcome::Completed))
        );
        let with_errors = plan(
            "noop",
            RunEnd::Returned(Outcome::CompletedWithErrors(result())),
            NONE,
        );
        assert_eq!(
            status_of(&with_errors),
            Some((
                JobStatus::CompletedWithErrors,
                AttemptOutcome::CompletedWithErrors
            ))
        );
    }

    #[test]
    fn a_cancel_request_wins_over_any_return_and_keeps_a_result() {
        let kept = plan(
            "noop",
            RunEnd::Returned(Outcome::Completed(result())),
            CANCEL,
        );
        assert_eq!(
            status_of(&kept),
            Some((JobStatus::Cancelled, AttemptOutcome::Cancelled))
        );
        assert!(matches!(
            kept,
            Conclusion::Finish {
                result: Some(_),
                ..
            }
        ));
        for outcome in [Outcome::Yield, Outcome::Paused(None), Outcome::Cancelled] {
            let c = plan("assist_loop", RunEnd::Returned(outcome), CANCEL);
            assert_eq!(
                status_of(&c),
                Some((JobStatus::Cancelled, AttemptOutcome::Cancelled))
            );
        }
    }

    #[test]
    fn a_handler_that_says_cancelled_without_a_request_is_a_failure_not_a_forbidden_move() {
        let c = plan("noop", RunEnd::Returned(Outcome::Cancelled), NONE);
        assert!(matches!(
            c,
            Conclusion::Finish { status: JobStatus::Failed, error_code: Some(ref code), .. } if code == "cancel_not_requested"
        ));
        let honoured = plan("noop", RunEnd::Returned(Outcome::Cancelled), CANCEL);
        assert_eq!(
            status_of(&honoured),
            Some((JobStatus::Cancelled, AttemptOutcome::Cancelled))
        );
    }

    #[test]
    fn a_failure_carries_its_code_and_a_panic_is_a_handler_error() {
        let failed = plan(
            "noop",
            RunEnd::Returned(Outcome::failed("embedding_space_refused", "no")),
            NONE,
        );
        assert!(matches!(
            failed,
            Conclusion::Finish { status: JobStatus::Failed, error_code: Some(ref c), .. } if c == "embedding_space_refused"
        ));
        let panicked = plan("noop", RunEnd::Panicked("boom".into()), NONE);
        match panicked {
            Conclusion::Finish {
                status: JobStatus::Failed,
                outcome: AttemptOutcome::Failed,
                error_code: Some(code),
                error_message: Some(message),
                ..
            } => {
                assert_eq!(code, "handler_error");
                assert!(message.contains("boom"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_soft_limit_fails_with_time_limit_and_spends_a_retry() {
        let c = plan("noop", RunEnd::TimeLimit, NONE);
        let (status, outcome) = status_of(&c).unwrap();
        assert_eq!(
            (status, outcome),
            (JobStatus::Failed, AttemptOutcome::TimeLimit)
        );
        assert!(outcome.spends_retry());
    }

    #[test]
    fn a_shutdown_interrupts_however_the_run_ended() {
        assert_eq!(
            plan("noop", RunEnd::DrainExpired, NONE),
            Conclusion::Interrupt
        );
        assert_eq!(
            plan("noop", RunEnd::Returned(Outcome::Interrupted), NONE),
            Conclusion::Interrupt
        );
    }

    #[test]
    fn a_pause_is_written_only_for_a_pausable_kind() {
        let paused = plan(
            "assist_loop",
            RunEnd::Returned(Outcome::Paused(Some(PauseReason::Budget))),
            PAUSE,
        );
        assert_eq!(
            paused,
            Conclusion::Pause {
                reason: Some(PauseReason::Budget)
            }
        );
        let slice_with_pause = plan("assist_loop", RunEnd::Returned(Outcome::Yield), PAUSE);
        assert_eq!(slice_with_pause, Conclusion::Pause { reason: None });
        let refused = plan("noop", RunEnd::Returned(Outcome::Paused(None)), PAUSE);
        assert!(matches!(
            refused,
            Conclusion::Finish { error_code: Some(ref c), .. } if c == "job_not_pausable"
        ));
    }

    #[test]
    fn a_yield_requeues_only_a_sliced_kind_and_spends_no_retry() {
        assert_eq!(
            plan("assist_loop", RunEnd::Returned(Outcome::Yield), NONE),
            Conclusion::Yield
        );
        let refused = plan("noop", RunEnd::Returned(Outcome::Yield), NONE);
        assert!(matches!(
            refused,
            Conclusion::Finish { error_code: Some(ref c), .. } if c == "yield_not_allowed"
        ));
        assert!(!AttemptOutcome::Yielded.spends_retry());
        assert!(!AttemptOutcome::Paused.spends_retry());
    }

    #[test]
    fn an_error_message_is_cut_to_the_contract_length() {
        let long = "x".repeat(5000);
        let c = plan_conclusion(
            spec("noop"),
            &RunEnd::Returned(Outcome::failed("e", long)),
            NONE,
            2000,
        );
        match c {
            Conclusion::Finish {
                error_message: Some(m),
                ..
            } => assert_eq!(m.chars().count(), 2000),
            other => panic!("{other:?}"),
        }
    }
}
