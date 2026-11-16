//! The `events` rows a job's life produces: `job.state` and `job.progress` snapshots.
//!
//! Written in the transaction of the `jobs` change that caused them, so a rolled-back change emits
//! nothing (plan 02b section 6.2). Payloads are full snapshots under the table's 7,000-byte cap,
//! never deltas. The wake-up for readers of the feed (`NOTIFY op_events`) is the events slice's.

use op_db::{DbError, Row, Scope, Statement, Tx};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::state::{JobStatus, PauseReason};
use crate::store::Control;

/// The columns every state-changing statement returns, so one reader serves them all.
macro_rules! state_columns {
    () => {
        "j.id, j.project_id, j.kind, j.queue, j.status, j.attempt, j.retry_count, \
         j.pause_reason, j.pause_requested_at IS NOT NULL AS pause_requested, \
         j.cancel_requested_at IS NOT NULL AS cancel_requested, j.error_code, \
         left(j.error_message, 500) AS error_message"
    };
}
pub(crate) use state_columns;

/// `events.payload` is capped at this many bytes as `jsonb` text.
const EVENT_PAYLOAD_MAX_BYTES: usize = 7000;

pub(crate) const EVENT_INSERT: Statement = Statement::new(
    "jobs.event_insert",
    "INSERT INTO events (project_id, job_id, type, payload) VALUES ($1, $2, $3, $4)",
);

/// A row of `state_columns!` read into plain values.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StateRow {
    pub(crate) id: Uuid,
    pub(crate) project_id: Option<Uuid>,
    pub(crate) kind: String,
    pub(crate) queue: String,
    pub(crate) status: JobStatus,
    pub(crate) attempt: i32,
    pub(crate) retry_count: i32,
    pub(crate) pause_reason: Option<PauseReason>,
    pub(crate) pause_requested: bool,
    pub(crate) cancel_requested: bool,
    pub(crate) error_code: Option<String>,
    pub(crate) error_message: Option<String>,
}

impl StateRow {
    pub(crate) fn read(row: &Row) -> Result<Self, DbError> {
        let get = |e| DbError::call("jobs.row", e);
        let status: String = row.try_get("status").map_err(get)?;
        let reason: Option<String> = row.try_get("pause_reason").map_err(get)?;
        Ok(Self {
            id: row.try_get("id").map_err(get)?,
            project_id: row.try_get("project_id").map_err(get)?,
            kind: row.try_get("kind").map_err(get)?,
            queue: row.try_get("queue").map_err(get)?,
            status: JobStatus::parse(&status).ok_or_else(|| DbError::Data {
                what: "jobs.status",
                detail: format!("unknown status {status:?}"),
            })?,
            attempt: row.try_get("attempt").map_err(get)?,
            retry_count: row.try_get("retry_count").map_err(get)?,
            pause_reason: reason.as_deref().and_then(PauseReason::parse),
            pause_requested: row.try_get("pause_requested").map_err(get)?,
            cancel_requested: row.try_get("cancel_requested").map_err(get)?,
            error_code: row.try_get("error_code").map_err(get)?,
            error_message: row.try_get("error_message").map_err(get)?,
        })
    }

    pub(crate) fn control(&self) -> Control {
        Control {
            status: self.status,
            cancel_requested: self.cancel_requested,
            pause_requested: self.pause_requested,
            pause_reason: self.pause_reason,
        }
    }

    /// The `job.state` payload (`state_payload` of `src/db/repos/job_events.py`).
    pub(crate) fn state_payload(&self) -> Value {
        json!({
            "job_id": self.id.hyphenated().to_string(),
            "kind": self.kind,
            "status": self.status.as_str(),
            "queue": self.queue,
            "attempt": self.attempt,
            "retry_count": self.retry_count,
            "pause_reason": self.pause_reason.map(PauseReason::as_str),
            "pause_requested": self.pause_requested,
            "cancel_requested": self.cancel_requested,
            "error_code": self.error_code,
            "error_message": self.error_message.as_deref().filter(|m| !m.is_empty()),
        })
    }
}

/// The length of `value` as `jsonb` text, which puts a space after `:` and `,`.
pub(crate) fn jsonb_text_len(value: &Value) -> usize {
    match value {
        Value::Array(items) => {
            2 + items.iter().map(jsonb_text_len).sum::<usize>() + 2 * items.len().saturating_sub(1)
        }
        Value::Object(map) => {
            2 + map
                .iter()
                .map(|(k, v)| jsonb_text_len(&Value::String(k.clone())) + 2 + jsonb_text_len(v))
                .sum::<usize>()
                + 2 * map.len().saturating_sub(1)
        }
        other => other.to_string().len(),
    }
}

/// The `job.progress` payload: counters that would break the size cap are dropped first.
pub(crate) fn progress_payload(job_id: Uuid, kind: &str, snapshot: &Value) -> Value {
    let full =
        json!({"job_id": job_id.hyphenated().to_string(), "kind": kind, "progress": snapshot});
    if jsonb_text_len(&full) <= EVENT_PAYLOAD_MAX_BYTES {
        return full;
    }
    let trimmed: serde_json::Map<String, Value> = ["done", "total", "stage"]
        .iter()
        .filter_map(|key| snapshot.get(*key).map(|v| ((*key).to_owned(), v.clone())))
        .collect();
    json!({"job_id": job_id.hyphenated().to_string(), "kind": kind, "progress": trimmed})
}

pub(crate) async fn insert<S: Scope>(
    tx: &mut Tx<S>,
    project_id: Option<Uuid>,
    job_id: Uuid,
    event_type: &str,
    payload: &Value,
) -> Result<(), DbError> {
    tx.execute(&EVENT_INSERT, &[&project_id, &job_id, &event_type, payload])
        .await
        .map(drop)
}

pub(crate) async fn emit_state<S: Scope>(tx: &mut Tx<S>, row: &StateRow) -> Result<(), DbError> {
    insert(
        tx,
        row.project_id,
        row.id,
        "job.state",
        &row.state_payload(),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row() -> StateRow {
        StateRow {
            id: Uuid::nil(),
            project_id: None,
            kind: "noop".into(),
            queue: "jobs".into(),
            status: JobStatus::Running,
            attempt: 1,
            retry_count: 0,
            pause_reason: None,
            pause_requested: false,
            cancel_requested: false,
            error_code: None,
            error_message: Some(String::new()),
        }
    }

    #[test]
    fn the_state_payload_has_the_python_keys() {
        let payload = row().state_payload();
        let mut keys: Vec<&str> = payload
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "attempt",
                "cancel_requested",
                "error_code",
                "error_message",
                "job_id",
                "kind",
                "pause_reason",
                "pause_requested",
                "queue",
                "retry_count",
                "status"
            ]
        );
        assert_eq!(payload["status"], "running");
        assert_eq!(
            payload["error_message"],
            Value::Null,
            "an empty message is null"
        );
        assert_eq!(payload["job_id"], "00000000-0000-0000-0000-000000000000");
    }

    #[test]
    fn jsonb_text_counts_the_spaces_postgres_adds() {
        assert_eq!(
            jsonb_text_len(&json!({"a": 1, "b": [1, 2]})),
            r#"{"a": 1, "b": [1, 2]}"#.len()
        );
        assert_eq!(jsonb_text_len(&json!([])), 2);
        assert_eq!(jsonb_text_len(&json!({})), 2);
    }

    #[test]
    fn a_progress_payload_over_the_cap_keeps_only_done_total_and_stage() {
        let small = json!({"done": 1, "total": 2, "stage": "s", "counters": {"a": 1}});
        let payload = progress_payload(Uuid::nil(), "noop", &small);
        assert_eq!(payload["progress"], small);
        let big =
            json!({"done": 1, "total": 2, "stage": "s", "counters": {"blob": "x".repeat(8000)}});
        let payload = progress_payload(Uuid::nil(), "noop", &big);
        assert_eq!(
            payload["progress"],
            json!({"done": 1, "total": 2, "stage": "s"})
        );
        assert!(jsonb_text_len(&payload) <= EVENT_PAYLOAD_MAX_BYTES);
    }
}
