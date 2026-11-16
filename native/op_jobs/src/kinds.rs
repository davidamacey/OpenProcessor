//! The job kind table, read from the generated `contracts/json/job_kinds.json`.
//!
//! The runner keeps no copy of the kinds, the state machine or the lease timing: the generator
//! (`scripts/codegen/job_kinds_contract.py`) renders them from the Python modules the wrapper runs,
//! and a drift hook keeps the file equal to them. The file is embedded at build time.

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Duration;

use serde::Deserialize;

const CONTRACT_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../contracts/json/job_kinds.json"
));

/// Which processes a kind's rows are about: a project, the whole deployment, a target project, or
/// either.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectScope {
    /// The job belongs to a project.
    Project,
    /// The job belongs to no project; it is written in deployment scope.
    Deployment,
    /// The job belongs to a target project named in its parameters.
    Target,
    /// Either of the above.
    Either,
}

/// Which side registers the handler of a kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Runtime {
    /// The Rust runner.
    Rust,
    /// A Python claim loop.
    Python,
}

/// One row of the kind table.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct KindSpec {
    /// The kind name, as stored in `jobs.kind`.
    pub name: String,
    /// The task name Celery knew the kind by (kept until Celery is removed).
    pub task_name: String,
    /// The queue the kind is routed to.
    pub queue: String,
    /// What the job's rows belong to.
    pub project_scope: ProjectScope,
    /// The soft time limit, in seconds: a handler running longer fails with `time_limit`.
    pub soft_limit_s: u64,
    /// The hard limit, in seconds: where a process supervisor kills a job that ignores the soft one.
    pub hard_limit_s: u64,
    /// How long a job may wait `queued` before the reaper fails it, in seconds.
    pub max_queue_wait_s: u64,
    /// At most one active job per project, group and scope value; `None` means no exclusivity.
    pub exclusive_group: Option<String>,
    /// The `params` key whose value narrows the group; empty means the whole project.
    pub exclusive_scope: String,
    /// Whether the reaper resumes an `interrupted` job by itself.
    pub auto_resume: bool,
    /// The retry budget.
    pub max_attempts: i32,
    /// Whether a pause request is accepted.
    pub pausable: bool,
    /// The time slice after which the handler returns `Yield`, in seconds; `None` for kinds that
    /// run to the end.
    pub slice_s: Option<u64>,
    /// `any_card` for kinds that call the GPU executor.
    pub gpu_affinity: String,
    /// Which side registers the handler.
    pub runtime: Runtime,
}

impl KindSpec {
    /// The soft limit as a duration.
    #[must_use]
    pub fn soft_limit(&self) -> Duration {
        Duration::from_secs(self.soft_limit_s)
    }

    /// The slice as a duration.
    #[must_use]
    pub fn slice(&self) -> Option<Duration> {
        self.slice_s.map(Duration::from_secs)
    }
}

/// The status vocabulary and transitions of the contract.
#[derive(Debug, Deserialize)]
pub struct JobStates {
    /// Every status, in the Python module's order.
    pub statuses: Vec<String>,
    /// The terminal ones, sorted.
    pub terminal: Vec<String>,
    /// The active ones, sorted.
    pub active: Vec<String>,
    /// The allowed `[from, to]` moves, sorted.
    pub transitions: Vec<[String; 2]>,
}

/// The lease timing of the contract (the defaults the generator runs with).
#[derive(Debug, Deserialize)]
pub struct ContractTiming {
    /// A lease past this many seconds without a renewal has expired.
    pub lease_ttl_s: u64,
    /// Seconds between renewals.
    pub heartbeat_tick_s: u64,
    /// A holder stops writing this many seconds before its lease can expire.
    pub fence_margin_s: u64,
    /// At most one progress write per job per this many seconds, except on a stage change.
    pub progress_min_interval_s: u64,
}

/// The constants of the Python wrapper the Rust runner repeats.
#[derive(Debug, Deserialize)]
pub struct WrapperConstants {
    /// A job of a project in one of these statuses fails `project_unavailable` instead of running.
    pub unavailable_project_statuses: Vec<String>,
    /// A job's soft-limit timer fires this many seconds early (at most a tenth of the limit).
    pub soft_limit_lead_s: f64,
    /// Error messages are cut to this many characters.
    pub message_chars: usize,
}

/// The whole contract file.
#[derive(Debug, Deserialize)]
pub struct JobContract {
    /// Every kind.
    pub kinds: Vec<KindSpec>,
    /// Every queue name.
    pub queues: Vec<String>,
    /// The state machine.
    pub job_states: JobStates,
    /// The lease timing.
    pub timing: ContractTiming,
    /// The values of `jobs.pause_reason`.
    pub pause_reasons: Vec<String>,
    /// The values of `job_attempts.outcome`.
    pub attempt_outcomes: Vec<String>,
    /// The outcomes that spend a retry, sorted.
    pub retry_outcomes: Vec<String>,
    /// Advisory-lock classes by name.
    pub advisory_locks: BTreeMap<String, i32>,
    /// Wrapper constants.
    pub wrapper: WrapperConstants,
}

/// The contract, parsed once.
///
/// # Panics
///
/// If the embedded file does not parse, which the generator's drift hook and this crate's tests
/// rule out.
#[must_use]
pub fn contract() -> &'static JobContract {
    static PARSED: OnceLock<JobContract> = OnceLock::new();
    PARSED.get_or_init(|| {
        serde_json::from_str(CONTRACT_JSON)
            .unwrap_or_else(|e| panic!("contracts/json/job_kinds.json does not parse: {e}"))
    })
}

/// The kind a name refers to.
#[must_use]
pub fn kind(name: &str) -> Option<&'static KindSpec> {
    contract().kinds.iter().find(|k| k.name == name)
}

/// The class number of a named advisory lock (`BEAT`, `REAPER`, ...).
#[must_use]
pub fn advisory_lock(name: &str) -> Option<i32> {
    contract().advisory_locks.get(name).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_contract_lists_every_kind_with_its_fields() {
        let c = contract();
        assert!(c.kinds.len() >= 40, "{} kinds", c.kinds.len());
        let reprocess = kind("reprocess").unwrap();
        assert_eq!(reprocess.exclusive_group.as_deref(), Some("reprocess"));
        assert!(reprocess.auto_resume);
        assert_eq!(reprocess.soft_limit(), Duration::from_secs(24 * 3600));
        assert_eq!(reprocess.runtime, Runtime::Python);
        let label_op = kind("label_op").unwrap();
        assert_eq!(label_op.exclusive_scope, "op_id");
        assert_eq!(label_op.queue, "interactive");
        let assist = kind("assist_loop").unwrap();
        assert!(assist.pausable);
        assert_eq!(assist.slice(), Some(Duration::from_secs(600)));
        assert_eq!(
            kind("noop").unwrap().project_scope,
            ProjectScope::Deployment
        );
        assert!(kind("train").unwrap().exclusive_group.is_none());
    }

    #[test]
    fn every_kind_is_well_formed() {
        let c = contract();
        let mut names: Vec<&str> = c.kinds.iter().map(|k| k.name.as_str()).collect();
        names.dedup();
        assert_eq!(names.len(), c.kinds.len(), "kind names are unique");
        for k in &c.kinds {
            assert!(c.queues.contains(&k.queue), "{}: unknown queue", k.name);
            assert!(
                k.soft_limit_s > 0 && k.soft_limit_s < k.hard_limit_s,
                "{}",
                k.name
            );
            assert!((1..=10).contains(&k.max_attempts), "{}", k.name);
            assert!(
                k.slice_s.is_none() || k.pausable,
                "{}: slice needs pausable",
                k.name
            );
        }
    }

    #[test]
    fn the_timing_is_the_wrappers_and_the_beat_lock_is_one() {
        let t = &contract().timing;
        assert_eq!(
            (t.lease_ttl_s, t.heartbeat_tick_s, t.fence_margin_s),
            (120, 15, 30)
        );
        assert_eq!(advisory_lock("BEAT"), Some(1));
        assert_eq!(advisory_lock("REAPER"), Some(16));
    }

    #[test]
    fn exactly_the_kinds_with_a_rust_handler_say_rust() {
        let rust: Vec<&str> = contract()
            .kinds
            .iter()
            .filter(|k| k.runtime == Runtime::Rust)
            .map(|k| k.name.as_str())
            .collect();
        assert_eq!(rust, vec!["noop", "sleep_noop"]);
    }
}
