//! The job conformance scenarios (`tests/job_conformance/scenarios.json`, plan 18a section 3.5)
//! replayed against the in-memory store. The same file is replayed on PostgreSQL by the database
//! tier (`db_tier`), and by the Python claim loop; all must give the same row states.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use op_db::ProjectId;
use serde_json::Value;
use uuid::Uuid;

use crate::exclusive::NewJob;
use crate::fake::FakeStore;
use crate::state::{AttemptOutcome, JobStatus};
use crate::store::{ClaimRequest, Conclusion, JobId, JobStore, Lease};

pub(crate) const SCENARIOS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/job_conformance/scenarios.json"
));

/// What a world says of an insert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Inserted {
    Created,
    Busy(Option<String>),
    Existing,
}

/// A job row as the scenarios read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RowView {
    pub status: String,
    pub attempt: i64,
    pub fencing_token: i64,
    pub retry_count: i64,
    pub owner: Option<String>,
    pub pause_reason: Option<String>,
    pub error_code: Option<String>,
    pub attempts: Vec<Option<String>>,
}

/// What reaper rules a and b did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reaped {
    pub expired: Vec<String>,
    pub requeued: usize,
    pub failed: usize,
}

/// The operations a scenario uses, on named jobs.
pub(crate) trait World {
    async fn insert(
        &mut self,
        job: &str,
        kind: &str,
        project: Option<&str>,
        params: Value,
    ) -> Inserted;
    /// Claim in claim order: `(alias, fencing token)`.
    async fn claim(&mut self, runner: &str, kinds: &[String], limit: usize) -> Vec<(String, i64)>;
    /// `Some((cancel_requested, pause_requested))`, or `None` for lost.
    async fn heartbeat(&mut self, job: &str, token: i64) -> Option<(bool, bool)>;
    /// The status after, or `None` for lost.
    async fn conclude(
        &mut self,
        job: &str,
        token: i64,
        how: &str,
        error_code: Option<&str>,
    ) -> Option<String>;
    async fn cancel(&mut self, job: &str) -> Option<String>;
    async fn request_pause(&mut self, job: &str) -> Option<String>;
    async fn resume(&mut self, job: &str) -> String;
    async fn expire_lease(&mut self, job: &str);
    async fn reap(&mut self) -> Reaped;
    async fn row(&mut self, job: &str) -> RowView;
}

/// The `how` of a `conclude` step as a [`Conclusion`].
pub(crate) fn conclusion_of(how: &str, error_code: Option<&str>) -> Conclusion {
    match how {
        "interrupt" => Conclusion::Interrupt,
        "pause" => Conclusion::Pause { reason: None },
        "yield" => Conclusion::Yield,
        finish => {
            let status = JobStatus::parse(finish.strip_prefix("finish:").expect("finish:<status>"))
                .expect("a status");
            let outcome = match status {
                JobStatus::Failed => AttemptOutcome::Failed,
                JobStatus::CompletedWithErrors => AttemptOutcome::CompletedWithErrors,
                _ => AttemptOutcome::Completed,
            };
            Conclusion::Finish {
                status,
                outcome,
                result: None,
                error_code: error_code.map(str::to_owned),
                error_message: error_code.map(|c| format!("scenario failure {c}")),
            }
        }
    }
}

fn text(step: &Value, key: &str) -> String {
    step[key]
        .as_str()
        .unwrap_or_else(|| panic!("step needs {key}: {step}"))
        .to_owned()
}

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("an array of strings: {value}"))
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

/// Replay every scenario, each on a fresh world from `fresh`.
pub(crate) async fn replay_all<W: World, F: AsyncFn() -> W>(fresh: F) {
    let doc: Value = serde_json::from_str(SCENARIOS).expect("scenarios.json parses");
    for scenario in doc["scenarios"].as_array().expect("scenarios") {
        let mut world = fresh().await;
        replay(&mut world, scenario).await;
    }
}

pub(crate) async fn replay<W: World>(world: &mut W, scenario: &Value) {
    let id = scenario["id"].as_str().unwrap_or("?");
    let mut run = Run {
        world,
        tokens: HashMap::new(),
    };
    for (n, step) in scenario["steps"]
        .as_array()
        .expect("steps")
        .iter()
        .enumerate()
    {
        run.step(step, &format!("scenario {id}, step {n}: {step}"))
            .await;
    }
}

/// One scenario's progress: the world, and the fencing tokens each job was claimed with.
struct Run<'a, W> {
    world: &'a mut W,
    /// Alias to the first and the latest token.
    tokens: HashMap<String, (i64, i64)>,
}

impl<W: World> Run<'_, W> {
    fn token(&self, step: &Value) -> i64 {
        let (first, current) = self.tokens[&text(step, "job")];
        if step["token"] == "first" {
            first
        } else {
            current
        }
    }

    async fn step(&mut self, step: &Value, at: &str) {
        match step["op"].as_str().expect("op") {
            "insert" => self.insert(step, at).await,
            "claim" => self.claim(step, at).await,
            "heartbeat" => self.heartbeat(step, at).await,
            "conclude" => self.conclude(step, at).await,
            "cancel" => {
                let got = self.world.cancel(&text(step, "job")).await;
                assert_eq!(
                    got.as_deref().unwrap_or("none"),
                    step["expect"].as_str().unwrap(),
                    "{at}"
                );
            }
            "request_pause" => {
                let got = self.world.request_pause(&text(step, "job")).await;
                assert_eq!(
                    got.as_deref().unwrap_or("none"),
                    step["expect"].as_str().unwrap(),
                    "{at}"
                );
            }
            "resume" => {
                let got = self.world.resume(&text(step, "job")).await;
                assert_eq!(got, step["expect"].as_str().unwrap(), "{at}");
            }
            "expire_lease" => self.world.expire_lease(&text(step, "job")).await,
            "reap" => self.reap(step, at).await,
            "expect" => expect_row(&self.world.row(&text(step, "job")).await, step, at),
            other => panic!("unknown op {other}"),
        }
    }

    async fn insert(&mut self, step: &Value, at: &str) {
        let params = step
            .get("params")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let got = self
            .world
            .insert(
                &text(step, "job"),
                &text(step, "kind"),
                step["project"].as_str(),
                params,
            )
            .await;
        let want = match step["expect"].as_str().unwrap() {
            "created" => Inserted::Created,
            "existing" => Inserted::Existing,
            "busy" => Inserted::Busy(step["active"].as_str().map(str::to_owned)),
            other => panic!("unknown insert expectation {other}"),
        };
        assert_eq!(got, want, "{at}");
    }

    async fn claim(&mut self, step: &Value, at: &str) {
        let limit = usize::try_from(step["limit"].as_u64().unwrap()).unwrap();
        let got = self
            .world
            .claim(&text(step, "runner"), &strings(&step["kinds"]), limit)
            .await;
        for (alias, token) in &got {
            let entry = self.tokens.entry(alias.clone()).or_insert((*token, *token));
            entry.1 = *token;
        }
        let aliases: Vec<String> = got.into_iter().map(|(a, _)| a).collect();
        assert_eq!(aliases, strings(&step["expect"]), "{at}");
    }

    async fn heartbeat(&mut self, step: &Value, at: &str) {
        let token = self.token(step);
        let got = self.world.heartbeat(&text(step, "job"), token).await;
        if step["expect"] == "lost" {
            assert_eq!(got, None, "{at}");
            return;
        }
        let (cancel, pause) = got.unwrap_or_else(|| panic!("lost, expected ok: {at}"));
        for (key, seen) in [("cancel_requested", cancel), ("pause_requested", pause)] {
            if let Some(want) = step[key].as_bool() {
                assert_eq!(seen, want, "{key}: {at}");
            }
        }
    }

    async fn conclude(&mut self, step: &Value, at: &str) {
        let token = self.token(step);
        let got = self
            .world
            .conclude(
                &text(step, "job"),
                token,
                &text(step, "how"),
                step["error_code"].as_str(),
            )
            .await;
        if step["expect"] == "lost" {
            assert_eq!(got, None, "{at}");
        } else {
            assert_eq!(got.as_deref(), step["status_after"].as_str(), "{at}");
        }
    }

    async fn reap(&mut self, step: &Value, at: &str) {
        let got = self.world.reap().await;
        let want = &step["expect"];
        assert_eq!(got.expired, strings(&want["expired"]), "{at}");
        assert_eq!(
            got.requeued as u64,
            want["requeued"].as_u64().unwrap(),
            "{at}"
        );
        assert_eq!(got.failed as u64, want["failed"].as_u64().unwrap(), "{at}");
    }
}

fn expect_row(row: &RowView, step: &Value, at: &str) {
    let number = |key: &str, got: i64| {
        if let Some(want) = step[key].as_i64() {
            assert_eq!(got, want, "{key}: {at}");
        }
    };
    let optional = |key: &str, got: &Option<String>| {
        if let Some(want) = step.get(key) {
            assert_eq!(got.as_deref(), want.as_str(), "{key}: {at}");
        }
    };
    if let Some(want) = step["status"].as_str() {
        assert_eq!(row.status, want, "status: {at}");
    }
    number("attempt", row.attempt);
    number("fencing_token", row.fencing_token);
    number("retry_count", row.retry_count);
    optional("owner", &row.owner);
    optional("pause_reason", &row.pause_reason);
    optional("error_code", &row.error_code);
    if let Some(want) = step.get("attempts") {
        let want: Vec<Option<String>> = want
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().map(str::to_owned))
            .collect();
        assert_eq!(row.attempts, want, "attempts: {at}");
    }
}

// --- the in-memory world ----------------------------------------------------------------------

struct FakeWorld {
    store: Arc<FakeStore>,
    jobs: HashMap<String, JobId>,
    names: HashMap<JobId, String>,
    leases: HashMap<String, Lease>,
    runners: HashMap<String, String>,
}

impl FakeWorld {
    fn new() -> Self {
        Self {
            store: Arc::new(FakeStore::new()),
            jobs: HashMap::new(),
            names: HashMap::new(),
            leases: HashMap::new(),
            runners: HashMap::new(),
        }
    }

    fn id(&self, job: &str) -> JobId {
        self.jobs[job]
    }
}

fn project_id(alias: &str) -> ProjectId {
    ProjectId::new(Uuid::from_u128(
        alias.bytes().fold(7, |acc, b| acc * 31 + u128::from(b)),
    ))
}

impl World for FakeWorld {
    async fn insert(
        &mut self,
        job: &str,
        kind: &str,
        project: Option<&str>,
        params: Value,
    ) -> Inserted {
        let new = NewJob::for_kind(
            kind,
            project.map(project_id),
            params.clone(),
            None,
            "scenario",
        )
        .unwrap();
        let group = new
            .exclusive_group
            .as_deref()
            .map(|g| (g.to_owned(), format!("{:?}|{}", new.project, new.scope)));
        let group_ref = group.as_ref().map(|(g, s)| (g.as_str(), s.as_str()));
        match self.store.insert(kind, params, group_ref) {
            Ok(id) => {
                self.jobs.insert(job.to_owned(), id);
                self.names.insert(id, job.to_owned());
                Inserted::Created
            }
            Err(_) => {
                let active = self.names.iter().find_map(|(id, name)| {
                    let j = self.store.job(*id);
                    (j.status.is_active() && j.group == group).then(|| name.clone())
                });
                Inserted::Busy(active)
            }
        }
    }

    async fn claim(&mut self, runner: &str, kinds: &[String], limit: usize) -> Vec<(String, i64)> {
        let owner = format!("owner-of-{runner}");
        self.runners.insert(owner.clone(), runner.to_owned());
        let claimed = self
            .store
            .claim(&ClaimRequest {
                kinds: kinds.to_vec(),
                limit,
                owner,
                lease_ttl: Duration::from_secs(120),
            })
            .await
            .unwrap();
        claimed
            .into_iter()
            .map(|c| {
                self.leases
                    .insert(self.names[&c.lease.job_id].clone(), c.lease);
                (self.names[&c.lease.job_id].clone(), c.lease.fencing_token)
            })
            .collect()
    }

    async fn heartbeat(&mut self, job: &str, token: i64) -> Option<(bool, bool)> {
        let lease = Lease {
            fencing_token: token,
            ..self.leases[job]
        };
        self.store
            .heartbeat(&lease, Duration::from_secs(120))
            .await
            .unwrap()
            .map(|c| (c.cancel_requested, c.pause_requested))
    }

    async fn conclude(
        &mut self,
        job: &str,
        token: i64,
        how: &str,
        error_code: Option<&str>,
    ) -> Option<String> {
        let lease = Lease {
            fencing_token: token,
            ..self.leases[job]
        };
        self.store
            .conclude(&lease, &conclusion_of(how, error_code))
            .await
            .unwrap()
            .map(|s| s.as_str().to_owned())
    }

    async fn cancel(&mut self, job: &str) -> Option<String> {
        let id = self.id(job);
        (!self.store.status(id).is_terminal()).then(|| self.store.cancel(id).as_str().to_owned())
    }

    async fn request_pause(&mut self, job: &str) -> Option<String> {
        let id = self.id(job);
        matches!(
            self.store.status(id),
            JobStatus::Queued | JobStatus::Running | JobStatus::Paused
        )
        .then(|| self.store.request_pause_route(id).as_str().to_owned())
    }

    async fn resume(&mut self, job: &str) -> String {
        match self.store.resume(self.id(job)) {
            Ok(()) => "queued".into(),
            Err("busy") => "busy".into(),
            Err(_) => "not_resumable".into(),
        }
    }

    async fn expire_lease(&mut self, job: &str) {
        self.store.expire_lease(self.id(job));
    }

    async fn reap(&mut self) -> Reaped {
        let expired = self
            .store
            .reap_expired()
            .into_iter()
            .map(|id| self.names[&id].clone())
            .collect();
        let (requeued, failed) = self.store.resume_interrupted();
        Reaped {
            expired,
            requeued,
            failed,
        }
    }

    async fn row(&mut self, job: &str) -> RowView {
        let j = self.store.job(self.id(job));
        RowView {
            status: j.status.as_str().to_owned(),
            attempt: i64::from(j.attempt),
            fencing_token: j.fencing_token,
            retry_count: i64::from(j.retry_count),
            owner: j.owner.map(|o| self.runners.get(&o).cloned().unwrap_or(o)),
            pause_reason: j.pause_reason.map(|r| r.as_str().to_owned()),
            error_code: j.error_code,
            attempts: j.attempts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn every_scenario_holds_on_the_in_memory_store() {
        replay_all(async || FakeWorld::new()).await;
    }

    #[test]
    fn the_scenario_file_is_well_formed_and_covers_what_the_slice_names() {
        let doc: Value = serde_json::from_str(SCENARIOS).unwrap();
        let scenarios = doc["scenarios"].as_array().unwrap();
        let mut ids: Vec<&str> = scenarios
            .iter()
            .map(|s| s["id"].as_str().unwrap())
            .collect();
        let total = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "scenario ids are unique");
        for wanted in [
            "claim_race_two_runners",
            "lease_expiry_takeover_and_the_dead_workers_late_write",
            "cancel_running_job",
            "pause_a_running_pausable_job_and_resume_it",
            "yield_requeues_behind_other_work_without_a_retry",
            "exclusive_group_refuses_a_second_active_job",
            "auto_resume_after_an_interruption",
        ] {
            assert!(ids.contains(&wanted), "missing {wanted}");
        }
        let known = [
            "insert",
            "claim",
            "heartbeat",
            "conclude",
            "cancel",
            "request_pause",
            "resume",
            "expire_lease",
            "reap",
            "expect",
        ];
        for scenario in scenarios {
            for step in scenario["steps"].as_array().unwrap() {
                assert!(known.contains(&step["op"].as_str().unwrap()), "{step}");
                if let Some(kind) = step["kind"].as_str() {
                    assert!(crate::kinds::kind(kind).is_some(), "unknown kind {kind}");
                }
            }
        }
    }
}
