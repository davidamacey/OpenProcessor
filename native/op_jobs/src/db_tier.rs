//! The database tier (cargo feature `db`; Docker needed): the same job statements, on PostgreSQL
//! with the schema head loaded by `op_db::testing`. Run with
//! `cargo test -p op_jobs --features db --lib db_tier -- --test-threads=1`.
//!
//! Every test starts its own container, so the tests are slow and independent. Seeding and
//! read-back go through the table owner's connection, which bypasses row-level security; the code
//! under test connects as the application role, which does not.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use op_db::testing::{RawClient, TestDatabase};
use op_db::{ListenEvents, Listener, Pool, Profile, ProjectId, deployment_tx, project_tx};
use serde_json::{Value, json};
use tokio::sync::watch;
use uuid::Uuid;

use crate::conformance::{Inserted, Reaped, RowView, World, conclusion_of, replay_all};
use crate::control::{self, Resumed};
use crate::exclusive::{self, NewJob};
use crate::handlers::register_builtin;
use crate::runner::{RunnerConfig, run_worker};
use crate::state::{JobStatus, PauseReason};
use crate::store::{ClaimRequest, Conclusion, JobId, JobStore, Lease};
use crate::watch::LeaseTiming;
use crate::{Handlers, PgJobStore, lease};

const OWNER_RESET: &str = "TRUNCATE jobs CASCADE";

/// The project `alias` (its slug), created through the owner's connection when it is not there.
pub(crate) async fn seed_project(owner: &RawClient, alias: &str) -> ProjectId {
    let found = owner
        .query_opt("SELECT id FROM projects WHERE slug = $1", &[&alias])
        .await
        .unwrap();
    if let Some(row) = found {
        return ProjectId::new(row.get::<_, Uuid>(0));
    }
    owner
        .execute(
            "INSERT INTO project_slugs (slug) VALUES ($1) ON CONFLICT DO NOTHING",
            &[&alias],
        )
        .await
        .unwrap();
    let row = owner
        .query_one(
            "INSERT INTO projects (slug, display_name, status, index_prefix, storage_prefix, \
             model_prefix, mlflow_experiment, created_by) VALUES ($1, $1, 'active', \
             'op_' || $1, $1, 'm_' || $1, 'e_' || $1, 'test') RETURNING id",
            &[&alias],
        )
        .await
        .unwrap();
    ProjectId::new(row.get::<_, Uuid>(0))
}

struct PgWorld {
    pool: Pool,
    store: PgJobStore,
    owner: RawClient,
    jobs: HashMap<String, JobId>,
    names: HashMap<JobId, String>,
    leases: HashMap<String, Lease>,
}

impl PgWorld {
    async fn new(db: &TestDatabase) -> Self {
        let pool = db.pool(Profile::Job, 4).await;
        let owner = db.owner().await;
        owner.batch_execute(OWNER_RESET).await.unwrap();
        Self {
            store: PgJobStore::new(pool.clone()),
            pool,
            owner,
            jobs: HashMap::new(),
            names: HashMap::new(),
            leases: HashMap::new(),
        }
    }

    async fn project(&self, alias: &str) -> ProjectId {
        seed_project(&self.owner, alias).await
    }

    fn id(&self, alias: &str) -> JobId {
        self.jobs[alias]
    }

    fn lease_with(&self, alias: &str, token: i64) -> Lease {
        Lease {
            fencing_token: token,
            ..self.leases[alias]
        }
    }
}

impl World for PgWorld {
    async fn insert(
        &mut self,
        job: &str,
        kind: &str,
        project: Option<&str>,
        params: Value,
    ) -> Inserted {
        let project = match project {
            Some(alias) => Some(self.project(alias).await),
            None => None,
        };
        let new = NewJob::for_kind(kind, project, params, None, "scenario").unwrap();
        let inserted = match project {
            Some(id) => {
                let mut tx = project_tx(&self.pool, id).await.unwrap();
                let out = exclusive::insert_queued(&mut tx, &new).await.unwrap();
                tx.commit().await.unwrap();
                out
            }
            None => {
                let mut tx = deployment_tx(&self.pool, "test insert").await.unwrap();
                let out = exclusive::insert_queued(&mut tx, &new).await.unwrap();
                tx.commit().await.unwrap();
                out
            }
        };
        match inserted {
            exclusive::Inserted::Created(id) => {
                self.jobs.insert(job.to_owned(), id);
                self.names.insert(id, job.to_owned());
                Inserted::Created
            }
            exclusive::Inserted::Busy { active } => {
                Inserted::Busy(active.and_then(|id| self.names.get(&id).cloned()))
            }
            exclusive::Inserted::Existing { .. } => Inserted::Existing,
        }
    }

    async fn claim(&mut self, runner: &str, kinds: &[String], limit: usize) -> Vec<(String, i64)> {
        let claimed = self
            .store
            .claim(&ClaimRequest {
                kinds: kinds.to_vec(),
                limit,
                owner: runner.to_owned(),
                lease_ttl: Duration::from_secs(120),
            })
            .await
            .unwrap();
        claimed
            .into_iter()
            .map(|c| {
                let alias = self.names[&c.lease.job_id].clone();
                self.leases.insert(alias.clone(), c.lease);
                (alias, c.lease.fencing_token)
            })
            .collect()
    }

    async fn heartbeat(&mut self, job: &str, token: i64) -> Option<(bool, bool)> {
        self.store
            .heartbeat(&self.lease_with(job, token), Duration::from_secs(120))
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
        self.store
            .conclude(
                &self.lease_with(job, token),
                &conclusion_of(how, error_code),
            )
            .await
            .unwrap()
            .map(|s| s.as_str().to_owned())
    }

    async fn cancel(&mut self, job: &str) -> Option<String> {
        let mut tx = deployment_tx(&self.pool, "test cancel").await.unwrap();
        let out = control::cancel(&mut tx, self.id(job), "scenario")
            .await
            .unwrap();
        tx.commit().await.unwrap();
        out.map(|c| c.status.as_str().to_owned())
    }

    async fn request_pause(&mut self, job: &str) -> Option<String> {
        let mut tx = deployment_tx(&self.pool, "test pause").await.unwrap();
        let out = control::pause(&mut tx, self.id(job), PauseReason::User)
            .await
            .unwrap();
        tx.commit().await.unwrap();
        out.map(|c| c.status.as_str().to_owned())
    }

    async fn resume(&mut self, job: &str) -> String {
        let mut tx = deployment_tx(&self.pool, "test resume").await.unwrap();
        let out = control::resume(&mut tx, self.id(job)).await.unwrap();
        tx.commit().await.unwrap();
        match out {
            Resumed::Queued(_) => "queued",
            Resumed::NotResumable => "not_resumable",
            Resumed::Busy => "busy",
        }
        .to_owned()
    }

    async fn expire_lease(&mut self, job: &str) {
        let id = self.id(job).0;
        self.owner
            .execute(
                "UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = $1",
                &[&id],
            )
            .await
            .unwrap();
    }

    async fn reap(&mut self) -> Reaped {
        let mut tx = deployment_tx(&self.pool, "test reaper rule a")
            .await
            .unwrap();
        let ended = lease::expire_leases(&mut tx, 200).await.unwrap();
        tx.commit().await.unwrap();
        let mut tx = deployment_tx(&self.pool, "test reaper rule b")
            .await
            .unwrap();
        let report = exclusive::resume_interrupted(&mut tx, Duration::ZERO, 200)
            .await
            .unwrap();
        tx.commit().await.unwrap();
        Reaped {
            expired: ended
                .into_iter()
                .map(|e| self.names[&e.job_id].clone())
                .collect(),
            requeued: report.requeued,
            failed: report.failed,
        }
    }

    async fn row(&mut self, job: &str) -> RowView {
        let id = self.id(job).0;
        let r = self
            .owner
            .query_one(
                "SELECT status, attempt, fencing_token, retry_count, owner, pause_reason, error_code \
                 FROM jobs WHERE id = $1",
                &[&id],
            )
            .await
            .unwrap();
        let attempts = self
            .owner
            .query(
                "SELECT outcome FROM job_attempts WHERE job_id = $1 ORDER BY attempt",
                &[&id],
            )
            .await
            .unwrap();
        RowView {
            status: r.get(0),
            attempt: i64::from(r.get::<_, i32>(1)),
            fencing_token: r.get(2),
            retry_count: i64::from(r.get::<_, i32>(3)),
            owner: r.get(4),
            pause_reason: r.get(5),
            error_code: r.get(6),
            attempts: attempts
                .iter()
                .map(|a| a.get::<_, Option<String>>(0))
                .collect(),
        }
    }
}

pub(crate) fn noop_job() -> NewJob {
    NewJob::for_kind("noop", None, json!({}), None, "test").unwrap()
}

pub(crate) async fn insert_noops(pool: &Pool, count: usize) {
    for _ in 0..count {
        let mut tx = deployment_tx(pool, "test seed").await.unwrap();
        exclusive::insert_queued(&mut tx, &noop_job())
            .await
            .unwrap();
        tx.commit().await.unwrap();
    }
}

pub(crate) fn claim_request(owner: &str, limit: usize) -> ClaimRequest {
    ClaimRequest {
        kinds: vec!["noop".into()],
        limit,
        owner: owner.into(),
        lease_ttl: Duration::from_secs(120),
    }
}

/// Owed to the sweep: `cargo test -p op_jobs --features db --lib db_tier::every_scenario -- --test-threads=1`
/// passes when every scenario of `tests/job_conformance/scenarios.json` holds on PostgreSQL.
#[tokio::test(flavor = "multi_thread")]
async fn every_scenario_holds_on_postgresql() {
    let db = TestDatabase::start().await;
    replay_all(async || PgWorld::new(&db).await).await;
}

/// Passes when every registered statement prepares against the schema head.
#[tokio::test(flavor = "multi_thread")]
async fn every_statement_of_the_registry_prepares() {
    let db = TestDatabase::start().await;
    db.prepare_all(op_db::STATEMENTS).await;
    db.prepare_all(&crate::statements()).await;
}

/// Passes when two runners racing for 100 rows claim every row exactly once.
#[tokio::test(flavor = "multi_thread")]
async fn two_runners_racing_for_100_rows_claim_each_once() {
    let db = TestDatabase::start().await;
    let pool = db.pool(Profile::Job, 6).await;
    insert_noops(&pool, 100).await;
    let mut racers = Vec::new();
    for name in ["r1", "r2"] {
        let store = PgJobStore::new(pool.clone());
        racers.push(tokio::spawn(async move {
            let mut mine = Vec::new();
            loop {
                let batch = store.claim(&claim_request(name, 7)).await.unwrap();
                if batch.is_empty() {
                    return mine;
                }
                mine.extend(
                    batch
                        .into_iter()
                        .map(|c| (c.lease.job_id, c.lease.fencing_token)),
                );
            }
        }));
    }
    let mut all = Vec::new();
    for racer in racers {
        let mine = racer.await.unwrap();
        assert!(!mine.is_empty(), "both runners got work");
        all.extend(mine);
    }
    assert_eq!(all.len(), 100);
    all.sort();
    all.dedup();
    assert_eq!(all.len(), 100, "no job was claimed twice");
    assert!(all.iter().all(|(_, token)| *token == 1));
    let owner = db.owner().await;
    let running: i64 = owner
        .query_one(
            "SELECT count(*) FROM jobs WHERE status = 'running' AND attempt = 1",
            &[],
        )
        .await
        .unwrap()
        .get(0);
    assert_eq!(running, 100);
}

/// Passes when exactly one of eight simultaneous starts of an exclusive group is created.
#[tokio::test(flavor = "multi_thread")]
async fn eight_simultaneous_starts_of_one_group_create_exactly_one_job() {
    let db = TestDatabase::start().await;
    let pool = db.pool(Profile::Job, 8).await;
    let mut starts = Vec::new();
    for _ in 0..8 {
        let pool = pool.clone();
        starts.push(tokio::spawn(async move {
            let new = NewJob::for_kind("persons_cluster", None, json!({}), None, "test").unwrap();
            let mut tx = deployment_tx(&pool, "test race").await.unwrap();
            let out = exclusive::insert_queued(&mut tx, &new).await.unwrap();
            tx.commit().await.unwrap();
            out
        }));
    }
    let mut created = 0;
    for start in starts {
        match start.await.unwrap() {
            exclusive::Inserted::Created(_) => created += 1,
            exclusive::Inserted::Busy { .. } => {}
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(created, 1);
}

/// Passes when a runner killed mid-job loses it after the lease expires and its late write is
/// refused. Real time: about 6 s.
#[tokio::test(flavor = "multi_thread")]
async fn a_killed_runner_loses_its_job_after_the_lease_and_its_late_write_is_refused() {
    let db = TestDatabase::start().await;
    let pool = db.pool(Profile::Job, 6).await;
    let store = Arc::new(PgJobStore::new(pool.clone()));
    let mut tx = deployment_tx(&pool, "test seed").await.unwrap();
    let new = NewJob::for_kind("sleep_noop", None, json!({"seconds": 600}), None, "test").unwrap();
    let exclusive::Inserted::Created(id) = exclusive::insert_queued(&mut tx, &new).await.unwrap()
    else {
        panic!("not created");
    };
    tx.commit().await.unwrap();

    let short = LeaseTiming {
        ttl: Duration::from_secs(3),
        tick: Duration::from_secs(1),
        fence_margin: Duration::from_secs(1),
    };
    let config = |owner: &str| RunnerConfig {
        owner: owner.into(),
        max_running: 1,
        poll_interval: Duration::from_millis(200),
        timing: short,
        progress_min_interval: Duration::from_secs(1),
        drain: Duration::from_secs(1),
    };
    let mut handlers = Handlers::new();
    register_builtin(&mut handlers).unwrap();
    let (_stop, shutdown) = watch::channel(false);
    let (_wake, events) = tokio::sync::mpsc::channel(1);
    let victim = tokio::spawn(run_worker(
        Arc::clone(&store),
        handlers,
        config("victim"),
        events,
        shutdown,
    ));
    let owner = db.owner().await;
    let uuid = id.0;
    for _ in 0..50 {
        let status: String = owner
            .query_one("SELECT status FROM jobs WHERE id = $1", &[&uuid])
            .await
            .unwrap()
            .get(0);
        if status == "running" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let old = Lease {
        job_id: id,
        project_id: None,
        fencing_token: 1,
        attempt: 1,
    };
    victim.abort(); // the process dies: nothing renews the lease, nothing is written
    tokio::time::sleep(Duration::from_secs(4)).await;
    let mut tx = deployment_tx(&pool, "test reaper").await.unwrap();
    let ended = lease::expire_leases(&mut tx, 10).await.unwrap();
    tx.commit().await.unwrap();
    assert_eq!(ended.len(), 1);
    assert_eq!(ended[0].status, JobStatus::Interrupted);
    let mut tx = deployment_tx(&pool, "test resume").await.unwrap();
    assert!(matches!(
        control::resume(&mut tx, id).await.unwrap(),
        Resumed::Queued(_)
    ));
    tx.commit().await.unwrap();
    let taken = store
        .claim(&claim_request_kind("sleep_noop", "taker", 1))
        .await
        .unwrap();
    assert_eq!(
        (taken[0].lease.fencing_token, taken[0].lease.attempt),
        (2, 2)
    );
    let late = Conclusion::Finish {
        status: JobStatus::Completed,
        outcome: crate::state::AttemptOutcome::Completed,
        result: None,
        error_code: None,
        error_message: None,
    };
    assert_eq!(store.conclude(&old, &late).await.unwrap(), None);
    assert!(
        store
            .heartbeat(&old, Duration::from_secs(3))
            .await
            .unwrap()
            .is_none()
    );
    let row = owner
        .query_one(
            "SELECT status, owner, fencing_token, retry_count FROM jobs WHERE id = $1",
            &[&uuid],
        )
        .await
        .unwrap();
    assert_eq!(row.get::<_, String>(0), "running");
    assert_eq!(row.get::<_, String>(1), "taker");
    assert_eq!((row.get::<_, i64>(2), row.get::<_, i32>(3)), (2, 1));
}

pub(crate) fn claim_request_kind(kind: &str, owner: &str, limit: usize) -> ClaimRequest {
    ClaimRequest {
        kinds: vec![kind.into()],
        limit,
        owner: owner.into(),
        lease_ttl: Duration::from_secs(120),
    }
}

/// Passes when a job inserted while a runner idles is claimed and finished within two seconds of
/// the commit through `NOTIFY` alone (the poll is an hour), and prints the measured time.
#[tokio::test(flavor = "multi_thread")]
async fn notify_to_claim_latency() {
    let db = TestDatabase::start().await;
    let pool = db.pool(Profile::Job, 6).await;
    let store = Arc::new(PgJobStore::new(pool.clone()));
    let mut handlers = Handlers::new();
    register_builtin(&mut handlers).unwrap();
    let mut config = RunnerConfig::from_env(&op_config::Env::default(), "latency".into()).unwrap();
    config.poll_interval = Duration::from_secs(3600);
    let mut listener = Listener::spawn_pg(db.settings(), "op-test-listen", &["op_jobs"]);
    let (_stop, shutdown) = watch::channel(false);
    let (forward, events) = tokio::sync::mpsc::channel(8);
    // The runner reads the listener through a channel here so the test can also time the wake-up.
    let pump = tokio::spawn(async move {
        while let Some(event) = ListenEvents::recv(&mut listener).await {
            if forward.send(event).await.is_err() {
                return;
            }
        }
    });
    tokio::spawn(run_worker(
        Arc::clone(&store),
        handlers,
        config,
        events,
        shutdown,
    ));
    tokio::time::sleep(Duration::from_secs(1)).await; // the first claim found nothing; LISTEN is up

    let owner = db.owner().await;
    let mut tx = deployment_tx(&pool, "test insert").await.unwrap();
    let exclusive::Inserted::Created(id) = exclusive::insert_queued(&mut tx, &noop_job())
        .await
        .unwrap()
    else {
        panic!("not created");
    };
    tx.commit().await.unwrap();
    let committed = Instant::now();
    let uuid = id.0;
    loop {
        let status: String = owner
            .query_one("SELECT status FROM jobs WHERE id = $1", &[&uuid])
            .await
            .unwrap()
            .get(0);
        if status == "completed" {
            break;
        }
        assert!(
            committed.elapsed() < Duration::from_secs(2),
            "not finished 2 s after the commit: {status}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    println!(
        "notify_to_claim_latency: commit to completed in {:?}",
        committed.elapsed()
    );
    pump.abort();
}

/// Passes when another project's scope cannot see, heartbeat or finish a project's job.
#[tokio::test(flavor = "multi_thread")]
async fn a_job_of_one_project_is_invisible_to_another_projects_scope() {
    let db = TestDatabase::start().await;
    let mut world = PgWorld::new(&db).await;
    world.insert("a", "autolabel", Some("p1"), json!({})).await;
    let other = world.project("p2").await;
    let claimed = world.claim("r1", &["autolabel".to_owned()], 1).await;
    assert_eq!(claimed.len(), 1);
    let own = world.leases["a"];
    assert!(own.project_id.is_some());
    let wrong_scope = Lease {
        project_id: Some(other),
        ..own
    };
    let store = PgJobStore::new(world.pool.clone());
    assert!(
        store
            .heartbeat(&wrong_scope, Duration::from_secs(120))
            .await
            .unwrap()
            .is_none()
    );
    let finish = conclusion_of("finish:completed", None);
    assert_eq!(store.conclude(&wrong_scope, &finish).await.unwrap(), None);
    assert!(
        store
            .heartbeat(&own, Duration::from_secs(120))
            .await
            .unwrap()
            .is_some()
    );
}

/// Passes when a queued job of an archived project fails at claim, and one whose retry budget is
/// spent fails instead of running.
#[tokio::test(flavor = "multi_thread")]
async fn a_job_that_can_never_run_fails_at_claim() {
    let db = TestDatabase::start().await;
    let mut world = PgWorld::new(&db).await;
    world
        .insert("gone", "autolabel", Some("p1"), json!({}))
        .await;
    world.insert("spent", "noop", None, json!({})).await;
    world
        .owner
        .batch_execute("UPDATE projects SET status = 'archived' WHERE slug = 'p1'; UPDATE jobs SET retry_count = max_attempts WHERE kind = 'noop'")
        .await
        .unwrap();
    let got = world
        .claim("r1", &["autolabel".to_owned(), "noop".to_owned()], 5)
        .await;
    assert!(got.is_empty());
    let gone = world.row("gone").await;
    assert_eq!(
        (gone.status.as_str(), gone.error_code.as_deref()),
        ("failed", Some("project_unavailable"))
    );
    let spent = world.row("spent").await;
    assert_eq!(
        (spent.status.as_str(), spent.error_code.as_deref()),
        ("failed", Some("attempts_exhausted"))
    );
}

/// Passes when a job's life leaves its state events: queued, running, completed.
#[tokio::test(flavor = "multi_thread")]
async fn a_jobs_life_writes_its_state_events_in_order() {
    let db = TestDatabase::start().await;
    let mut world = PgWorld::new(&db).await;
    world.insert("a", "noop", None, json!({})).await;
    let claimed = world.claim("r1", &["noop".to_owned()], 1).await;
    assert_eq!(claimed.len(), 1);
    world
        .conclude("a", 1, "finish:completed", None)
        .await
        .unwrap();
    let id = world.id("a").0;
    let rows = world
        .owner
        .query(
            "SELECT type, payload->>'status' FROM events WHERE job_id = $1 ORDER BY id",
            &[&id],
        )
        .await
        .unwrap();
    let seen: Vec<(String, String)> = rows.iter().map(|r| (r.get(0), r.get(1))).collect();
    let want: Vec<(String, String)> = ["queued", "running", "completed"]
        .iter()
        .map(|s| ("job.state".to_owned(), (*s).to_owned()))
        .collect();
    assert_eq!(seen, want);
}
