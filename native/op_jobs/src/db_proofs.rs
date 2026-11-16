//! More database-tier proofs (cargo feature `db`; Docker needed): what a person's jobs depend on
//! when the system is busy or hurt. The plan of the claim on a big queue, many runners and many
//! jobs, a database connection that is cut, and cancel and pause requests from another connection.
//! Run with `cargo test -p op_jobs --features db --lib db_proofs -- --test-threads=1`.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use op_db::testing::{RawClient, TestDatabase};
use op_db::{Listener, Pool, Profile, Statement, deployment_tx};
use serde_json::{Value, json};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::control::{self, Resumed};
use crate::db_tier::{claim_request_kind, noop_job, seed_project};
use crate::exclusive::{self, NewJob};
use crate::handlers::register_builtin;
use crate::runner::{RunnerConfig, run_worker};
use crate::state::{AttemptOutcome, JobStatus, PauseReason};
use crate::store::{Conclusion, JobId, JobStore, Lease};
use crate::watch::{LeaseTiming, LeaseWatch, heartbeat_loop};
use crate::{Handlers, PgJobStore, lease};

/// A lease short enough to expire inside a test: 6 s, renewed every second, with a 2 s margin.
const SHORT: LeaseTiming = LeaseTiming {
    ttl: Duration::from_secs(6),
    tick: Duration::from_secs(1),
    fence_margin: Duration::from_secs(2),
};

/// A poll interval so long that only a notification, or a reconnect, can wake a runner.
const NO_POLL: Duration = Duration::from_secs(3600);

async fn wait_until<F, Fut>(within: Duration, what: &str, mut probe: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let started = Instant::now();
    while !probe().await {
        assert!(
            started.elapsed() < within,
            "gave up after {within:?} waiting for {what}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn status_of(owner: &RawClient, id: JobId) -> String {
    owner
        .query_one("SELECT status FROM jobs WHERE id = $1", &[&id.0])
        .await
        .unwrap()
        .get(0)
}

async fn wait_status(owner: &RawClient, id: JobId, want: &str, within: Duration) {
    wait_until(within, want, || async move {
        status_of(owner, id).await == want
    })
    .await;
}

async fn queue(pool: &Pool, new: &NewJob) -> JobId {
    let mut tx = deployment_tx(pool, "test insert").await.unwrap();
    let exclusive::Inserted::Created(id) = exclusive::insert_queued(&mut tx, new).await.unwrap()
    else {
        panic!("not created");
    };
    tx.commit().await.unwrap();
    id
}

fn sleeper(seconds: u64) -> NewJob {
    NewJob::for_kind(
        "sleep_noop",
        None,
        json!({"seconds": seconds}),
        None,
        "test",
    )
    .unwrap()
}

struct Runner {
    task: JoinHandle<()>,
    stop: watch::Sender<bool>,
}

impl Runner {
    fn start(
        db: &TestDatabase,
        pool: &Pool,
        owner: &str,
        max_running: usize,
        poll: Duration,
    ) -> Self {
        let mut handlers = Handlers::new();
        register_builtin(&mut handlers).unwrap();
        let config = RunnerConfig {
            owner: owner.into(),
            max_running,
            poll_interval: poll,
            timing: SHORT,
            progress_min_interval: Duration::from_secs(1),
            drain: Duration::from_secs(2),
        };
        let listener = Listener::spawn_pg(db.settings(), "op-test-listen", &["op_jobs"]);
        let (stop, shutdown) = watch::channel(false);
        let task = tokio::spawn(run_worker(
            Arc::new(PgJobStore::new(pool.clone())),
            handlers,
            config,
            listener,
            shutdown,
        ));
        Self { task, stop }
    }

    async fn stop(self) {
        self.stop.send(true).unwrap();
        self.task.await.unwrap();
    }
}

fn nodes<'a>(plan: &'a Value, out: &mut Vec<&'a Value>) {
    out.push(plan);
    for child in plan["Plans"].as_array().into_iter().flatten() {
        nodes(child, out);
    }
}

/// 100,000 queued rows of 50 projects, every fifth a `noop` and the rest `autolabel`, with
/// `queued_at` given by an SQL expression of the row number `g`. Inserted, not updated, so the
/// table has no dead rows to confuse the plan.
fn seed_queue(queued_at: &str) -> String {
    format!(
        "INSERT INTO project_slugs (slug) SELECT 'p' || g FROM generate_series(1, 50) g;\n\
         INSERT INTO projects (slug, display_name, status, index_prefix, storage_prefix, \
           model_prefix, mlflow_experiment, created_by) \
           SELECT 'p' || g, 'p' || g, 'active', 'op_p' || g, 'p' || g, 'm_p' || g, 'e_p' || g, \
             'test' FROM generate_series(1, 50) g;\n\
         INSERT INTO jobs (project_id, kind, queue, status, max_queue_wait, requested_by, queued_at) \
           SELECT p.id, CASE WHEN g % 5 = 0 THEN 'noop' ELSE 'autolabel' END, 'jobs', 'queued', \
             interval '1 hour', 'test', {queued_at} \
           FROM generate_series(1, 100000) g JOIN projects p ON p.slug = 'p' || (1 + g % 50);\n\
         ANALYZE jobs;\n"
    )
}

/// The claim's plan with its actual rows and buffers (`EXPLAIN (ANALYZE, BUFFERS)`), run as the
/// application role in a transaction that is rolled back.
async fn explain_claim(db: &TestDatabase) -> Value {
    let pool = db.pool(Profile::Job, 2).await;
    let sql: &'static str = Box::leak(
        format!(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {}",
            crate::claim::CLAIM.sql
        )
        .into_boxed_str(),
    );
    let explain = Statement::new("test.explain_claim", sql);
    let mut tx = deployment_tx(&pool, "test explain").await.unwrap();
    let kinds = vec!["noop".to_owned(), "autolabel".to_owned()];
    let rows = tx
        .query(&explain, &[&kinds, &8_i64, &"explainer", &120.0_f64])
        .await
        .unwrap();
    tx.rollback().await.unwrap();
    rows[0].get(0)
}

fn pages_of(plan: &Value) -> i64 {
    plan[0]["Plan"]["Shared Hit Blocks"].as_i64().unwrap()
        + plan[0]["Plan"]["Shared Read Blocks"].as_i64().unwrap()
}

/// Assert the plan reads `jobs` through `ix_jobs_queued` only and touches few pages; print it.
fn assert_queue_plan(plan: &Value, what: &str) {
    println!("{what}: plan {plan}");
    let mut all = Vec::new();
    nodes(&plan[0]["Plan"], &mut all);
    let on_jobs: Vec<(&str, &str)> = all
        .iter()
        .filter(|n| n["Relation Name"] == "jobs")
        .map(|n| {
            (
                n["Node Type"].as_str().unwrap_or(""),
                n["Index Name"].as_str().unwrap_or(""),
            )
        })
        .collect();
    assert!(
        on_jobs.iter().all(|(node, _)| !node.contains("Seq Scan")),
        "{what}: a table scan of jobs: {on_jobs:?}"
    );
    assert!(
        on_jobs
            .iter()
            .any(|(node, index)| *node == "Index Scan" && index.starts_with("ix_jobs_queued")),
        "{what}: the queued index is not what picks the rows: {on_jobs:?}"
    );
    let pages = pages_of(plan);
    println!("{what}: {pages} pages touched, {on_jobs:?}");
    assert!(pages < 500, "{what}: {pages} pages for a claim of 8 rows");
}

/// Passes when the claim, on a table of 100,000 queued rows of 50 projects, reads the partial
/// index of queued rows and touches a handful of pages, not the table.
#[tokio::test(flavor = "multi_thread")]
async fn the_claim_on_a_queue_of_100000_rows_uses_the_queued_index_and_no_table_scan() {
    let db = TestDatabase::start().await;
    let seed = seed_queue("now() - random() * interval '1 day'");
    db.owner().await.batch_execute(&seed).await.unwrap();
    assert_queue_plan(&explain_claim(&db).await, "distinct queued_at");
}

/// The same, when every row was queued in one transaction (one bulk start): `queued_at` is the
/// transaction's `now()`, so all 100,000 values are equal.
#[tokio::test(flavor = "multi_thread")]
async fn the_claim_stays_on_the_index_when_every_row_was_queued_at_the_same_instant() {
    let db = TestDatabase::start().await;
    let seed = seed_queue("now()");
    db.owner().await.batch_execute(&seed).await.unwrap();
    assert_queue_plan(&explain_claim(&db).await, "equal queued_at");
}

/// Passes when four runners work 2,000 short jobs: each completes exactly once, none is left,
/// and the attempts table agrees with the jobs table. Prints the claim rate (an indication only).
#[tokio::test(flavor = "multi_thread")]
async fn four_runners_finish_2000_jobs_each_exactly_once() {
    let db = TestDatabase::start().await;
    let owner = db.owner().await;
    owner
        .batch_execute(
            "INSERT INTO jobs (kind, queue, status, max_queue_wait, requested_by, queued_at) \
             SELECT 'noop', 'jobs', 'queued', interval '1 hour', 'test', now() \
             FROM generate_series(1, 2000)",
        )
        .await
        .unwrap();
    let started = Instant::now();
    let mut runners = Vec::new();
    for name in ["r1", "r2", "r3", "r4"] {
        let pool = db.pool(Profile::Job, 12).await;
        runners.push(Runner::start(
            &db,
            &pool,
            name,
            8,
            Duration::from_millis(100),
        ));
    }
    wait_until(Duration::from_secs(180), "2000 jobs to finish", || async {
        let open: i64 = owner
            .query_one(
                "SELECT count(*) FROM jobs WHERE status NOT IN ('completed', 'failed', 'cancelled')",
                &[],
            )
            .await
            .unwrap()
            .get(0);
        open == 0
    })
    .await;
    let took = started.elapsed();
    for runner in runners {
        runner.stop().await;
    }
    println!(
        "four_runners: 2000 noop jobs in {took:?}, {:.0} jobs per second",
        2000.0 / took.as_secs_f64()
    );
    let one =
        |sql: &'static str| async { owner.query_one(sql, &[]).await.unwrap().get::<_, i64>(0) };
    assert_eq!(
        one(
            "SELECT count(*) FROM jobs WHERE status = 'completed' AND attempt = 1 \
             AND fencing_token = 1 AND retry_count = 0"
        )
        .await,
        2000
    );
    assert_eq!(
        one(
            "SELECT count(*) FROM job_attempts a JOIN jobs j ON j.id = a.job_id \
             WHERE a.outcome = 'completed' AND a.attempt = j.attempt \
             AND a.fencing_token = j.fencing_token AND j.owner IS NULL"
        )
        .await,
        2000,
        "one finished attempt per job, equal to the job's own row"
    );
    assert_eq!(one("SELECT count(*) FROM job_attempts").await, 2000);
    assert_eq!(
        one(
            "SELECT count(DISTINCT job_id) FROM events WHERE type = 'job.state' \
             AND payload->>'status' = 'completed'"
        )
        .await,
        2000
    );
    let spread: Vec<i64> = owner
        .query(
            "SELECT count(*) FROM job_attempts GROUP BY owner ORDER BY owner",
            &[],
        )
        .await
        .unwrap()
        .iter()
        .map(|r| r.get(0))
        .collect();
    println!("four_runners: jobs per runner {spread:?}");
    assert!(spread.len() > 1, "the work was shared: {spread:?}");
}

/// Ends every session of the application role, as a network failure or a restart of the
/// database would.
async fn cut_connections(superuser: &RawClient) {
    superuser
        .batch_execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
             WHERE usename = 'op_app' AND pid <> pg_backend_pid()",
        )
        .await
        .unwrap();
}

/// Passes when every connection of a runner is cut for a moment (shorter than the lease): the
/// job it runs survives under the same fencing token, and its listener comes back, so a job
/// queued afterwards is claimed through `NOTIFY` alone.
#[tokio::test(flavor = "multi_thread")]
async fn a_short_cut_of_every_connection_is_survived_and_the_listener_comes_back() {
    let db = TestDatabase::start().await;
    let (owner, superuser) = (db.owner().await, db.superuser().await);
    let pool = db.pool(Profile::Job, 6).await;
    let runner = Runner::start(&db, &pool, "cut", 2, NO_POLL);
    tokio::time::sleep(Duration::from_secs(1)).await;
    let long = queue(&pool, &sleeper(14)).await;
    wait_status(&owner, long, "running", Duration::from_secs(10)).await;
    cut_connections(&superuser).await;
    tokio::time::sleep(Duration::from_secs(3)).await;
    let quick = queue(&pool, &noop_job()).await;
    wait_status(&owner, quick, "completed", Duration::from_secs(15)).await;
    wait_status(&owner, long, "completed", Duration::from_secs(30)).await;
    let row = owner
        .query_one(
            "SELECT attempt, fencing_token, retry_count FROM jobs WHERE id = $1",
            &[&long.0],
        )
        .await
        .unwrap();
    assert_eq!(
        (
            row.get::<_, i32>(0),
            row.get::<_, i64>(1),
            row.get::<_, i32>(2)
        ),
        (1, 1, 0),
        "the cut did not cost the job an attempt"
    );
    runner.stop().await;
}

/// Passes when a runner cut off from the database for longer than its lease loses the job: the
/// lease expires, the job is taken over under a new token, the first holder's late writes are
/// refused, and once the database is back the runner's listener reconnects and it claims again.
#[tokio::test(flavor = "multi_thread")]
async fn a_long_cut_loses_the_job_to_a_takeover_and_the_runner_recovers() {
    let db = TestDatabase::start().await;
    let (owner, superuser) = (db.owner().await, db.superuser().await);
    let pool = db.pool(Profile::Job, 6).await;
    let runner = Runner::start(&db, &pool, "victim", 4, NO_POLL);
    tokio::time::sleep(Duration::from_secs(1)).await;
    let id = queue(&pool, &sleeper(120)).await;
    wait_status(&owner, id, "running", Duration::from_secs(10)).await;
    let old = Lease {
        job_id: id,
        project_id: None,
        fencing_token: 1,
        attempt: 1,
    };

    superuser
        .batch_execute("ALTER ROLE op_app NOLOGIN")
        .await
        .unwrap();
    cut_connections(&superuser).await;
    tokio::time::sleep(Duration::from_secs(9)).await;
    superuser
        .batch_execute("ALTER ROLE op_app LOGIN")
        .await
        .unwrap();

    let admin = db.pool(Profile::Job, 4).await;
    let mut tx = deployment_tx(&admin, "test reaper").await.unwrap();
    let ended = lease::expire_leases(&mut tx, 10).await.unwrap();
    tx.commit().await.unwrap();
    assert_eq!(ended.len(), 1);
    assert_eq!(ended[0].status, JobStatus::Interrupted);
    let mut tx = deployment_tx(&admin, "test resume").await.unwrap();
    assert!(matches!(
        control::resume(&mut tx, id).await.unwrap(),
        Resumed::Queued(_)
    ));
    tx.commit().await.unwrap();

    wait_until(Duration::from_secs(40), "the takeover", || async {
        let row = owner
            .query_one(
                "SELECT status, fencing_token FROM jobs WHERE id = $1",
                &[&id.0],
            )
            .await
            .unwrap();
        (row.get::<_, String>(0), row.get::<_, i64>(1)) == ("running".to_owned(), 2)
    })
    .await;
    let store = PgJobStore::new(admin.clone());
    let late = Conclusion::Finish {
        status: JobStatus::Completed,
        outcome: AttemptOutcome::Completed,
        result: None,
        error_code: None,
        error_message: None,
    };
    assert_eq!(store.conclude(&old, &late).await.unwrap(), None);
    assert!(store.heartbeat(&old, SHORT.ttl).await.unwrap().is_none());
    let outcomes: Vec<Option<String>> = owner
        .query(
            "SELECT outcome FROM job_attempts WHERE job_id = $1 ORDER BY attempt",
            &[&id.0],
        )
        .await
        .unwrap()
        .iter()
        .map(|r| r.get(0))
        .collect();
    assert_eq!(outcomes, vec![Some("lost".to_owned()), None]);

    let after = queue(&admin, &noop_job()).await;
    wait_status(&owner, after, "completed", Duration::from_secs(30)).await;
    runner.task.abort();
}

/// Passes when a cancel request made from another connection while the handler runs reaches the
/// handler at the next heartbeat and the job ends `cancelled`, within a few ticks.
#[tokio::test(flavor = "multi_thread")]
async fn a_cancel_from_another_connection_is_seen_at_the_next_heartbeat() {
    let db = TestDatabase::start().await;
    let owner = db.owner().await;
    let pool = db.pool(Profile::Job, 6).await;
    let runner = Runner::start(&db, &pool, "cancel", 2, NO_POLL);
    tokio::time::sleep(Duration::from_secs(1)).await;
    let id = queue(&pool, &sleeper(120)).await;
    wait_status(&owner, id, "running", Duration::from_secs(10)).await;

    let second = db.pool(Profile::Api, 2).await;
    let asked = Instant::now();
    let mut tx = deployment_tx(&second, "test cancel").await.unwrap();
    let control = control::cancel(&mut tx, id, "test").await.unwrap().unwrap();
    tx.commit().await.unwrap();
    assert_eq!(control.status, JobStatus::Cancelling);
    wait_status(&owner, id, "cancelled", Duration::from_secs(10)).await;
    let seen = asked.elapsed();
    println!("cancel: ended {seen:?} after the request, heartbeat tick 1 s");
    assert!(seen < Duration::from_secs(5), "{seen:?}");
    let outcome: Option<String> = owner
        .query_one(
            "SELECT outcome FROM job_attempts WHERE job_id = $1",
            &[&id.0],
        )
        .await
        .unwrap()
        .get(0);
    assert_eq!(outcome.as_deref(), Some("cancelled"));
    runner.stop().await;
}

/// Passes when a pause request made from another connection reaches the heartbeat's watch within
/// two ticks, and not before it was made.
#[tokio::test(flavor = "multi_thread")]
async fn a_pause_from_another_connection_is_seen_at_the_next_heartbeat() {
    let db = TestDatabase::start().await;
    let owner = db.owner().await;
    let pool = db.pool(Profile::Job, 4).await;
    let project = seed_project(&owner, "pa").await;
    let new = NewJob::for_kind("autolabel", Some(project), json!({}), None, "test").unwrap();
    let id = queue_in(&pool, project, &new).await;
    let store = Arc::new(PgJobStore::new(pool.clone()));
    let claimed = store
        .claim(&claim_request_kind("autolabel", "pauser", 1))
        .await
        .unwrap();
    assert_eq!(claimed[0].lease.job_id, id);
    let watch = LeaseWatch::new(SHORT);
    let beats = tokio::spawn(heartbeat_loop(
        Arc::clone(&store),
        claimed[0].lease,
        watch.clone(),
    ));
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert!(!watch.pause_requested(), "nothing asked for a pause yet");

    let second = db.pool(Profile::Api, 2).await;
    let asked = Instant::now();
    let mut tx = deployment_tx(&second, "test pause").await.unwrap();
    control::pause(&mut tx, id, PauseReason::User)
        .await
        .unwrap()
        .unwrap();
    tx.commit().await.unwrap();
    wait_until(
        Duration::from_secs(5),
        "the pause to reach the heartbeat",
        || async { watch.pause_requested() },
    )
    .await;
    let seen = asked.elapsed();
    println!("pause: seen {seen:?} after the request, heartbeat tick 1 s");
    assert!(seen < Duration::from_millis(2500), "{seen:?}");
    beats.abort();
    let row = owner
        .query_one(
            "SELECT status, pause_requested_at IS NOT NULL FROM jobs WHERE id = $1",
            &[&id.0],
        )
        .await
        .unwrap();
    assert_eq!(
        (row.get::<_, String>(0), row.get::<_, bool>(1)),
        ("running".to_owned(), true)
    );
}

async fn queue_in(pool: &Pool, project: op_db::ProjectId, new: &NewJob) -> JobId {
    let mut tx = op_db::project_tx(pool, project).await.unwrap();
    let exclusive::Inserted::Created(id) = exclusive::insert_queued(&mut tx, new).await.unwrap()
    else {
        panic!("not created");
    };
    tx.commit().await.unwrap();
    id
}
