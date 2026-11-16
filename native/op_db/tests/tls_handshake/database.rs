//! Owed to the sweep (needs Docker): the pool, the `LISTEN` connection and the session lock
//! connect to a real PostgreSQL started with `ssl=on` and a generated certificate, under
//! `verify-full` with `OP_DB_SSLROOTCERT`, and are refused under a wrong authority.
//!
//! Run: `cargo test -p op_db --features db --test tls_handshake database -- --test-threads=1`

use op_db::testing::{ServerTls, TestDatabase};
use op_db::{
    DbSettings, ListenConnector, PgConnector, Pool, PoolSpec, Profile, SessionLock, SslMode,
};
use tempfile::TempDir;

use crate::pki::{Ca, loopback};

const LOCK_CLASS: i32 = 0x7473;

fn verifying(base: &DbSettings, dir: &TempDir, name: &str, ca: &Ca) -> DbSettings {
    let file = dir.path().join(name);
    std::fs::write(&file, &ca.pem).unwrap();
    let mut settings = base.clone();
    settings.ssl_mode = SslMode::VerifyFull;
    settings.ssl_root_cert = Some(file);
    settings
}

/// How many of the server's sessions with this `application_name` use TLS.
async fn tls_sessions(db: &TestDatabase, application_name: &str) -> i64 {
    db.superuser()
        .await
        .query_one(
            "SELECT count(*) FROM pg_stat_activity a JOIN pg_stat_ssl s USING (pid) \
             WHERE a.application_name = $1 AND s.ssl",
            &[&application_name],
        )
        .await
        .unwrap()
        .get(0)
}

#[tokio::test]
async fn every_connection_kind_uses_tls_under_verify_full_and_a_wrong_authority_is_refused() {
    let (right, wrong) = (
        Ca::new("op test authority"),
        Ca::new("some other authority"),
    );
    let (cert, key) = right.leaf(vec![loopback()]);
    let db = TestDatabase::start_with_tls(&ServerTls {
        cert_pem: cert.pem(),
        key_pem: key.serialize_pem(),
    })
    .await;
    let dir = TempDir::new().unwrap();
    let good = verifying(db.settings(), &dir, "right.pem", &right);
    let bad = verifying(db.settings(), &dir, "wrong.pem", &wrong);
    let spec = |name: &str| PoolSpec {
        application_name: name.to_owned(),
        max_size: 2,
        profile: Profile::Api,
    };

    let pool = Pool::connect(&good, spec("op-tls-pool")).await.unwrap();
    assert_eq!(tls_sessions(&db, "op-tls-pool").await, 1);
    Pool::connect(&bad, spec("op-tls-pool-bad"))
        .await
        .unwrap_err();

    let mut listen = PgConnector::new(&good, "op-tls-listen", &["op_tls_test"]);
    let session = listen.connect().await.unwrap();
    assert_eq!(tls_sessions(&db, "op-tls-listen").await, 1);
    PgConnector::new(&bad, "op-tls-listen-bad", &["op_tls_test"])
        .connect()
        .await
        .unwrap_err();

    let lock = SessionLock::try_acquire(&good, "op-tls-lock", LOCK_CLASS)
        .await
        .unwrap()
        .expect("nobody else holds the lock");
    assert_eq!(tls_sessions(&db, "op-tls-lock").await, 1);
    SessionLock::try_acquire(&bad, "op-tls-lock-bad", LOCK_CLASS)
        .await
        .unwrap_err();

    drop((pool, session, lock));
}
