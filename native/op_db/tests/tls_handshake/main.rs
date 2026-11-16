//! TLS between `op_db` and a PostgreSQL server, against an in-process server that speaks the
//! `SSLRequest` preamble and then TLS with a certificate generated here. No database, no network
//! beyond the loopback socket. What each `OP_DB_SSLMODE` must do is libpq's table
//! (<https://www.postgresql.org/docs/18/libpq-ssl.html>), made stricter where the driver cannot
//! match it; the mode table is in the module comment of `native/op_db/src/tls.rs`.
#![expect(
    clippy::unwrap_used,
    reason = "the certificate and connection helpers are test code outside #[test] functions"
)]

#[cfg(feature = "db")]
mod database;
mod pki;
mod server;

use std::error::Error;
use std::path::PathBuf;

use op_config::Env;
use op_db::DbSettings;
use pki::Ca;
use rcgen::SanType;
use server::{Answer, FakeServer, Seen};
use tempfile::TempDir;

struct Client {
    dir: TempDir,
}

impl Client {
    fn new() -> Self {
        Self {
            dir: TempDir::new().unwrap(),
        }
    }

    fn write(&self, name: &str, content: &str) -> PathBuf {
        let path = self.dir.path().join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    fn settings(&self, port: u16, mode: &str, roots: Option<&Ca>) -> DbSettings {
        let password = self.write("password", "pw");
        let mut pairs = vec![
            ("OP_DB_HOST", "127.0.0.1".to_owned()),
            ("OP_DB_PORT", port.to_string()),
            ("OP_DB_SSLMODE", mode.to_owned()),
            ("OP_DB_APP_PASSWORD_FILE", password.display().to_string()),
        ];
        if let Some(ca) = roots {
            let file = self.write("roots.pem", &ca.pem);
            pairs.push(("OP_DB_SSLROOTCERT", file.display().to_string()));
        }
        DbSettings::from_env(&Env::from_pairs(pairs)).unwrap()
    }
}

/// Connect the way the pool, the listener and the lock do; the error text is the whole chain.
async fn connect(settings: &DbSettings) -> Result<(), String> {
    let config = settings.pg_config("op-tls-test", true, false).unwrap();
    let tls = settings.tls_connector().unwrap();
    match config.connect(tls).await {
        Ok((client, connection)) => {
            drop((client, connection));
            Ok(())
        }
        Err(e) => Err(chain(&e)),
    }
}

fn chain(error: &dyn Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(e) = source {
        text.push_str(": ");
        text.push_str(&e.to_string());
        source = e.source();
    }
    text
}

fn dns(name: &str) -> Vec<SanType> {
    vec![SanType::DnsName(name.try_into().unwrap())]
}

async fn connected(
    mode: &str,
    roots: Option<&Ca>,
    server_ca: &Ca,
    names: Option<Vec<SanType>>,
) -> (Result<(), String>, Seen) {
    let cert = match names {
        Some(names) => server_ca.server_cert(names),
        None => server_ca.loopback_cert(),
    };
    let mut server = FakeServer::start(Answer::Tls(cert)).await;
    let client = Client::new();
    let result = connect(&client.settings(server.port, mode, roots)).await;
    (result, server.seen().await)
}

#[tokio::test]
async fn verify_full_accepts_the_right_name_and_the_trusted_chain() {
    let ca = Ca::new("trusted");
    let (result, seen) = connected("verify-full", Some(&ca), &ca, None).await;
    result.unwrap();
    assert!(seen.ssl_request && seen.tls && seen.startup, "{seen:?}");
}

#[tokio::test]
async fn verify_full_refuses_a_certificate_for_another_name() {
    let ca = Ca::new("trusted");
    let (result, seen) =
        connected("verify-full", Some(&ca), &ca, Some(dns("db.example.test"))).await;
    let error = result.unwrap_err();
    assert!(error.contains("not valid for name"), "{error}");
    assert!(!seen.startup, "{seen:?}");
}

#[tokio::test]
async fn verify_full_refuses_a_chain_from_an_untrusted_authority() {
    let (trusted, other) = (Ca::new("trusted"), Ca::new("other"));
    let (result, seen) = connected("verify-full", Some(&trusted), &other, None).await;
    let error = result.unwrap_err();
    assert!(error.contains("UnknownIssuer"), "{error}");
    assert!(!seen.startup, "{seen:?}");
}

#[tokio::test]
async fn the_roots_file_is_the_only_trust_root() {
    // No roots file: the platform's roots decide, and they do not include a test authority.
    let ca = Ca::new("private");
    let (result, seen) = connected("verify-full", None, &ca, None).await;
    let error = result.unwrap_err();
    assert!(error.contains("UnknownIssuer"), "{error}");
    assert!(!seen.startup, "{seen:?}");
}

#[tokio::test]
async fn verify_ca_checks_the_chain_and_not_the_name() {
    let (trusted, other) = (Ca::new("trusted"), Ca::new("other"));
    let (result, seen) = connected(
        "verify-ca",
        Some(&trusted),
        &trusted,
        Some(dns("db.example.test")),
    )
    .await;
    result.unwrap();
    assert!(seen.tls && seen.startup, "{seen:?}");

    let (result, seen) = connected(
        "verify-ca",
        Some(&trusted),
        &other,
        Some(dns("db.example.test")),
    )
    .await;
    let error = result.unwrap_err();
    assert!(error.contains("UnknownIssuer"), "{error}");
    assert!(!seen.startup, "{seen:?}");
}

#[tokio::test]
async fn require_encrypts_and_checks_the_chain_only_when_there_is_a_roots_file() {
    let (trusted, other) = (Ca::new("trusted"), Ca::new("other"));
    // libpq: no root file, no verification; the connection is still encrypted.
    let (result, seen) = connected("require", None, &other, Some(dns("db.example.test"))).await;
    result.unwrap();
    assert!(seen.tls && seen.startup, "{seen:?}");
    // with a roots file: verify-ca
    let (result, seen) = connected("require", Some(&trusted), &other, None).await;
    assert!(result.unwrap_err().contains("UnknownIssuer"));
    assert!(!seen.startup, "{seen:?}");
    let (result, _) = connected(
        "require",
        Some(&trusted),
        &trusted,
        Some(dns("db.example.test")),
    )
    .await;
    result.unwrap();
}

#[tokio::test]
async fn the_modes_that_need_tls_fail_when_the_server_answers_no() {
    for mode in ["require", "verify-ca", "verify-full"] {
        let mut server = FakeServer::start(Answer::No).await;
        let client = Client::new();
        let ca = Ca::new("trusted");
        let error = connect(&client.settings(server.port, mode, Some(&ca)))
            .await
            .unwrap_err();
        let seen = server.seen().await;
        assert!(error.contains("does not support TLS"), "{mode}: {error}");
        assert!(seen.ssl_request && !seen.startup, "{mode}: {seen:?}");
    }
}

#[tokio::test]
async fn prefer_and_allow_use_tls_when_offered_without_checking_the_certificate() {
    let other = Ca::new("anyone");
    for mode in ["prefer", "allow"] {
        let (result, seen) = connected(mode, None, &other, Some(dns("db.example.test"))).await;
        result.unwrap();
        assert!(seen.tls && seen.startup, "{mode}: {seen:?}");
    }
}

#[tokio::test]
async fn prefer_and_allow_fall_back_to_plain_text_when_the_server_answers_no() {
    for mode in ["prefer", "allow"] {
        let mut server = FakeServer::start(Answer::No).await;
        let client = Client::new();
        connect(&client.settings(server.port, mode, None))
            .await
            .unwrap();
        let seen = server.seen().await;
        assert!(
            seen.ssl_request && !seen.tls && seen.startup,
            "{mode}: {seen:?}"
        );
    }
}

#[tokio::test]
async fn prefer_does_not_retry_in_plain_text_after_a_failed_handshake() {
    let mut server = FakeServer::start(Answer::TlsThenClose).await;
    let client = Client::new();
    connect(&client.settings(server.port, "prefer", None))
        .await
        .unwrap_err();
    let seen = server.seen().await;
    assert!(seen.ssl_request && !seen.startup, "{seen:?}");
}

#[tokio::test]
async fn disable_never_sends_an_ssl_request() {
    for answer in [Answer::No, Answer::Tls(Ca::new("any").loopback_cert())] {
        let mut server = FakeServer::start(answer).await;
        let client = Client::new();
        connect(&client.settings(server.port, "disable", None))
            .await
            .unwrap();
        let seen = server.seen().await;
        assert!(!seen.ssl_request && !seen.tls && seen.startup, "{seen:?}");
    }
}
