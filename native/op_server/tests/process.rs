//! The `op-server` binary run as a process: what it refuses at start and what `health` reports.
//!
//! The refusal of a retired variable has one test per variable of the generated contract
//! (`contracts/json/retired_env.json`), so a variable added to the Python table is tested the
//! moment the contract is regenerated. A custom harness (libtest-mimic) lists them as tests.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Command, Output, Stdio};
use std::thread;

use libtest_mimic::{Arguments, Failed, Trial};
use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_op-server");
const CONTRACT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../contracts/json/retired_env.json"
);

fn run(args: &[&str], env: &[(&str, &str)]) -> Output {
    Command::new(BIN)
        .args(args)
        .env_clear()
        .envs(env.iter().copied())
        .output()
        .expect("op-server starts")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn refuses_a_retired_variable(name: &str, was: &str) -> Result<(), Failed> {
    let output = run(&["api"], &[(name, "x")]);
    let text = stderr(&output);
    if output.status.success() {
        return Err(format!("{name}: the api role started").into());
    }
    if !text.contains(name) || !text.contains(was) {
        return Err(format!("{name}: the refusal does not name it and say {was:?}: {text}").into());
    }
    Ok(())
}

fn refuses_without_an_upstream() -> Result<(), Failed> {
    let output = run(&["api"], &[]);
    let text = stderr(&output);
    if output.status.success() || !text.contains("OP_PY_API_URL") {
        return Err(format!("expected a refusal naming OP_PY_API_URL, got {text}").into());
    }
    Ok(())
}

fn refuses_a_bad_value() -> Result<(), Failed> {
    let output = run(
        &["api"],
        &[
            ("OP_PY_API_URL", "http://api-py:4613"),
            ("OP_API_LISTEN", "everywhere"),
        ],
    );
    let text = stderr(&output);
    if output.status.success() || !text.contains("OP_API_LISTEN") {
        return Err(format!("expected a refusal naming OP_API_LISTEN, got {text}").into());
    }
    Ok(())
}

/// A one-answer HTTP server on a free loopback port; returns its address.
fn answers_once(status_line: &'static str) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer);
            let _ = write!(
                stream,
                "HTTP/1.1 {status_line}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            );
        }
    });
    addr
}

fn health(addr: SocketAddr, path: &str) -> Output {
    let listen = format!("0.0.0.0:{}", addr.port());
    run(
        &["health", "--path", path, "--timeout-s", "5"],
        &[("OP_API_LISTEN", &listen)],
    )
}

fn health_succeeds_on_a_success() -> Result<(), Failed> {
    let output = health(answers_once("200 OK"), "/live");
    if output.status.success() {
        Ok(())
    } else {
        Err(stderr(&output).into())
    }
}

fn health_fails_on_a_503_and_says_so() -> Result<(), Failed> {
    let output = health(answers_once("503 Service Unavailable"), "/health");
    let text = stderr(&output);
    if output.status.success() || !text.contains("503") {
        return Err(format!("expected a failure naming 503, got {text}").into());
    }
    Ok(())
}

fn health_fails_when_nothing_listens() -> Result<(), Failed> {
    let closed = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = closed.local_addr().expect("addr");
    drop(closed);
    if health(addr, "/live").status.success() {
        return Err("health succeeded with nothing listening".to_owned().into());
    }
    Ok(())
}

/// Run the `api` role on a free port, ask it for `/live`, stop it with SIGTERM and read what it
/// logged. The wait for the start is a read of its log, which is written when it is ready.
fn serves_logs_with_the_request_id_and_stops_on_sigterm() -> Result<(), Failed> {
    let probe = TcpListener::bind("127.0.0.1:0")?;
    let addr = probe.local_addr()?;
    drop(probe);
    let mut child = Command::new(BIN)
        .arg("api")
        .env_clear()
        .env("OP_PY_API_URL", "http://127.0.0.1:9")
        .env("OP_API_LISTEN", addr.to_string())
        .stderr(Stdio::piped())
        .spawn()?;
    let stderr = child.stderr.take().ok_or("no stderr pipe")?;
    let mut lines = BufReader::new(stderr).lines();
    let started = lines.next().ok_or("the role logged nothing")??;
    if !started.contains("the front is serving") {
        return Err(format!("expected the ready line first, got {started}").into());
    }
    let mut connection = TcpStream::connect(addr)?;
    connection.write_all(b"GET /live HTTP/1.1\r\nhost: t\r\nx-request-id: from-the-client\r\nconnection: close\r\n\r\n")?;
    let mut answer = String::new();
    connection.read_to_string(&mut answer)?;
    let terminated = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()?;
    let status = child.wait()?;
    let rest: Vec<String> = lines.map_while(Result::ok).collect();
    if !terminated.success() || !status.success() {
        return Err(format!("SIGTERM did not end the role cleanly: {status}").into());
    }
    if !answer.starts_with("HTTP/1.1 200") || !answer.contains("x-request-id: from-the-client") {
        return Err(format!("unexpected answer: {answer}").into());
    }
    let log = rest.join("\n");
    if !log.contains("\"request_id\":\"from-the-client\"") || !log.contains("stopped") {
        return Err(format!("the request id or the stop is not in the log: {log}").into());
    }
    Ok(())
}

fn main() {
    let arguments = Arguments::from_args();
    let contract: Value = serde_json::from_str(
        &std::fs::read_to_string(CONTRACT).expect("the contract is committed"),
    )
    .expect("the contract is JSON");
    let mut trials: Vec<Trial> = contract["retired"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|entry| {
            let name = entry["name"].as_str().expect("a name").to_owned();
            let was = entry["was"].as_str().expect("words").to_owned();
            Trial::test(format!("retired_variable_{name}_is_refused"), move || {
                refuses_a_retired_variable(&name, &was)
            })
        })
        .collect();
    trials.push(Trial::test(
        "a_missing_upstream_is_refused_by_name",
        refuses_without_an_upstream,
    ));
    trials.push(Trial::test(
        "a_bad_setting_is_refused_by_name",
        refuses_a_bad_value,
    ));
    trials.push(Trial::test(
        "the_role_serves_logs_the_request_id_and_stops_on_sigterm",
        serves_logs_with_the_request_id_and_stops_on_sigterm,
    ));
    trials.push(Trial::test(
        "health_succeeds_on_a_2xx",
        health_succeeds_on_a_success,
    ));
    trials.push(Trial::test(
        "health_fails_on_a_503",
        health_fails_on_a_503_and_says_so,
    ));
    trials.push(Trial::test(
        "health_fails_with_nothing_listening",
        health_fails_when_nothing_listens,
    ));
    libtest_mimic::run(&arguments, trials).exit();
}
