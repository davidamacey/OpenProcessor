//! The `op-server` binary and its roles: `api`, `worker`, `periodic` and `migrate`. Until a role is built it refuses to start.
#![forbid(unsafe_code)]

use std::process::ExitCode;

fn main() -> ExitCode {
    eprintln!("op-server: no role built yet");
    ExitCode::FAILURE
}
