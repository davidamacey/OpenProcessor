//! The `op-server` binary and its roles. `api` runs the front on the API's port; `health` asks a
//! running front whether it is up (the container's health check). The `worker`, `periodic` and
//! `migrate` roles are added by the slices that build them.
#![forbid(unsafe_code)]

mod api;
mod probe;

use std::process::ExitCode;

use clap::{Parser, Subcommand};
use op_config::Env;

#[derive(Debug, Parser)]
#[command(version, about = "The OpenProcessor server")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the front: answer the routes Rust owns and forward the rest to the Python API.
    Api,
    /// Exit 0 when the front on this machine answers the path with a success.
    Health {
        /// The path to ask for. `/live` asks only the front; `/health` asks through it.
        #[arg(long, default_value = "/live")]
        path: String,
        /// Seconds to wait for the answer.
        #[arg(long, default_value_t = 5)]
        timeout_s: u64,
    },
}

fn main() -> ExitCode {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .json()
        .flatten_event(true)
        .with_current_span(true)
        .with_span_list(false)
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .init();
    let env = Env::from_process();
    let outcome = match args.command {
        Command::Api => api::run(&env),
        Command::Health { path, timeout_s } => probe::run(&env, &path, timeout_s),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(refusal) => {
            tracing::error!("{refusal}");
            ExitCode::FAILURE
        }
    }
}
