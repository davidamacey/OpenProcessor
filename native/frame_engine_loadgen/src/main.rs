//! `op-fe-loadgen`: see the library's documentation.

use std::process::ExitCode;

use clap::Parser;
use frame_engine_loadgen::commands::{Command, run};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Measures a frame engine (protocol version 1) and the models alone"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let cli = Cli::parse();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("op-fe-loadgen: cannot start the runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(run(cli.command, argv)) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("op-fe-loadgen: {message}");
            ExitCode::FAILURE
        }
    }
}
