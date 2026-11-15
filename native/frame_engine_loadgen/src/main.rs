//! `op-fe-loadgen`: see the library's documentation.

use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Measures a frame engine (protocol version 1) and the models alone"
)]
struct Args {}

fn main() {
    let _ = Args::parse();
}
