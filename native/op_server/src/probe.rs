//! `op-server health`: the container's health check.

use std::error::Error;
use std::net::SocketAddr;
use std::time::Duration;

use op_api::health::probe;
use op_api::settings::{ApiSettings, LISTEN};
use op_config::Env;

/// Ask the front this container runs for `path`.
///
/// It reads the address the front binds ([`LISTEN`]) and asks on the loopback address of the same
/// port. It needs no other setting, so a container whose upstream setting is wrong is still
/// reported by the front's own refusal to start, not by a second error here.
pub fn run(env: &Env, path: &str, timeout_s: u64) -> Result<(), Box<dyn Error>> {
    let listen: SocketAddr = env.get(LISTEN, ApiSettings::DEFAULT_LISTEN)?;
    let target = SocketAddr::from(([127, 0, 0, 1], listen.port()));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(probe(target, path, Duration::from_secs(timeout_s)))?;
    Ok(())
}
