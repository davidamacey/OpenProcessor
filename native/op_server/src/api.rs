//! The `api` role: the front on the API's port.

use std::error::Error;

use op_api::serve::{ServeConfig, Stopped, serve, stop_signal};
use op_api::settings::ApiSettings;
use op_config::Env;
use op_config::retired::reject_retired_env;
use tokio::net::TcpListener;

/// Refuse retired settings, read the front's settings, bind and serve until SIGTERM or SIGINT.
pub fn run(env: &Env) -> Result<(), Box<dyn Error>> {
    reject_retired_env(env)?;
    let settings = ApiSettings::from_env(env)?;
    let router = op_api::build(&settings)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let stop = stop_signal()?;
        let listener = TcpListener::bind(settings.listen).await?;
        tracing::info!(
            listen = %listener.local_addr()?,
            upstream = %settings.upstream.authority,
            "the front is serving"
        );
        let config = ServeConfig {
            keep_alive: settings.keep_alive,
            shutdown_grace: settings.shutdown_grace,
            max_connections: settings.max_connections,
        };
        match serve(listener, router, config, stop).await {
            Stopped::Drained => tracing::info!("stopped, every connection finished"),
            Stopped::GraceExpired { open } => {
                tracing::warn!(open, "stopped, connections still open were cut");
            }
        }
        Ok(())
    })
}
