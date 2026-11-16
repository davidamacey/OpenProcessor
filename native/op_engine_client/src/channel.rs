//! One HTTP/2 connection to a gRPC server, with the settings of `ENGINE_CHANNEL_OPTIONS`
//! (`src/clients/frame_engine.py`) in tonic's terms.
//!
//! * Keep-alive: a ping every 30 s, also while no call is in flight, 10 s to answer
//!   (`Endpoint::http2_keep_alive_interval`, `keep_alive_timeout`, `keep_alive_while_idle`,
//!   <https://docs.rs/tonic/0.14.9/tonic/transport/channel/struct.Endpoint.html>). A server
//!   disconnects a client that pings more often than its policy allows ("too many pings",
//!   gRPC's keepalive guide, <https://grpc.io/docs/guides/keepalive/>); the engine's front
//!   answers every ping and Triton runs with `--grpc-http2-min-recv-ping-interval-without-data=20000`
//!   and idle pings permitted (`docker-compose.yml`), both under 30 s, so this client stays inside
//!   them; `tests/keepalive_policy.rs` pins the two to the contract's `ping_policy` and the compose
//!   file.
//! * Windows: 1 MiB per stream, 2 MiB per connection, no adaptive window
//!   (`initial_stream_window_size`, `initial_connection_window_size`, `http2_adaptive_window`),
//!   as the engine's front sizes its own.
//! * Messages: 512 MiB both ways (`Grpc::max_decoding_message_size`, `max_encoding_message_size`).
//! * A channel made with `connect_lazy` dials on first use and again after the connection is
//!   lost: the next call finds a restarted engine again, as the Python channels' bounded
//!   reconnect backoff does. tonic reports a failed dial as `UNAVAILABLE` (`ConnectError`).
//!   Each channel is its own TCP connection; the pool spreads calls over them round robin.

use tonic::transport::{Channel, Endpoint};

use crate::error::ConfigError;
use crate::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::settings::{CONNECT_TIMEOUT, KEEPALIVE_INTERVAL, KEEPALIVE_TIMEOUT};

/// A gRPC client of one connection.
pub(crate) type Client = GrpcInferenceServiceClient<Channel>;

const MIB: usize = 1 << 20;
const MAX_MESSAGE_BYTES: usize = 512 * MIB;

/// `host:port` as a URI; an address that has a scheme is kept.
fn uri(url: &str) -> String {
    if url.contains("://") {
        url.to_owned()
    } else {
        format!("http://{url}")
    }
}

fn endpoint(url: &str) -> Result<Endpoint, ConfigError> {
    let endpoint = Endpoint::from_shared(uri(url)).map_err(|e| ConfigError::Url {
        url: url.to_owned(),
        reason: e.to_string(),
    })?;
    Ok(endpoint
        .tcp_nodelay(true)
        .http2_keep_alive_interval(KEEPALIVE_INTERVAL)
        .keep_alive_timeout(KEEPALIVE_TIMEOUT)
        .keep_alive_while_idle(true)
        .connect_timeout(CONNECT_TIMEOUT))
}

fn wrap(channel: Channel) -> Client {
    Client::new(channel)
        .max_decoding_message_size(MAX_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_MESSAGE_BYTES)
}

/// A channel for photos: the windows are sized for a 10 MB body.
///
/// Must be called inside a tokio runtime (the channel's buffer worker is spawned on it).
pub(crate) fn engine_channel(url: &str) -> Result<Client, ConfigError> {
    let endpoint = endpoint(url)?
        .initial_stream_window_size(u32::try_from(MIB).ok())
        .initial_connection_window_size(u32::try_from(2 * MIB).ok())
        .http2_adaptive_window(false);
    Ok(wrap(endpoint.connect_lazy()))
}

/// A channel for the model server's small calls (readiness, load).
///
/// Must be called inside a tokio runtime.
pub(crate) fn model_server_channel(url: &str) -> Result<Client, ConfigError> {
    Ok(wrap(endpoint(url)?.connect_lazy()))
}
