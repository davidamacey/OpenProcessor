//! Serving the front: the channel settings of plan 17 section 2.4, the oversize mapping and the
//! drain on shutdown.

use std::future::Future;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::codegen::http;
use tonic::transport::Server;
use tower::util::MapResponseLayer;

use super::{Front, FrontHandle};
use crate::pipeline::Pipeline;
use crate::proto::grpc_inference_service_server::GrpcInferenceServiceServer;

const MIB: u32 = 1024 * 1024;

/// Maps the one status tonic answers for an oversize message (`OUT_OF_RANGE`) to the one gRPC
/// C-core answers (`RESOURCE_EXHAUSTED`), which the API retries (plan 17 section 1.1). The
/// status of a request refused before the handler runs is in the response headers (a
/// trailers-only response, gRPC over HTTP/2 protocol document).
fn oversize_as_resource_exhausted<B>(mut response: http::Response<B>) -> http::Response<B> {
    const OUT_OF_RANGE: &str = "11";
    const RESOURCE_EXHAUSTED: &str = "8";
    let headers = response.headers_mut();
    if headers
        .get("grpc-status")
        .is_some_and(|v| v == OUT_OF_RANGE)
    {
        headers.insert(
            "grpc-status",
            http::HeaderValue::from_static(RESOURCE_EXHAUSTED),
        );
    }
    response
}

/// What [`serve`] needs besides the service.
#[derive(Debug, Clone, Copy)]
pub struct ServeConfig {
    /// `OP_FRAME_ENGINE_MAX_MESSAGE_MIB`, both directions.
    pub max_message_mib: u32,
    /// `OP_FRAME_ENGINE_DRAIN_SECONDS`.
    pub drain: Duration,
}

/// Serves `front` on `listener` until `shutdown` completes, then drains: readiness turns false,
/// photos in flight get `drain` to finish, then the connections are closed gracefully (tonic's
/// `serve_with_incoming_shutdown`: GOAWAY, then the open streams finish) and the call returns.
///
/// # Errors
///
/// The gRPC server fails while serving; a late drain is not an error.
pub async fn serve<P: Pipeline>(
    listener: TcpListener,
    front: Front<P>,
    handle: FrontHandle<P>,
    config: ServeConfig,
    shutdown: impl Future<Output = ()>,
) -> Result<(), tonic::transport::Error> {
    let limit = usize::try_from(config.max_message_mib.saturating_mul(MIB)).unwrap_or(usize::MAX);
    let service = GrpcInferenceServiceServer::new(front)
        .max_decoding_message_size(limit)
        .max_encoding_message_size(limit);
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = Server::builder()
        .initial_stream_window_size(MIB)
        .initial_connection_window_size(2 * MIB)
        .layer(MapResponseLayer::new(oversize_as_resource_exhausted))
        .add_service(service)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
            let _ = stopped.await;
        });
    tokio::pin!(server);
    tokio::select! {
        ended = &mut server => return ended,
        () = shutdown => {}
    }
    let until = tokio::time::Instant::now() + config.drain;
    handle.begin_drain();
    tracing::info!(
        "stopping: draining {} requests in flight",
        handle.in_flight()
    );
    let _ = tokio::time::timeout_at(until, handle.wait_idle()).await;
    let _ = stop.send(());
    match tokio::time::timeout_at(until, &mut server).await {
        Ok(ended) => ended,
        Err(_late) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_oversize_status_is_remapped_and_nothing_else_in_the_response_changes() {
        let with = |status: &str| {
            http::Response::builder()
                .header("grpc-status", status)
                .header("grpc-message", "too%20large")
                .body(())
                .expect("response")
        };
        let mapped = oversize_as_resource_exhausted(with("11"));
        assert_eq!(mapped.headers()["grpc-status"], "8");
        assert_eq!(mapped.headers()["grpc-message"], "too%20large");
        for same in ["0", "3", "5", "8", "13", "14"] {
            assert_eq!(
                oversize_as_resource_exhausted(with(same)).headers()["grpc-status"],
                same
            );
        }
        let none = http::Response::builder().body(()).expect("response");
        assert!(
            oversize_as_resource_exhausted(none)
                .headers()
                .get("grpc-status")
                .is_none()
        );
    }
}
