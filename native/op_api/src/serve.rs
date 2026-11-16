//! The accept loop: one task per client connection, bounded, with the idle timeout and the
//! graceful stop of the uvicorn it replaces.
//!
//! `axum::serve` "is intentionally simple and doesn't support any configuration"
//! (<https://docs.rs/axum/0.8.9/axum/serve/fn.serve.html>) and has neither the 75 s keep-alive
//! nor a bound on the drain after the stop signal, so the loop is the one hyper-util documents
//! (<https://docs.rs/hyper-util/0.1.21/hyper_util/server/graceful/index.html>): hyper's HTTP/1
//! connection, a `GracefulShutdown` watching every connection, and a timeout around the drain.
//!
//! - Keep-alive: hyper's `header_read_timeout` runs whenever the connection waits for the next
//!   request head, including after a response, so one value bounds both an idle connection and a
//!   client that sends its headers slowly
//!   (<https://docs.rs/hyper/1.12.0/hyper/server/conn/http1/struct.Builder.html#method.header_read_timeout>).
//! - Connections: a semaphore of `max_connections` permits; with none left the loop stops
//!   accepting, and the kernel's backlog holds the clients until a connection closes.
//! - Every connection gets `TCP_NODELAY`: answers are written whole, and Nagle's algorithm would
//!   hold the tail of a small one.
//! - Upgrades are not enabled: `Connection: Upgrade` is never honoured by this server.

use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::ConnectInfo;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower::ServiceExt as _;

/// Pause after a failed `accept` (out of file descriptors, for one), as `axum::serve` does, so a
/// persistent failure is not a busy loop.
const ACCEPT_ERROR_PAUSE: Duration = Duration::from_millis(100);

/// The bounds of the accept loop.
#[derive(Debug, Clone, Copy)]
pub struct ServeConfig {
    /// Idle connection and request-head deadline.
    pub keep_alive: Duration,
    /// Time open connections get to finish after the stop signal.
    pub shutdown_grace: Duration,
    /// Open connections at most.
    pub max_connections: usize,
}

/// How the server stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stopped {
    /// Every connection finished within the grace.
    Drained,
    /// The grace ran out; this many connections were still open and are cut.
    GraceExpired {
        /// Connections still open.
        open: usize,
    },
}

/// Serve `app` on `listener` until `shutdown` completes, then stop accepting and give open
/// connections `config.shutdown_grace` to finish.
pub async fn serve(
    listener: TcpListener,
    app: Router,
    config: ServeConfig,
    shutdown: impl Future<Output = ()>,
) -> Stopped {
    let permits = Arc::new(Semaphore::new(config.max_connections));
    let graceful = GracefulShutdown::new();
    let mut shutdown = pin!(shutdown);
    while let Some((stream, peer, permit)) =
        next_connection(&listener, &permits, &mut shutdown).await
    {
        spawn_connection(
            &graceful,
            stream,
            peer,
            app.clone(),
            config.keep_alive,
            permit,
        );
    }
    drop(listener);
    let open = graceful.count();
    match tokio::time::timeout(config.shutdown_grace, graceful.shutdown()).await {
        Ok(()) => Stopped::Drained,
        Err(_) => Stopped::GraceExpired { open },
    }
}

async fn next_connection(
    listener: &TcpListener,
    permits: &Arc<Semaphore>,
    shutdown: &mut std::pin::Pin<&mut impl Future<Output = ()>>,
) -> Option<(TcpStream, SocketAddr, OwnedSemaphorePermit)> {
    loop {
        let permit = tokio::select! {
            permit = Arc::clone(permits).acquire_owned() => permit.ok()?,
            () = shutdown.as_mut() => return None,
        };
        let accepted = tokio::select! {
            accepted = listener.accept() => accepted,
            () = shutdown.as_mut() => return None,
        };
        match accepted {
            Ok((stream, peer)) => return Some((stream, peer, permit)),
            Err(error) => {
                tracing::warn!(%error, "accept failed");
                drop(permit);
                tokio::time::sleep(ACCEPT_ERROR_PAUSE).await;
            }
        }
    }
}

fn spawn_connection(
    graceful: &GracefulShutdown,
    stream: TcpStream,
    peer: SocketAddr,
    app: Router,
    keep_alive: Duration,
    permit: OwnedSemaphorePermit,
) {
    if let Err(error) = stream.set_nodelay(true) {
        tracing::debug!(%error, %peer, "TCP_NODELAY not set");
    }
    let service = TowerToHyperService::new(app.map_request(
        move |mut request: http::Request<Incoming>| {
            request.extensions_mut().insert(ConnectInfo(peer));
            request.map(Body::new)
        },
    ));
    let connection = http1::Builder::new()
        .timer(TokioTimer::new())
        .header_read_timeout(keep_alive)
        .keep_alive(true)
        .serve_connection(TokioIo::new(stream), service);
    let watched = graceful.watch(connection);
    tokio::spawn(async move {
        if let Err(error) = watched.await {
            tracing::debug!(%error, %peer, "connection ended with an error");
        }
        drop(permit);
    });
}

/// Install the handlers for SIGTERM (what `docker stop` sends) and SIGINT and return a future
/// that completes when either arrives. The handlers are installed before it returns, so a signal
/// that comes while the server is still starting is not lost.
///
/// # Errors
///
/// The error of installing a signal handler.
pub fn stop_signal() -> io::Result<impl Future<Output = ()>> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut terminate = signal(SignalKind::terminate())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    Ok(async move {
        tokio::select! {
            _ = terminate.recv() => {}
            _ = interrupt.recv() => {}
        }
    })
}
