/// Rust preprocessing service for Triton inference.
///
/// High-performance alternative to Python yolo-api with:
/// - Single process instead of 32 workers
/// - True parallelism via tokio (no GIL)
/// - Zero-copy image processing
/// - SIMD-optimized resize

mod config;
mod error;
mod handlers;
mod hash;
mod opensearch;
mod postprocess;
mod preprocess;
mod triton;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use config::Config;
use handlers::detect::AppState as DetectState;
use handlers::embed::AppState as EmbedState;
use handlers::ingest::AppState as IngestState;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing::info;

fn main() -> Result<(), error::AppError> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("rust_preprocess=info,tower_http=debug")
        .init();

    eprintln!("[STARTUP] Rust preprocessing service starting...");

    // Configure tokio runtime with more threads to match Python's 32 worker processes
    // Use all available cores for worker threads
    // Allow up to 128 blocking threads for CPU-bound preprocessing (vs Python's 32 processes)
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())  // Use all CPU cores
        .max_blocking_threads(128)         // Allow 128 concurrent blocking tasks (vs Python's 32 processes)
        .enable_all()
        .build()
        .map_err(|e| error::AppError::Internal(format!("Failed to build runtime: {e}")))?;

    eprintln!("[STARTUP] Tokio runtime configured: {} worker threads, 128 max blocking threads", num_cpus::get());

    runtime.block_on(async_main())
}

async fn async_main() -> Result<(), error::AppError> {
    eprintln!("[STARTUP] Starting async main...");

    let config = Config::from_env();
    eprintln!("[STARTUP] Config loaded");
    info!("Starting rust-preprocess service");
    info!("Triton URL: {}", config.triton_url);
    info!("OpenSearch URL: {}", config.opensearch_url);
    info!("Port: {}", config.port);
    info!("gRPC pool size: {}", config.grpc_pool_size);

    // Initialize clients
    eprintln!("[STARTUP] Connecting to Triton at {}...", config.triton_url);
    let triton = match triton::TritonClient::new(&config.triton_url, config.grpc_pool_size).await {
        Ok(client) => {
            eprintln!("[STARTUP] Triton client connected successfully");
            client
        }
        Err(e) => {
            eprintln!("[STARTUP ERROR] Failed to connect to Triton: {}", e);
            return Err(e);
        }
    };

    eprintln!("[STARTUP] Creating OpenSearch client...");
    let opensearch = opensearch::OpenSearchClient::new(config.opensearch_url.clone());

    // Create shared state
    // Note: We're using separate AppState types per handler module, but they all contain the same clients.
    // In production, you'd use a single unified AppState, but this matches the modular structure.
    let detect_state = Arc::new(DetectState {
        triton: triton.clone(),
    });

    let embed_state = Arc::new(EmbedState {
        triton: triton.clone(),
    });

    let ingest_state = Arc::new(IngestState {
        triton: triton.clone(),
        opensearch: opensearch.clone(),
    });

    // Build router
    let app = Router::new()
        .route("/health", get(handlers::health::health))
        .route(
            "/detect",
            post({
                let state = detect_state.clone();
                move |multipart| handlers::detect::detect_single(axum::extract::State(state), multipart)
            }),
        )
        .route(
            "/detect/batch",
            post({
                let state = detect_state.clone();
                move |multipart| handlers::detect::detect_batch(axum::extract::State(state), multipart)
            }),
        )
        .route(
            "/embed/image",
            post({
                let state = embed_state.clone();
                move |multipart| handlers::embed::embed_single(axum::extract::State(state), multipart)
            }),
        )
        .route(
            "/embed/batch",
            post({
                let state = embed_state.clone();
                move |multipart| handlers::embed::embed_batch(axum::extract::State(state), multipart)
            }),
        )
        .route(
            "/ingest/batch",
            post({
                let state = ingest_state.clone();
                move |multipart| handlers::ingest::ingest_batch(axum::extract::State(state), multipart)
            }),
        )
        .layer(DefaultBodyLimit::max(1024 * 1024 * 1024)) // 1GB for batch uploads (64 images × ~10MB worst case)
        .layer(TraceLayer::new_for_http());

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    eprintln!("[STARTUP] Binding to {}...", addr);
    info!("Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("[STARTUP] Server ready! Listening on http://{}", addr);

    axum::serve(listener, app).await?;

    eprintln!("[SHUTDOWN] Server stopped");
    Ok(())
}
