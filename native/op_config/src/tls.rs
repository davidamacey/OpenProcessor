//! The process-wide TLS crypto provider.
//!
//! The Rust server has one TLS stack: rustls with the ring provider (no system OpenSSL, no
//! aws-lc, whose OpenSSL licence `native/deny.toml` does not allow). Every crate that builds a TLS
//! client, the object store in `op_storage` and the database connections in `op_db`, calls
//! [`install_tls_provider`] before it does.

use std::sync::Arc;

use rustls::crypto::{CryptoProvider, ring};

/// Install the ring crypto provider as rustls's process default, once, and return the provider in
/// force. `install_default` answers `Err` when a provider is already installed (by an earlier call,
/// another crate or a test); that is the wanted end state, so the result is ignored, and the
/// provider returned is the one that is installed.
/// <https://docs.rs/rustls/0.23.46/rustls/crypto/struct.CryptoProvider.html#method.install_default>
pub fn install_tls_provider() -> Arc<CryptoProvider> {
    let ours = Arc::new(ring::default_provider());
    let _ = CryptoProvider::install_default((*ours).clone());
    CryptoProvider::get_default().cloned().unwrap_or(ours)
}
