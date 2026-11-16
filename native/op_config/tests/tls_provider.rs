//! The process-wide rustls crypto provider: ring, installed once, whoever asks first.

use rustls::crypto::{CryptoProvider, ring};

#[test]
fn the_provider_is_ring_and_asking_twice_changes_nothing() {
    let first = op_config::install_tls_provider();
    let second = op_config::install_tls_provider();
    let ring = ring::default_provider();
    assert_eq!(first.cipher_suites, ring.cipher_suites);
    assert_eq!(second.cipher_suites, ring.cipher_suites);
    let installed = CryptoProvider::get_default().expect("the provider is the process default");
    assert_eq!(installed.cipher_suites, ring.cipher_suites);
}
