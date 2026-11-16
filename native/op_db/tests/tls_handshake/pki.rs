//! Certificates generated for a test: an authority and the server certificates it signs.

use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;

use rcgen::{
    BasicConstraints, Certificate, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa,
    Issuer, KeyPair, KeyUsagePurpose, SanType,
};
use rustls::ServerConfig;
use rustls::crypto::ring;
use rustls::pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};

/// A certificate authority that signs server certificates.
pub struct Ca {
    pub pem: String,
    issuer: Issuer<'static, KeyPair>,
}

/// A server certificate with its key, as the server's TLS configuration.
pub struct ServerCert(pub Arc<ServerConfig>);

impl Ca {
    pub fn new(name: &str) -> Self {
        let key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        params.distinguished_name.push(DnType::CommonName, name);
        let pem = params.self_signed(&key).unwrap().pem();
        Self {
            pem,
            issuer: Issuer::new(params, key),
        }
    }

    /// A server certificate valid for the given alternative names only, with its key.
    pub fn leaf(&self, names: Vec<SanType>) -> (Certificate, KeyPair) {
        let key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
        params.subject_alt_names = names;
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
        let cert = params.signed_by(&key, &self.issuer).unwrap();
        (cert, key)
    }

    /// The same, as a TLS server's configuration.
    pub fn server_cert(&self, names: Vec<SanType>) -> ServerCert {
        let (cert, key) = self.leaf(names);
        let config = ServerConfig::builder_with_provider(Arc::new(ring::default_provider()))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(
                vec![cert.der().clone()],
                PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.serialize_der())),
            )
            .unwrap();
        ServerCert(Arc::new(config))
    }

    /// A server certificate valid for the loopback address the tests connect to.
    pub fn loopback_cert(&self) -> ServerCert {
        self.server_cert(vec![loopback()])
    }
}

/// The alternative name of the loopback address the tests connect to.
pub fn loopback() -> SanType {
    SanType::IpAddress(IpAddr::V4(Ipv4Addr::LOCALHOST))
}
