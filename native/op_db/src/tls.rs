//! The TLS connector every database connection of this crate uses: the pool, the `LISTEN`
//! connection, the session lock and the test harness.
//!
//! The connector is `tokio-postgres-rustls` 0.14.0 (<https://docs.rs/tokio-postgres-rustls/0.14.0/>),
//! the maintained rustls adapter for tokio-postgres's `MakeTlsConnect`; it also supplies the
//! `tls-server-end-point` channel binding that SCRAM-SHA-256-PLUS needs. tokio-postgres decides
//! whether a TLS handshake happens at all (`SslMode`, <https://docs.rs/tokio-postgres/0.7.18/tokio_postgres/config/enum.SslMode.html>:
//! it sends the `SSLRequest`, and `Require` fails when the server answers `N` while `Prefer`
//! carries on in plain text); this module decides what the handshake verifies.
//!
//! What is verified follows libpq's `sslmode` table
//! (<https://www.postgresql.org/docs/18/libpq-ssl.html>, "Protection Provided in Different Modes"):
//!
//! | mode | certificate chain | host name |
//! |---|---|---|
//! | `prefer` | not verified | not verified |
//! | `require` | verified when `OP_DB_SSLROOTCERT` is set (libpq's behaviour when a root file exists), else not | not verified |
//! | `verify-ca` | verified | not verified |
//! | `verify-full` | verified | verified |
//!
//! The trust roots of a verifying mode are the platform's (the verifier `rustls-platform-verifier`
//! gives `reqwest` for the object store) or, when `OP_DB_SSLROOTCERT` names a file, exactly the
//! certificates in that file.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use rustls::client::WebPkiServerVerifier;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, verify_tls12_signature, verify_tls13_signature};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{
    CertificateError, ClientConfig, DigitallySignedStruct, Error as TlsError, RootCertStore,
    SignatureScheme,
};
use rustls_platform_verifier::Verifier as PlatformVerifier;
use tokio_postgres_rustls::MakeRustlsConnect;

use crate::settings::{SettingsError, SslMode};

/// The variable naming the trust roots file.
pub(crate) const ROOT_CERT_KEY: &str = "OP_DB_SSLROOTCERT";
const MODE_KEY: &str = "OP_DB_SSLMODE";

/// What a handshake checks of the server's certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Check {
    /// Nothing: the connection is encrypted, the peer is not authenticated.
    Nothing,
    /// The chain up to a trusted root, not the host name.
    Chain,
    /// The chain and the host name.
    ChainAndName,
}

fn check_of(mode: SslMode, has_roots_file: bool) -> Check {
    match mode {
        SslMode::Disable | SslMode::Prefer => Check::Nothing,
        SslMode::Require if has_roots_file => Check::Chain,
        SslMode::Require => Check::Nothing,
        SslMode::VerifyCa => Check::Chain,
        SslMode::VerifyFull => Check::ChainAndName,
    }
}

/// Build the connector for `mode`.
///
/// # Errors
///
/// [`SettingsError::Invalid`] naming `OP_DB_SSLROOTCERT` when the file cannot be read or holds no
/// usable certificate, or naming `OP_DB_SSLMODE` when the platform's roots cannot be loaded.
pub(crate) fn connector(
    mode: SslMode,
    root_cert: Option<&Path>,
) -> Result<MakeRustlsConnect, SettingsError> {
    let provider = op_config::install_tls_provider();
    let verifier: Arc<dyn ServerCertVerifier> = match check_of(mode, root_cert.is_some()) {
        Check::Nothing => Arc::new(AcceptAny(provider.clone())),
        check => {
            let trusted = trusted(root_cert, &provider)?;
            if check == Check::Chain {
                Arc::new(IgnoreName(trusted))
            } else {
                trusted
            }
        }
    };
    let config = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| invalid(MODE_KEY, format!("the TLS client cannot be built: {e}")))?
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_no_client_auth();
    Ok(MakeRustlsConnect::new(config))
}

fn invalid(key: &'static str, reason: String) -> SettingsError {
    SettingsError::Invalid { key, reason }
}

/// The verifier that checks chain and name against the roots file, or the platform's roots.
fn trusted(
    root_cert: Option<&Path>,
    provider: &Arc<CryptoProvider>,
) -> Result<Arc<dyn ServerCertVerifier>, SettingsError> {
    if let Some(path) = root_cert {
        let roots = roots_from_file(path)?;
        let verifier =
            WebPkiServerVerifier::builder_with_provider(Arc::new(roots), provider.clone())
                .build()
                .map_err(|e| invalid(ROOT_CERT_KEY, format!("{}: {e}", path.display())))?;
        return Ok(verifier);
    }
    let verifier = PlatformVerifier::new(provider.clone()).map_err(|e| {
        invalid(
            MODE_KEY,
            format!(
                "the platform's trust roots cannot be loaded ({e}); install the CA certificates \
                 or name a CA bundle in {ROOT_CERT_KEY}"
            ),
        )
    })?;
    Ok(Arc::new(verifier))
}

/// The certificates of a PEM bundle (a managed service publishes its own, for example the Amazon
/// RDS bundle <https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.SSL.html>).
fn roots_from_file(path: &Path) -> Result<RootCertStore, SettingsError> {
    let shown = path.display();
    let certs = CertificateDer::pem_file_iter(path)
        .map_err(|e| invalid(ROOT_CERT_KEY, format!("{shown} cannot be read: {e}")))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| invalid(ROOT_CERT_KEY, format!("{shown} is not a PEM bundle: {e}")))?;
    let mut roots = RootCertStore::empty();
    let (added, ignored) = roots.add_parsable_certificates(certs);
    if added == 0 {
        return Err(invalid(
            ROOT_CERT_KEY,
            format!("{shown} holds no usable CA certificate"),
        ));
    }
    if ignored > 0 {
        tracing::warn!(
            ignored,
            "CA certificates in {ROOT_CERT_KEY} were not usable"
        );
    }
    Ok(roots)
}

/// `prefer` and an unverified `require`: the connection is encrypted and the peer is not
/// authenticated, as libpq's table says. The handshake signature is still checked, so the peer
/// must hold the private key of the certificate it presents.
struct AcceptAny(Arc<CryptoProvider>);

impl fmt::Debug for AcceptAny {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AcceptAny")
    }
}

impl ServerCertVerifier for AcceptAny {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

/// `verify-ca` (and `require` with a roots file): the chain is verified, the host name is not.
/// webpki checks the chain before the name, so a "name does not match" answer means the chain
/// already verified; every other error stays an error.
#[derive(Debug)]
struct IgnoreName(Arc<dyn ServerCertVerifier>);

impl ServerCertVerifier for IgnoreName {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        match self
            .0
            .verify_server_cert(end_entity, intermediates, server_name, ocsp_response, now)
        {
            Err(TlsError::InvalidCertificate(
                CertificateError::NotValidForName | CertificateError::NotValidForNameContext { .. },
            )) => Ok(ServerCertVerified::assertion()),
            other => other,
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        self.0.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        self.0.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.supported_verify_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_mode_checks_what_libpqs_table_says() {
        for (mode, roots_file, check) in [
            (SslMode::Disable, false, Check::Nothing),
            (SslMode::Prefer, false, Check::Nothing),
            (SslMode::Require, false, Check::Nothing),
            (SslMode::Require, true, Check::Chain),
            (SslMode::VerifyCa, false, Check::Chain),
            (SslMode::VerifyCa, true, Check::Chain),
            (SslMode::VerifyFull, false, Check::ChainAndName),
            (SslMode::VerifyFull, true, Check::ChainAndName),
        ] {
            assert_eq!(check_of(mode, roots_file), check, "{mode:?} {roots_file}");
        }
    }
}
