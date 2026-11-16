//! An in-process PostgreSQL stand-in on a loopback socket: it answers the `SSLRequest`, does the
//! TLS handshake with a certificate the test generated, reads the startup message and answers
//! `AuthenticationOk` and `ReadyForQuery`, so a client that gets that far has connected. It
//! reports what it saw of each connection.

use std::io;
use std::net::Ipv4Addr;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_rustls::TlsAcceptor;

use crate::pki::ServerCert;

const SSL_REQUEST_CODE: u32 = 80_877_103;
/// `AuthenticationOk`, then `ReadyForQuery` (idle).
const WELCOME: [u8; 14] = [b'R', 0, 0, 0, 8, 0, 0, 0, 0, b'Z', 0, 0, 0, 5];

/// What the server does when it gets an `SSLRequest`.
pub enum Answer {
    /// `S`, then a TLS handshake with this certificate.
    Tls(ServerCert),
    /// `N`: no TLS here.
    No,
    /// `S`, then the connection closes: a handshake that cannot succeed.
    TlsThenClose,
}

/// What the server saw of one connection.
#[derive(Debug, Default, Clone)]
pub struct Seen {
    /// The client sent an `SSLRequest` first.
    pub ssl_request: bool,
    /// The TLS handshake completed.
    pub tls: bool,
    /// The client sent its startup message (so it accepted the server).
    pub startup: bool,
    /// The handshake failed, and how the server saw it.
    pub handshake_error: Option<String>,
}

pub struct FakeServer {
    pub port: u16,
    seen: mpsc::Receiver<Seen>,
}

impl FakeServer {
    pub async fn start(answer: Answer) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, seen) = mpsc::channel(16);
        let answer = Arc::new(answer);
        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                let (answer, tx) = (answer.clone(), tx.clone());
                tokio::spawn(async move {
                    let seen = serve(socket, &answer).await;
                    drop(tx.send(seen).await);
                });
            }
        });
        Self { port, seen }
    }

    /// What the server saw of the next finished connection.
    pub async fn seen(&mut self) -> Seen {
        tokio::time::timeout(Duration::from_secs(10), self.seen.recv())
            .await
            .expect("the server saw no connection")
            .expect("the server stopped")
    }
}

async fn serve(mut socket: TcpStream, answer: &Answer) -> Seen {
    let mut seen = Seen::default();
    let mut head = [0_u8; 8];
    if socket.read_exact(&mut head).await.is_err() {
        return seen;
    }
    if u32::from_be_bytes([head[4], head[5], head[6], head[7]]) != SSL_REQUEST_CODE {
        seen.startup = finish_startup(&mut socket, head).await.is_ok();
        return seen;
    }
    seen.ssl_request = true;
    match answer {
        Answer::No => serve_plain_after_no(socket, &mut seen).await,
        Answer::TlsThenClose => drop(socket.write_all(b"S").await),
        Answer::Tls(cert) => serve_tls(socket, cert, &mut seen).await,
    }
    seen
}

async fn serve_plain_after_no(mut socket: TcpStream, seen: &mut Seen) {
    let mut head = [0_u8; 8];
    if socket.write_all(b"N").await.is_ok() && socket.read_exact(&mut head).await.is_ok() {
        seen.startup = finish_startup(&mut socket, head).await.is_ok();
    }
}

async fn serve_tls(mut socket: TcpStream, cert: &ServerCert, seen: &mut Seen) {
    if socket.write_all(b"S").await.is_err() {
        return;
    }
    match TlsAcceptor::from(cert.0.clone()).accept(socket).await {
        Err(e) => seen.handshake_error = Some(e.to_string()),
        Ok(mut tls) => {
            seen.tls = true;
            let mut head = [0_u8; 8];
            if tls.read_exact(&mut head).await.is_ok() {
                seen.startup = finish_startup(&mut tls, head).await.is_ok();
            }
        }
    }
}

/// Read the rest of the startup message whose first eight bytes are `head`, and welcome the
/// client.
async fn finish_startup<S>(stream: &mut S, head: [u8; 8]) -> io::Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let length = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
    let mut rest = vec![0_u8; length.saturating_sub(8)];
    stream.read_exact(&mut rest).await?;
    let mut welcome = WELCOME.to_vec();
    welcome.push(b'I');
    stream.write_all(&welcome).await?;
    stream.flush().await
}
