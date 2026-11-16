//! The region's registration with the model server: purge at start, register, register again
//! when the server lost it (single flight), unregister at the end (`model_server.rs` has the
//! rules). The server is reached through [`ServerApi`], so the lifecycle is tested without one.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::Mutex;
use tonic::Status;

use crate::region::IPC_HANDLE_BYTES;

/// The calls the registration makes.
pub(crate) trait ServerApi: Send + Sync {
    fn server_ready(&self, timeout: Duration) -> impl Future<Output = Result<bool, Status>> + Send;
    /// The names of every registered CUDA region (status with an empty name).
    fn region_names(
        &self,
        timeout: Duration,
    ) -> impl Future<Output = Result<Vec<String>, Status>> + Send;
    fn register(
        &self,
        name: &str,
        raw_handle: &[u8],
        device_id: i64,
        bytes: u64,
        timeout: Duration,
    ) -> impl Future<Output = Result<(), Status>> + Send;
    /// Unregister one region; an empty name is refused before anything is sent.
    fn unregister(
        &self,
        name: &str,
        timeout: Duration,
    ) -> impl Future<Output = Result<(), Status>> + Send;
}

/// Why the region was registered (the Python engine's metric label).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistrationCause {
    /// The first registration, after the purge.
    Start,
    /// The health tick found the region missing (the server restarted).
    ServerLostRegion,
    /// A call that used the region was refused and the server's status confirmed it missing.
    RegionUnknown,
}

impl RegistrationCause {
    /// The short name of the cause, used as a metric label.
    pub fn label(self) -> &'static str {
        match self {
            RegistrationCause::Start => "start",
            RegistrationCause::ServerLostRegion => "server_lost_region",
            RegistrationCause::RegionUnknown => "region_unknown",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// The answer of one health check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Health {
    /// Whether the server answered its readiness probe.
    pub server_ready: bool,
    /// Whether this engine's region is registered on the server.
    pub registered: bool,
    /// Leftover regions of this prefix unregistered by this check.
    pub purged: usize,
}

impl Health {
    /// Whether the server is ready and holds the region, so calls may be sent.
    pub fn ready(&self) -> bool {
        self.server_ready && self.registered
    }
}

/// The names among `names` that carry `prefix` (an empty prefix matches nothing, an empty name
/// is never returned: unregistering an empty name unregisters every region).
pub fn stale_regions(names: &[String], prefix: &str) -> Vec<String> {
    if prefix.is_empty() {
        return Vec::new();
    }
    names
        .iter()
        .filter(|n| !n.is_empty() && n.starts_with(prefix))
        .cloned()
        .collect()
}

/// What the server is told about the region.
pub(crate) struct Identity {
    /// Every region of this card's engine carries it; purged at start.
    pub prefix: String,
    pub name: String,
    pub handle: [u8; IPC_HANDLE_BYTES],
    pub bytes: u64,
    /// The card as the model server numbers it.
    pub device_id: i64,
}

pub(crate) struct Registrar<A> {
    api: A,
    id: Identity,
    timeout: Duration,
    /// The server may hold an import (the region's own flag): set before a register is sent,
    /// cleared only by a successful unregister, so the memory is never freed under an import.
    may_import: Arc<AtomicBool>,
    registered: AtomicBool,
    server_ready: AtomicBool,
    purged: AtomicBool,
    generation: AtomicU64,
    single: Mutex<()>,
    counts: [AtomicU64; 3],
}

impl<A: ServerApi> Registrar<A> {
    pub(crate) fn new(
        api: A,
        id: Identity,
        timeout: Duration,
        may_import: Arc<AtomicBool>,
    ) -> Self {
        Registrar {
            api,
            id,
            timeout,
            may_import,
            registered: AtomicBool::new(false),
            server_ready: AtomicBool::new(false),
            purged: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            single: Mutex::new(()),
            counts: Default::default(),
        }
    }

    pub(crate) fn api(&self) -> &A {
        &self.api
    }

    pub(crate) fn name(&self) -> &str {
        &self.id.name
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.server_ready.load(Ordering::Acquire) && self.registered.load(Ordering::Acquire)
    }

    pub(crate) fn registrations(&self, cause: RegistrationCause) -> u64 {
        self.counts[cause.index()].load(Ordering::Relaxed)
    }

    pub(crate) async fn check(&self) -> Health {
        let ready = matches!(self.api.server_ready(self.timeout).await, Ok(true));
        self.server_ready.store(ready, Ordering::Release);
        if !ready {
            self.registered.store(false, Ordering::Release);
            return Health {
                server_ready: false,
                registered: false,
                purged: 0,
            };
        }
        let mut purged = 0;
        let outcome = if self.purged.load(Ordering::Acquire) {
            match self.api.region_names(self.timeout).await {
                Ok(names) if names.contains(&self.id.name) => {
                    self.registered.store(true, Ordering::Release);
                    Ok(())
                }
                Ok(_) => {
                    self.reregister(self.generation(), RegistrationCause::ServerLostRegion)
                        .await
                }
                Err(e) => Err(e),
            }
        } else {
            let _one = self.single.lock().await;
            if self.purged.load(Ordering::Acquire) {
                Ok(())
            } else {
                match self.purge().await {
                    Ok(n) => {
                        purged = n;
                        let r = self.register_locked(RegistrationCause::Start).await;
                        self.purged.store(r.is_ok(), Ordering::Release);
                        r
                    }
                    Err(e) => Err(e),
                }
            }
        };
        if outcome.is_err() {
            self.registered.store(false, Ordering::Release);
        }
        Health {
            server_ready: true,
            registered: self.registered.load(Ordering::Acquire),
            purged,
        }
    }

    async fn purge(&self) -> Result<usize, Status> {
        let names = self.api.region_names(self.timeout).await?;
        let stale = stale_regions(&names, &self.id.prefix);
        for name in &stale {
            self.api.unregister(name, self.timeout).await?;
            if *name == self.id.name {
                self.may_import.store(false, Ordering::Release);
            }
        }
        Ok(stale.len())
    }

    /// Register under the single-flight lock (held by the caller).
    async fn register_locked(&self, cause: RegistrationCause) -> Result<(), Status> {
        self.registered.store(false, Ordering::Release);
        // Not registered (the usual case) or the server is down: the register call says which.
        let _ = self.api.unregister(&self.id.name, self.timeout).await;
        self.may_import.store(true, Ordering::Release);
        self.api
            .register(
                &self.id.name,
                &self.id.handle,
                self.id.device_id,
                self.id.bytes,
                self.timeout,
            )
            .await?;
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.registered.store(true, Ordering::Release);
        self.counts[cause.index()].fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Register again unless another task did since `seen` (a generation).
    pub(crate) async fn reregister(
        &self,
        seen: u64,
        cause: RegistrationCause,
    ) -> Result<(), Status> {
        let _one = self.single.lock().await;
        if self.generation() != seen {
            return Ok(());
        }
        self.register_locked(cause).await
    }

    /// Unregister; true when the server holds no import any more.
    pub(crate) async fn close(&self) -> bool {
        let _one = self.single.lock().await;
        self.registered.store(false, Ordering::Release);
        if !self.may_import.load(Ordering::Acquire) {
            return true;
        }
        let released = self
            .api
            .unregister(&self.id.name, self.timeout)
            .await
            .is_ok();
        if released {
            self.may_import.store(false, Ordering::Release);
        }
        released
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[derive(Default)]
    struct FakeServer {
        down: AtomicBool,
        refuse_unregister: AtomicBool,
        regions: StdMutex<Vec<String>>,
        log: StdMutex<Vec<String>>,
    }

    impl FakeServer {
        fn with(regions: &[&str]) -> FakeServer {
            let s = FakeServer::default();
            *s.regions.lock().unwrap() = regions.iter().map(|r| r.to_string()).collect();
            s
        }

        fn note(&self, line: String) {
            self.log.lock().unwrap().push(line);
        }

        fn gone(&self) -> Result<(), Status> {
            if self.down.load(Ordering::SeqCst) {
                Err(Status::unavailable("down"))
            } else {
                Ok(())
            }
        }
    }

    impl ServerApi for &FakeServer {
        async fn server_ready(&self, _t: Duration) -> Result<bool, Status> {
            self.gone().map(|_| true)
        }

        async fn region_names(&self, _t: Duration) -> Result<Vec<String>, Status> {
            self.gone()?;
            Ok(self.regions.lock().unwrap().clone())
        }

        async fn register(
            &self,
            name: &str,
            handle: &[u8],
            _d: i64,
            _b: u64,
            _t: Duration,
        ) -> Result<(), Status> {
            self.gone()?;
            assert_eq!(handle.len(), IPC_HANDLE_BYTES);
            tokio::task::yield_now().await;
            self.note(format!("register {name}"));
            self.regions.lock().unwrap().push(name.to_string());
            Ok(())
        }

        async fn unregister(&self, name: &str, _t: Duration) -> Result<(), Status> {
            assert!(
                !name.is_empty(),
                "an empty name would unregister every region"
            );
            self.gone()?;
            if self.refuse_unregister.load(Ordering::SeqCst) {
                return Err(Status::internal("refused"));
            }
            self.note(format!("unregister {name}"));
            self.regions.lock().unwrap().retain(|r| r != name);
            Ok(())
        }
    }

    const PREFIX: &str = "op-fe-card0-fe0-";
    const NAME: &str = "op-fe-card0-fe0-host-42";

    fn registrar(server: &FakeServer) -> (Registrar<&FakeServer>, Arc<AtomicBool>) {
        let flag = Arc::new(AtomicBool::new(false));
        let id = Identity {
            prefix: PREFIX.into(),
            name: NAME.into(),
            handle: [7; 64],
            bytes: 1 << 20,
            device_id: 0,
        };
        let r = Registrar::new(server, id, Duration::from_secs(1), flag.clone());
        (r, flag)
    }

    #[test]
    fn the_purge_matches_the_prefix_only() {
        let names: Vec<String> = [
            "op-fe-card0-fe0-old-1",
            "op-fe-card0-fe1-x",
            "op-fe-card1-fe0-y",
            "other",
            "",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(stale_regions(&names, PREFIX), ["op-fe-card0-fe0-old-1"]);
        assert!(stale_regions(&names, "").is_empty());
    }

    #[tokio::test]
    async fn the_first_check_purges_leftovers_then_registers() {
        let server = FakeServer::with(&[
            "op-fe-card0-fe0-dead-7",
            "op-fe-card1-fe0-peer-3",
            "other-client",
        ]);
        let (r, flag) = registrar(&server);
        let health = r.check().await;
        assert_eq!(
            health,
            Health {
                server_ready: true,
                registered: true,
                purged: 1
            }
        );
        assert!(r.is_ready() && flag.load(Ordering::SeqCst));
        assert_eq!(
            *server.regions.lock().unwrap(),
            ["op-fe-card1-fe0-peer-3", "other-client", NAME]
        );
        assert_eq!(r.registrations(RegistrationCause::Start), 1);
        // A later healthy check sends no register.
        assert_eq!(r.check().await.purged, 0);
        assert_eq!(
            server
                .log
                .lock()
                .unwrap()
                .iter()
                .filter(|l| l.starts_with("register"))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn a_server_that_lost_the_region_gets_it_again() {
        let server = FakeServer::with(&[]);
        let (r, _) = registrar(&server);
        assert!(r.check().await.ready());
        server.down.store(true, Ordering::SeqCst);
        assert!(
            !r.check().await.ready() && !r.is_ready(),
            "not ready while the server is down"
        );
        server.regions.lock().unwrap().clear(); // the restarted server forgot it
        server.down.store(false, Ordering::SeqCst);
        assert!(r.check().await.ready());
        assert_eq!(r.registrations(RegistrationCause::ServerLostRegion), 1);
        assert_eq!(
            r.registrations(RegistrationCause::Start),
            1,
            "the purge runs once"
        );
    }

    #[tokio::test]
    async fn concurrent_reregisters_register_once() {
        let server = FakeServer::with(&[]);
        let (r, _) = registrar(&server);
        assert!(r.check().await.ready());
        let seen = r.generation();
        let (a, b) = tokio::join!(
            r.reregister(seen, RegistrationCause::RegionUnknown),
            r.reregister(seen, RegistrationCause::RegionUnknown)
        );
        assert!(a.is_ok() && b.is_ok());
        assert_eq!(r.registrations(RegistrationCause::RegionUnknown), 1);
        assert_eq!(r.generation(), seen + 1);
    }

    #[tokio::test]
    async fn close_unregisters_before_the_memory_may_go() {
        let server = FakeServer::with(&[]);
        let (r, flag) = registrar(&server);
        assert!(r.check().await.ready());
        server.refuse_unregister.store(true, Ordering::SeqCst);
        assert!(
            !r.close().await,
            "a refused unregister leaves the memory to process exit"
        );
        assert!(flag.load(Ordering::SeqCst));
        server.refuse_unregister.store(false, Ordering::SeqCst);
        assert!(r.close().await);
        assert!(!flag.load(Ordering::SeqCst) && !r.is_ready());
        assert_eq!(
            server.log.lock().unwrap().last().map(String::as_str),
            Some(&*format!("unregister {NAME}"))
        );
    }

    #[tokio::test]
    async fn a_register_that_fails_keeps_the_import_flag() {
        let server = FakeServer::with(&[]);
        server.down.store(true, Ordering::SeqCst);
        let (r, flag) = registrar(&server);
        assert!(!r.check().await.ready());
        assert!(
            !flag.load(Ordering::SeqCst),
            "nothing was sent while the server was down"
        );
        server.down.store(false, Ordering::SeqCst);
        assert!(r.check().await.ready());
        assert!(flag.load(Ordering::SeqCst));
    }
}
