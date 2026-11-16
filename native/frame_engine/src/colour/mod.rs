//! The colour step (`frame_engine/colour.py`, plan 17 section 3 "Colour"): a photo whose
//! embedded profile moves colours is converted to sRGB on the card, before any branch reads its
//! frame, with the rule both decode routes follow ([`classify`], a port of
//! `src/utils/icc_srgb.py`). An untagged or sRGB-class photo is left exactly as decoded and costs
//! nothing here; a grey, CMYK or unreadable profile is ignored.
//!
//! A `convert` profile is applied as Little CMS's answer at every 8-bit input: a table of
//! `2^24 x 3` bytes (48 MiB), built once per distinct profile, uploaded, then gathered per pixel
//! in place on the frame by `kernels/colour.cu`. Tables: at most `OP_FRAME_ENGINE_COLOUR_TABLES`
//! stay on the card, least recently used first out, charged to the frame cap
//! (`OP_FRAME_ENGINE_POOL_MIB`); a table's memory returns only when its last holder (a decode job
//! that waited for its stream) drops it.
//!
//! Single flight: the first photo that meets a new profile starts its build as one job (on the
//! finish pool, so no thread that feeds the card builds and the transform is used on that one
//! thread); every photo with the same profile, the first included, waits for that job's answer
//! asynchronously in its control task, holding no pool thread, bounded by [`BUILD_WAIT`] (then an
//! overload with cause `colour`, retried by the API). The job writes its answer into the cache
//! itself, so a waiter that gives up never makes a second build start. A build that fails is not
//! remembered: the photos waiting for it fail, and the next photo with that profile builds again.
//! The kind of every profile seen is remembered (a hash and a word each, at most [`KINDS`]).

pub mod classify;
pub mod curves;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use gpu_frame::DeviceRange;
use prometheus_client::metrics::gauge::Gauge;
use sha2::{Digest, Sha256};
use tokio::sync::watch;

use self::classify::Kind;
use crate::admission::Overload;
use crate::fault::GpuFault;

/// Bytes of one conversion table on the card.
pub const TABLE_BYTES: usize = classify::TABLE_ENTRIES * 3;
/// How long a photo waits for a table another photo is building (`BUILD_WAIT_SECONDS`).
pub const BUILD_WAIT: Duration = Duration::from_secs(30);
/// Distinct profiles whose kind is remembered before the memo starts over (`KINDS`).
pub const KINDS: usize = 4096;

/// A conversion table on the card.
pub type Table = Arc<dyn DeviceRange + Send + Sync>;

/// What the decode does with the photo's colours.
#[derive(Clone)]
pub enum Colour {
    /// Untagged or an sRGB-class profile: the frame is left as decoded.
    Srgb,
    /// Not an RGB profile, or a profile of a grey frame: left as decoded.
    Ignored,
    /// Gathered through this table.
    Convert(Table),
}

impl Colour {
    /// The `op_frame_engine_colour` label of the outcome.
    pub fn label(&self) -> &'static str {
        match self {
            Colour::Srgb => "srgb",
            Colour::Ignored => "ignored",
            Colour::Convert(_) => "converted",
        }
    }
}

impl std::fmt::Debug for Colour {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Why a photo got no colour answer.
#[derive(Debug, Clone)]
pub enum ColourError {
    /// The table was not built within [`BUILD_WAIT`].
    Overload(Overload),
    /// The build failed on the card (`gpu_memory` when it is an allocation).
    Fault(GpuFault),
}

type Key = [u8; 32];
type Built = Result<Kind, GpuFault>;

#[derive(Default)]
struct State {
    kinds: HashMap<Key, Kind>,
    tables: VecDeque<(Key, Table)>,
    building: HashMap<Key, watch::Receiver<Option<Built>>>,
}

/// The step's caches, shared by every photo of the engine.
pub struct ColourStep {
    capacity: usize,
    wait: Duration,
    state: Arc<Mutex<State>>,
    table_bytes: Gauge,
}

/// Runs one build job somewhere off the card-feeding threads (the finish pool).
pub type Spawn<'a> = &'a (dyn Fn(Box<dyn FnOnce() + Send>) -> Result<(), GpuFault> + Sync);
/// Uploads a host table to the card, on the thread that runs the build.
pub type Upload = Arc<dyn Fn(&[u8]) -> Result<Table, GpuFault> + Send + Sync>;

impl ColourStep {
    /// A step keeping at most `capacity` tables, reporting their bytes in `table_bytes`.
    pub fn new(capacity: usize, table_bytes: Gauge) -> Self {
        Self::with_wait(capacity, table_bytes, BUILD_WAIT)
    }

    fn with_wait(capacity: usize, table_bytes: Gauge, wait: Duration) -> Self {
        Self {
            capacity: capacity.max(1),
            wait,
            state: Arc::new(Mutex::new(State::default())),
            table_bytes,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// What the decode of a frame of `components` channels tagged with `icc` does
    /// (`engine._decode`: a profile is read for colour frames only).
    ///
    /// # Errors
    ///
    /// [`ColourError::Overload`] when the table is not built within the wait;
    /// [`ColourError::Fault`] when its build failed.
    pub async fn lookup(
        &self,
        icc: Option<&[u8]>,
        components: u32,
        spawn: Spawn<'_>,
        upload: &Upload,
    ) -> Result<Colour, ColourError> {
        let Some(icc) = icc.filter(|b| !b.is_empty()) else {
            return Ok(Colour::Srgb);
        };
        if components != 3 {
            return Ok(Colour::Ignored);
        }
        let key: Key = Sha256::digest(icc).into();
        loop {
            let mut waiting = match self.cached(&key) {
                Ok(colour) => return Ok(colour),
                Err(Some(rx)) => rx,
                Err(None) => self.start(key, icc.to_vec(), spawn, upload)?,
            };
            let answer = tokio::time::timeout(self.wait, waiting.wait_for(Option::is_some)).await;
            let built = match answer {
                Ok(Ok(done)) => done.clone(),
                _ => None,
            };
            match built {
                Some(Ok(Kind::Convert)) => {} // in the cache now, or evicted since: look again
                Some(Ok(kind)) => return Ok(simple(kind)),
                Some(Err(fault)) => return Err(ColourError::Fault(fault)),
                None => {
                    return Err(ColourError::Overload(Overload {
                        cause: "colour".into(),
                        detail: format!(
                            "profile table not built within {} s",
                            self.wait.as_secs_f64()
                        ),
                    }));
                }
            }
        }
    }

    /// The answer the caches hold, or the build to wait for (`Err(Some)`), or none (`Err(None)`).
    fn cached(&self, key: &Key) -> Result<Colour, Option<watch::Receiver<Option<Built>>>> {
        let mut state = self.lock();
        match state.kinds.get(key) {
            Some(Kind::Convert) | None => {}
            Some(kind) => return Ok(simple(*kind)),
        }
        if let Some(k) = state.tables.iter().position(|(k, _)| k == key)
            && let Some(entry) = state.tables.remove(k)
        {
            let table = Arc::clone(&entry.1);
            state.tables.push_back(entry);
            return Ok(Colour::Convert(table));
        }
        Err(state.building.get(key).cloned())
    }

    /// Register this photo's build of `key` and start it; the receiver answers when it ends.
    fn start(
        &self,
        key: Key,
        icc: Vec<u8>,
        spawn: Spawn<'_>,
        upload: &Upload,
    ) -> Result<watch::Receiver<Option<Built>>, ColourError> {
        let (tx, rx) = watch::channel(None);
        {
            let mut state = self.lock();
            if let Some(other) = state.building.get(&key) {
                return Ok(other.clone());
            }
            state.building.insert(key, rx.clone());
        }
        let job = BuildJob {
            key,
            icc,
            upload: Arc::clone(upload),
            state: Arc::clone(&self.state),
            capacity: self.capacity,
            table_bytes: self.table_bytes.clone(),
            tx: Some(tx),
        };
        if let Err(fault) = spawn(Box::new(move || job.run())) {
            self.lock().building.remove(&key);
            return Err(ColourError::Fault(fault));
        }
        Ok(rx)
    }

    /// Device bytes of the tables held now.
    pub fn table_bytes(&self) -> usize {
        self.lock().tables.len() * TABLE_BYTES
    }
}

fn simple(kind: Kind) -> Colour {
    match kind {
        Kind::Srgb => Colour::Srgb,
        Kind::Other | Kind::Convert => Colour::Ignored,
    }
}

/// One build: classify, upload a `convert` table, then answer every waiter through the cache.
struct BuildJob {
    key: Key,
    icc: Vec<u8>,
    upload: Upload,
    state: Arc<Mutex<State>>,
    capacity: usize,
    table_bytes: Gauge,
    /// Taken when the job answers; still here when the job is dropped unrun or unwinds.
    tx: Option<watch::Sender<Option<Built>>>,
}

impl Drop for BuildJob {
    /// A build that never ran (the pool dropped it) or panicked answers its waiters with a
    /// failure and leaves no build registered, so the next photo builds again.
    fn drop(&mut self) {
        if let Some(tx) = self.tx.take() {
            self.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .building
                .remove(&self.key);
            let _ = tx.send(Some(Err(GpuFault::Failed(
                "the colour table build did not finish".into(),
            ))));
        }
    }
}

impl BuildJob {
    fn run(mut self) {
        let (kind, host) = classify::classify(&self.icc);
        let table = host.map(|h| (self.upload)(&h)).transpose();
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.building.remove(&self.key);
        let answer = match table {
            Ok(table) => {
                if state.kinds.len() >= KINDS {
                    state.kinds.clear();
                }
                state.kinds.insert(self.key, kind);
                if let Some(table) = table {
                    state.tables.push_back((self.key, table));
                    while state.tables.len() > self.capacity {
                        state.tables.pop_front();
                    }
                }
                let bytes = state.tables.len() * TABLE_BYTES;
                self.table_bytes
                    .set(i64::try_from(bytes).unwrap_or(i64::MAX));
                Ok(kind)
            }
            Err(fault) => Err(fault),
        };
        drop(state);
        // no receiver left is fine: every waiter gave up, the cache holds the answer
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Some(answer));
        }
    }
}

#[cfg(test)]
mod tests;
