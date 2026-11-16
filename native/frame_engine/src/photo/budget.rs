//! The frame budget (`frame_engine/prep_budget.py`, plan 17 sections 2.2 and 2.3): what a photo's
//! frame holds on the card, reserved before it decodes and released with the frame, through
//! `gpu_frame`'s fair admission. One admission instance bounds both the photos in work
//! (`OP_FRAME_ENGINE_WORKERS`, past the front's in-flight cap) and the bytes
//! (`OP_FRAME_ENGINE_BUDGET_MIB`), waiting at most `OP_FRAME_ENGINE_BUDGET_WAIT_SECONDS`.

use std::sync::Arc;
use std::time::Instant;

use gpu_frame::{Admission, AdmitError, Permit};
use prometheus_client::metrics::gauge::Gauge;

use crate::admission::{FrameBudget, Overload};
use crate::jpeg_header::{JpegHeader, oriented_size};
use crate::pyfmt::format_g;

/// `prep_budget.need_bytes` without renditions: the decoded frame, twice while an EXIF tag other
/// than 1 is applied (the oriented copy next to it). This decides `over_budget`, as in Python.
pub fn need_bytes(h: &JpegHeader) -> u64 {
    let frame = u64::from(h.width) * u64::from(h.height) * 3;
    if h.orientation != 1 { 2 * frame } else { frame }
}

/// What is reserved: the need, or the frame and the largest resample scratch when that is more
/// (plan 17 section 2.3: `rows x width x 12` bytes, a term the Python budget leaves out). The
/// scratch holds the source rows one output reads, at most every row of the oriented frame, at
/// the widest output written (`widths`).
pub fn reserve_bytes(h: &JpegHeader, widths: &[u32]) -> u64 {
    let (_, rows) = oriented_size(h);
    let frame = u64::from(h.width) * u64::from(h.height) * 3;
    need_bytes(h).max(frame + scratch_bytes(rows, widths))
}

/// The same for a host-route frame of `(width, height)`: it arrives upright, so its need is the
/// one frame (`prep_budget.host_need_bytes`) and the reserve adds the scratch.
pub fn host_reserve_bytes((width, height): (u32, u32), widths: &[u32]) -> u64 {
    u64::from(width) * u64::from(height) * 3 + scratch_bytes(height, widths)
}

/// The resample scratch: every row of the frame at the widest output, 12 bytes each.
fn scratch_bytes(rows: u32, widths: &[u32]) -> u64 {
    let widest = widths.iter().copied().max().unwrap_or(0);
    u64::from(rows) * u64::from(widest) * 12
}

/// The budget on `gpu_frame`'s admission, keeping the in-use gauge.
pub struct GpuBudget {
    admission: Arc<Admission>,
    in_use: Gauge,
    wait_seconds: f64,
}

impl GpuBudget {
    /// A budget of `budget_bytes` of decoded frame for `workers` photos, waiting at most `wait_seconds`; `in_use` is the gauge it keeps current.
    pub fn new(workers: u32, budget_bytes: u64, wait_seconds: f64, in_use: Gauge) -> Self {
        Self {
            admission: Arc::new(Admission::new(workers, budget_bytes)),
            in_use,
            wait_seconds,
        }
    }

    /// The whole frame budget, bytes.
    pub fn total_bytes(&self) -> u64 {
        self.admission.budget_bytes()
    }

    /// Bytes of decoded frame reserved now.
    pub fn bytes_in_use(&self) -> u64 {
        self.admission.bytes_in_use()
    }

    /// Worker slots held now.
    pub fn workers_in_use(&self) -> u64 {
        self.admission.count_in_use()
    }
}

/// A worker slot and the frame's bytes, held until dropped (with the frame).
pub struct BudgetPermit {
    permit: Option<Permit>,
    admission: Arc<Admission>,
    in_use: Gauge,
}

impl Drop for BudgetPermit {
    fn drop(&mut self) {
        drop(self.permit.take());
        self.in_use
            .set(i64::try_from(self.admission.bytes_in_use()).unwrap_or(i64::MAX));
    }
}

impl FrameBudget for GpuBudget {
    type Permit = BudgetPermit;

    async fn admit(&self, bytes: u64, deadline: Instant) -> Result<BudgetPermit, Overload> {
        let at = tokio::time::Instant::from_std(deadline);
        let permit = self.admission.admit(1, bytes, at).await.map_err(|e| {
            let detail = match e {
                AdmitError::Overload(_) => format!(
                    "{bytes} bytes not free within {} s",
                    format_g(self.wait_seconds)
                ),
                AdmitError::NeverFits(_) => format!("{bytes} bytes exceed the whole budget"),
            };
            Overload {
                cause: "budget".into(),
                detail,
            }
        })?;
        self.in_use
            .set(i64::try_from(self.admission.bytes_in_use()).unwrap_or(i64::MAX));
        Ok(BudgetPermit {
            permit: Some(permit),
            admission: Arc::clone(&self.admission),
            in_use: self.in_use.clone(),
        })
    }
}
