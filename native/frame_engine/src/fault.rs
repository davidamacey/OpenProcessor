//! What a failure on the card means for the photo and the process (`frame_engine/cuda_fault.py`
//! and `prep_budget.is_memory_error`, plan 17 sections 1.3 and 2.3): a lost context refuses every
//! later photo and ends the process for a restart; a failed allocation answers the photo
//! `gpu_memory`; anything else fails that photo alone.
//!
//! The classification itself is `gpu_frame`'s (the sticky-code table of `cuda/error.rs`,
//! `AllocError::is_out_of_memory`); this module only carries it to the photo's answer.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use gpu_frame::{AllocError, CudaError, Fault, JobError, KernelError};

use crate::geometry::GeometryError;
use crate::pipeline::PipelineError;

/// A failure of the photo's work on the card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuFault {
    /// A device allocation failed (the engine's cap, `CUDA_ERROR_OUT_OF_MEMORY` or nvJPEG's
    /// allocator): the photo answers `gpu_memory`.
    OutOfMemory(String),
    /// A sticky code: the context is unusable.
    ContextLost(String),
    /// Anything else: the photo fails (`INTERNAL` with the engine's prefix).
    Failed(String),
}

impl fmt::Display for GpuFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory(m) => write!(f, "device memory: {m}"),
            Self::ContextLost(m) => write!(f, "CUDA context lost: {m}"),
            Self::Failed(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for GpuFault {}

impl From<CudaError> for GpuFault {
    fn from(e: CudaError) -> Self {
        match e.fault() {
            Fault::ContextLost => Self::ContextLost(e.to_string()),
            Fault::OutOfMemory => Self::OutOfMemory(format!("the card's memory: {e}")),
            Fault::Other => Self::Failed(e.to_string()),
        }
    }
}

/// Memory refusals seen since start (`FrameEngine.memory_events`).
static MEMORY_EVENTS: AtomicU64 = AtomicU64::new(0);

/// Log a photo answered `gpu_memory`: where, the photo's oriented size and the refusal (bytes
/// asked, bytes in use and the limit, or the card's own refusal). The first and then every
/// hundredth are logged, as the Python engine's `_note_memory_failure` does, so a flood of
/// refusals (a card shared beyond its memory) cannot flood the log; the count is in each line.
pub fn note_memory_failure(stage: &str, (w, h): (u32, u32), detail: &str) {
    let event = MEMORY_EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
    if event % 100 == 1 {
        tracing::warn!(
            "gpu_memory in {stage}: photo {w} x {h}, frame {} bytes: {detail}; memory refusal {event}",
            u64::from(w) * u64::from(h) * 3
        );
    }
}

impl From<AllocError> for GpuFault {
    fn from(e: AllocError) -> Self {
        match e {
            AllocError::Cuda(cuda) => cuda.into(),
            cap @ AllocError::Cap(_) => Self::OutOfMemory(cap.to_string()),
        }
    }
}

impl From<KernelError> for GpuFault {
    fn from(e: KernelError) -> Self {
        match e {
            KernelError::Cuda(cuda) => cuda.into(),
            compile @ KernelError::Compile { .. } => Self::Failed(compile.to_string()),
        }
    }
}

impl From<GeometryError> for GpuFault {
    fn from(e: GeometryError) -> Self {
        Self::Failed(e.to_string())
    }
}

/// A pool job that gave no answer: a panic fails the photo (the thread goes on, plan 17 section
/// 1.3); a cancelled job belongs to a caller that is gone; a closed pool is a stopping process.
pub fn job_failed(stage: &str, e: &JobError) -> PipelineError {
    PipelineError::Engine(format!("{stage}: {e}"))
}

#[cfg(test)]
mod tests {
    use gpu_frame::CapExceeded;
    use gpu_frame::cuda::{CUresult, STICKY};

    use super::*;

    fn driver(code: CUresult) -> CudaError {
        CudaError::Driver {
            call: "cuLaunchKernel",
            code,
        }
    }

    #[test]
    fn every_sticky_code_loses_the_context_and_memory_answers_gpu_memory() {
        for code in STICKY {
            assert!(
                matches!(GpuFault::from(driver(code)), GpuFault::ContextLost(_)),
                "{code:?}"
            );
        }
        assert!(matches!(
            GpuFault::from(driver(CUresult::CUDA_ERROR_OUT_OF_MEMORY)),
            GpuFault::OutOfMemory(_)
        ));
        // 214, an uncorrectable ECC error, carries no "relaunch" sentence: the photo fails alone
        assert!(matches!(
            GpuFault::from(driver(CUresult::CUDA_ERROR_ECC_UNCORRECTABLE)),
            GpuFault::Failed(_)
        ));
    }

    #[test]
    fn the_engines_cap_and_a_driver_refusal_are_both_memory_and_a_kernel_carries_its_cause() {
        let cap = AllocError::Cap(CapExceeded {
            need: 10,
            in_use: 5,
            limit: 8,
        });
        assert!(matches!(GpuFault::from(cap), GpuFault::OutOfMemory(_)));
        let oom = AllocError::Cuda(driver(CUresult::CUDA_ERROR_OUT_OF_MEMORY));
        assert!(matches!(GpuFault::from(oom), GpuFault::OutOfMemory(_)));
        let lost = KernelError::Cuda(driver(CUresult::CUDA_ERROR_ILLEGAL_ADDRESS));
        assert!(matches!(GpuFault::from(lost), GpuFault::ContextLost(_)));
        let compile = KernelError::Compile {
            name: "orient".into(),
            log: "error".into(),
        };
        assert!(matches!(GpuFault::from(compile), GpuFault::Failed(_)));
    }

    #[test]
    fn a_job_that_panicked_fails_the_photo_with_its_stage() {
        let e = job_failed("decode", &JobError::Panicked("boom".into()));
        assert_eq!(e, PipelineError::Engine("decode: panicked: boom".into()));
    }
}
