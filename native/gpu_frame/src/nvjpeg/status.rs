//! nvJPEG's vocabulary as values: the backends a lane tries and the status codes, named as
//! `nvjpeg.h` names them (`nvidia-nvjpeg` 13.2.3.58).

use std::ffi::c_int;
use std::fmt;

/// The nvJPEG implementations a lane tries, in this order (`nvjpeg_lanes.py`): Huffman decoding
/// on the GPU first (baseline JPEGs), then on the calling thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Huffman decoding on the GPU (`NVJPEG_BACKEND_GPU_HYBRID`); baseline JPEGs only.
    GpuHybrid,
    /// Huffman decoding on the calling thread (`NVJPEG_BACKEND_HYBRID`), the rest on the GPU.
    Hybrid,
}

impl Backend {
    /// The order of attempt.
    pub const ORDER: [Backend; 2] = [Backend::GpuHybrid, Backend::Hybrid];

    pub(crate) fn raw(self) -> c_int {
        match self {
            Backend::Hybrid => 1,
            Backend::GpuHybrid => 2,
        }
    }

    /// The name the Python engine reports (`gpu_hybrid`, `hybrid`).
    pub fn name(self) -> &'static str {
        match self {
            Backend::GpuHybrid => "gpu_hybrid",
            Backend::Hybrid => "hybrid",
        }
    }
}

/// An `nvjpegStatus_t` other than success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvjpegStatus(pub i32);

impl NvjpegStatus {
    /// `NVJPEG_STATUS_ALLOCATOR_FAILURE`: the photo answers `gpu_memory` (`prep_budget.py`).
    pub const ALLOCATOR_FAILURE: NvjpegStatus = NvjpegStatus(5);

    /// The header's name of the status.
    pub fn name(self) -> &'static str {
        match self.0 {
            0 => "NVJPEG_STATUS_SUCCESS",
            1 => "NVJPEG_STATUS_NOT_INITIALIZED",
            2 => "NVJPEG_STATUS_INVALID_PARAMETER",
            3 => "NVJPEG_STATUS_BAD_JPEG",
            4 => "NVJPEG_STATUS_JPEG_NOT_SUPPORTED",
            5 => "NVJPEG_STATUS_ALLOCATOR_FAILURE",
            6 => "NVJPEG_STATUS_EXECUTION_FAILED",
            7 => "NVJPEG_STATUS_ARCH_MISMATCH",
            8 => "NVJPEG_STATUS_INTERNAL_ERROR",
            9 => "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED",
            10 => "NVJPEG_STATUS_INCOMPLETE_BITSTREAM",
            _ => "NVJPEG_STATUS_UNKNOWN",
        }
    }
}

impl fmt::Display for NvjpegStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name(), self.0)
    }
}

/// A failed nvJPEG call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvjpegFailure {
    /// The nvJPEG function that failed, as `nvjpeg.h` names it.
    pub call: &'static str,
    /// The status code it answered.
    pub status: NvjpegStatus,
}

impl fmt::Display for NvjpegFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} answered {}", self.call, self.status)
    }
}

pub(crate) fn check(call: &'static str, status: c_int) -> Result<(), NvjpegFailure> {
    if status == 0 {
        Ok(())
    } else {
        Err(NvjpegFailure {
            call,
            status: NvjpegStatus(status),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_status_is_named_as_the_header_names_it() {
        let names: Vec<&str> = (0..=10).map(|s| NvjpegStatus(s).name()).collect();
        assert_eq!(
            names,
            [
                "NVJPEG_STATUS_SUCCESS",
                "NVJPEG_STATUS_NOT_INITIALIZED",
                "NVJPEG_STATUS_INVALID_PARAMETER",
                "NVJPEG_STATUS_BAD_JPEG",
                "NVJPEG_STATUS_JPEG_NOT_SUPPORTED",
                "NVJPEG_STATUS_ALLOCATOR_FAILURE",
                "NVJPEG_STATUS_EXECUTION_FAILED",
                "NVJPEG_STATUS_ARCH_MISMATCH",
                "NVJPEG_STATUS_INTERNAL_ERROR",
                "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED",
                "NVJPEG_STATUS_INCOMPLETE_BITSTREAM",
            ]
        );
        assert_eq!(NvjpegStatus(11).name(), "NVJPEG_STATUS_UNKNOWN");
        assert_eq!(NvjpegStatus(-1).name(), "NVJPEG_STATUS_UNKNOWN");
        assert_eq!(
            NvjpegStatus::ALLOCATOR_FAILURE.name(),
            "NVJPEG_STATUS_ALLOCATOR_FAILURE"
        );
    }

    #[test]
    fn backends_are_tried_gpu_huffman_first_with_the_python_names() {
        assert_eq!(
            Backend::ORDER.map(Backend::raw),
            [2, 1],
            "NVJPEG_BACKEND_GPU_HYBRID, then HYBRID"
        );
        assert_eq!(Backend::ORDER.map(Backend::name), ["gpu_hybrid", "hybrid"]);
    }
}
