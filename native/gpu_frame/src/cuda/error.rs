//! Errors of CUDA and NVRTC calls as values, and how each bears on the process (plan 17
//! section 1.3): [`Fault::ContextLost`] for the codes after which "the process must be terminated
//! and relaunched", [`Fault::OutOfMemory`] for `CUDA_ERROR_OUT_OF_MEMORY` (section 2.3).

use std::fmt;

use cudarc::driver::result;

pub use cudarc::driver::sys::CUresult;

/// The codes after which the CUDA documentation says the process "must be terminated and
/// relaunched" (runtime API `cudaError`, plan 17 section 1.3), as the driver API names them.
/// Three such runtime codes have no `CUresult` in cudarc 0.19.10's CUDA 13.2 bindings and cannot
/// be returned through them: 226 `cudaErrorContained`, 721 `cudaErrorTensorMemoryLeak`, 810
/// `cudaErrorMpsClientTerminated` (the test `sticky_codes_follow_the_runtime_documentation`).
pub const STICKY: [CUresult; 10] = [
    CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
    CUresult::CUDA_ERROR_LAUNCH_TIMEOUT,
    CUresult::CUDA_ERROR_ASSERT,
    CUresult::CUDA_ERROR_HARDWARE_STACK_ERROR,
    CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION,
    CUresult::CUDA_ERROR_MISALIGNED_ADDRESS,
    CUresult::CUDA_ERROR_INVALID_ADDRESS_SPACE,
    CUresult::CUDA_ERROR_INVALID_PC,
    CUresult::CUDA_ERROR_LAUNCH_FAILED,
    CUresult::CUDA_ERROR_EXTERNAL_DEVICE,
];

/// How a failure bears on the process (plan 17 sections 1.3 and 2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// The context is unusable: refuse work, report not ready, exit for a restart.
    ContextLost,
    /// A device allocation failed: the photo answers `gpu_memory`.
    OutOfMemory,
    /// Anything else: the photo's own failure.
    Other,
}

/// The classification of one driver code.
pub fn fault_of(code: CUresult) -> Fault {
    if STICKY.contains(&code) {
        Fault::ContextLost
    } else if code == CUresult::CUDA_ERROR_OUT_OF_MEMORY {
        Fault::OutOfMemory
    } else {
        Fault::Other
    }
}

/// A failed CUDA or NVRTC call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    /// A shared library could not be loaded (no driver, no NVRTC on the library path).
    LibraryMissing {
        /// The library that failed to load, named for the log.
        library: &'static str,
    },
    /// A driver API call answered a code other than `CUDA_SUCCESS`.
    Driver {
        /// The driver function that failed, as the CUDA driver API names it.
        call: &'static str,
        /// The `CUresult` it answered.
        code: CUresult,
    },
    /// NVRTC refused a program; `log` is the compiler's log.
    Nvrtc {
        /// The NVRTC function that failed.
        call: &'static str,
        /// The `nvrtcResult` it answered.
        code: i32,
        /// The compiler's log; empty when the call gave none.
        log: String,
    },
    /// A request this crate refuses before any call (a range outside its buffer, a bad name).
    Invalid {
        /// What was refused, in words for the log.
        what: String,
    },
}

impl CudaError {
    /// How this failure bears on the process.
    pub fn fault(&self) -> Fault {
        match self {
            CudaError::Driver { code, .. } => fault_of(*code),
            _ => Fault::Other,
        }
    }

    pub(crate) fn invalid(what: impl Into<String>) -> Self {
        CudaError::Invalid { what: what.into() }
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::LibraryMissing { library } => {
                write!(
                    f,
                    "the {library} library could not be loaded (library path)"
                )
            }
            CudaError::Driver { call, code } => write!(f, "{call} answered {code:?}"),
            CudaError::Nvrtc { call, code, log } => write!(f, "{call} answered {code}: {log}"),
            CudaError::Invalid { what } => write!(f, "refused: {what}"),
        }
    }
}

impl std::error::Error for CudaError {}

pub(crate) fn driver<T>(
    call: &'static str,
    r: Result<T, result::DriverError>,
) -> Result<T, CudaError> {
    r.map_err(|e| CudaError::Driver { call, code: e.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (runtime code, runtime name, the driver's code): every `cudaError` whose documentation
    /// says the process must be terminated and relaunched
    /// (https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html, read
    /// 2026-11-15). `None`: no `CUresult` in cudarc 0.19.10's CUDA 13.2 bindings, so the driver
    /// calls this crate makes cannot return it.
    const RUNTIME_STICKY: [(i32, &str, Option<&str>); 13] = [
        (226, "cudaErrorContained", None),
        (
            700,
            "cudaErrorIllegalAddress",
            Some("CUDA_ERROR_ILLEGAL_ADDRESS"),
        ),
        (
            702,
            "cudaErrorLaunchTimeout",
            Some("CUDA_ERROR_LAUNCH_TIMEOUT"),
        ),
        (710, "cudaErrorAssert", Some("CUDA_ERROR_ASSERT")),
        (
            714,
            "cudaErrorHardwareStackError",
            Some("CUDA_ERROR_HARDWARE_STACK_ERROR"),
        ),
        (
            715,
            "cudaErrorIllegalInstruction",
            Some("CUDA_ERROR_ILLEGAL_INSTRUCTION"),
        ),
        (
            716,
            "cudaErrorMisalignedAddress",
            Some("CUDA_ERROR_MISALIGNED_ADDRESS"),
        ),
        (
            717,
            "cudaErrorInvalidAddressSpace",
            Some("CUDA_ERROR_INVALID_ADDRESS_SPACE"),
        ),
        (718, "cudaErrorInvalidPc", Some("CUDA_ERROR_INVALID_PC")),
        // The driver spells this one LAUNCH_FAILED.
        (
            719,
            "cudaErrorLaunchFailure",
            Some("CUDA_ERROR_LAUNCH_FAILED"),
        ),
        (721, "cudaErrorTensorMemoryLeak", None),
        (810, "cudaErrorMpsClientTerminated", None),
        (
            911,
            "cudaErrorExternalDevice",
            Some("CUDA_ERROR_EXTERNAL_DEVICE"),
        ),
    ];

    #[test]
    fn sticky_codes_follow_the_runtime_documentation() {
        let mapped: Vec<(i32, &str)> = RUNTIME_STICKY
            .iter()
            .filter_map(|(code, _, name)| name.map(|n| (*code, n)))
            .collect();
        assert_eq!(
            STICKY.len(),
            mapped.len(),
            "one driver code per documented runtime code"
        );
        for (code, name) in mapped {
            let found = STICKY.iter().find(|c| format!("{c:?}") == name);
            let found = found.unwrap_or_else(|| panic!("{name} missing from STICKY"));
            assert_eq!(*found as i32, code, "{name} has the runtime's value");
            assert_eq!(fault_of(*found), Fault::ContextLost);
        }
    }

    #[test]
    fn ecc_and_memory_errors_fail_only_the_photo() {
        // 214 carries no "terminated and relaunched" sentence (plan 17 section 1.3).
        assert_eq!(
            fault_of(CUresult::CUDA_ERROR_ECC_UNCORRECTABLE),
            Fault::Other
        );
        assert_eq!(
            fault_of(CUresult::CUDA_ERROR_OUT_OF_MEMORY),
            Fault::OutOfMemory
        );
        assert_eq!(fault_of(CUresult::CUDA_ERROR_INVALID_VALUE), Fault::Other);
        let e = CudaError::Driver {
            call: "cuLaunchKernel",
            code: CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
        };
        assert_eq!(e.fault(), Fault::ContextLost);
        assert_eq!(CudaError::invalid("x").fault(), Fault::Other);
    }
}
