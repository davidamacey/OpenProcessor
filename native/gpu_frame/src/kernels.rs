//! The kernel loader: CUDA C source text compiled by NVRTC with exactly the options CuPy 14.2.0's
//! `RawModule` passes, loaded as a module of the current device, cached by name and source
//! (plan 17 section 3, "Kernel build"; review row M1).
//!
//! CuPy's rule, read from `cupy/cuda/compiler.py` at v14.2.0
//! (<https://github.com/cupy/cupy/blob/v14.2.0/cupy/cuda/compiler.py>):
//! * `_compile_with_cache_cuda` adds `-ftz=true`;
//! * `_compile` then adds the arch option from `_get_arch_for_options_for_nvrtc` and
//!   `--device-as-default-execution-space` (jitify and `--pch` are off by default);
//! * the arch is the device's compute capability, capped by what NVRTC can target
//!   (`_get_max_compute_capability`: NVRTC 12.0 to 12.7 -> 90, 12.8 -> 120, 12.9 and 13+ -> 121;
//!   Tegra archs 32, 53, 62, 72, 87 are kept as they are);
//! * the output is a cubin with `-arch=sm_<arch>` when the arch is within that cap, else PTX
//!   (`_get_arch_for_options_for_nvrtc`, `CUPY_COMPILE_WITH_PTX` unset). The capped arch is
//!   always within the cap, so CuPy's rule always yields a cubin; the PTX branch is not ported.
//!
//! The same NVRTC build and the same options give the same binary as the Python engine's. The
//! kernel files themselves belong to the engine crate; this module compiles what it is given.
//! NVRTC: <https://docs.nvidia.com/cuda/nvrtc/index.html> (`nvrtcGetCUBIN`, supported options).

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use crate::cuda::{
    Context, CudaError, LoadedFunction, LoadedModule, Stream, nvrtc_compile, nvrtc_version,
};

const TEGRA_ARCHS: [u32; 5] = [32, 53, 62, 72, 87];

/// The highest arch NVRTC `version` targets, as CuPy 14.2.0 decides it.
pub fn nvrtc_max_arch(version: (i32, i32)) -> u32 {
    match version {
        (12, minor) if minor < 8 => 90,
        (12, 8) => 120,
        _ => 121,
    }
}

/// The NVRTC options for a device of compute capability `cc` under NVRTC `version`, in CuPy's
/// order; the output is a cubin for that arch.
pub fn cupy_options(cc: (i32, i32), version: (i32, i32)) -> Vec<String> {
    let device = (cc.0.max(0) as u32) * 10 + cc.1.max(0) as u32;
    let arch = if TEGRA_ARCHS.contains(&device) {
        device
    } else {
        device.min(nvrtc_max_arch(version))
    };
    vec![
        "-ftz=true".to_string(),
        format!("-arch=sm_{arch}"),
        "--device-as-default-execution-space".to_string(),
    ]
}

/// Why a kernel did not load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// NVRTC refused the source; the log says why.
    Compile {
        /// The program name given to [`Kernels::load`].
        name: String,
        /// NVRTC's compile log.
        log: String,
    },
    /// A driver or library call failed (no NVRTC, module load, missing kernel name).
    Cuda(CudaError),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::Compile { name, log } => write!(f, "{name} did not compile:\n{log}"),
            KernelError::Cuda(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for KernelError {}

impl From<CudaError> for KernelError {
    fn from(e: CudaError) -> Self {
        match e {
            CudaError::Nvrtc {
                call: "nvrtcCompileProgram",
                log,
                ..
            } => KernelError::Compile {
                name: String::new(),
                log,
            },
            other => KernelError::Cuda(other),
        }
    }
}

/// Compiles and caches the modules of one device.
pub struct Kernels {
    ctx: Arc<Context>,
    options: Vec<String>,
    cache: Mutex<HashMap<(String, String), Module>>,
}

impl fmt::Debug for Kernels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernels")
            .field("options", &self.options)
            .finish()
    }
}

impl Kernels {
    /// A loader for the context's device; reads the compute capability and NVRTC's version.
    ///
    /// # Errors
    ///
    /// [`KernelError::Cuda`] when the compute capability or NVRTC's version cannot be read (no
    /// driver, no NVRTC).
    pub fn new(ctx: &Arc<Context>) -> Result<Kernels, KernelError> {
        let cc = ctx.compute_capability()?;
        let version = nvrtc_version()?;
        let options = cupy_options(cc, version);
        Ok(Kernels {
            ctx: ctx.clone(),
            options,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// The options every program is compiled with.
    pub fn options(&self) -> &[String] {
        &self.options
    }

    /// The module compiled from `source` under the program name `name` (compiled once per
    /// name and source; later calls return the cached module).
    ///
    /// # Errors
    ///
    /// [`KernelError::Compile`] when NVRTC refuses `source`; [`KernelError::Cuda`] when the module
    /// cannot be loaded.
    pub fn load(&self, name: &str, source: &str) -> Result<Module, KernelError> {
        let key = (name.to_string(), source.to_string());
        // Loads happen at start; the lock is never taken on a photo's path.
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(m) = cache.get(&key) {
            return Ok(m.clone());
        }
        let cubin =
            nvrtc_compile(source, name, &self.options).map_err(|e| match KernelError::from(e) {
                KernelError::Compile { log, .. } => KernelError::Compile {
                    name: name.to_string(),
                    log,
                },
                other => other,
            })?;
        let module = Module {
            inner: Arc::new(LoadedModule::load(&self.ctx, &cubin)?),
        };
        cache.insert(key, module.clone());
        Ok(module)
    }
}

/// A loaded module; cheap to clone.
#[derive(Clone)]
pub struct Module {
    inner: Arc<LoadedModule>,
}

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Module").finish_non_exhaustive()
    }
}

impl Module {
    /// The kernel `name`; resolve once and keep it.
    ///
    /// # Errors
    ///
    /// [`KernelError::Cuda`] when the module has no kernel `name`.
    pub fn function(&self, name: &str) -> Result<Kernel, KernelError> {
        Ok(Kernel {
            inner: self.inner.function(name)?,
        })
    }

    /// Launch the kernel `name` on `stream` with `args` (resolves the name on each call; a hot
    /// path keeps a [`Kernel`] instead).
    ///
    /// The crate cannot check a kernel's parameter list: `args` must match it in number, order
    /// and type, and every address in it must name memory the kernel may touch. A kernel that
    /// writes outside its arguments corrupts device memory, which the driver reports as a sticky
    /// error that marks the context lost.
    ///
    /// # Errors
    ///
    /// [`KernelError::Cuda`] when `name` is not in the module or the driver refuses the launch.
    pub fn launch(
        &self,
        name: &str,
        cfg: &LaunchConfig,
        stream: &Stream,
        args: &[KernelArg],
    ) -> Result<(), KernelError> {
        self.function(name)?.launch(cfg, stream, args)
    }
}

/// One kernel of a loaded module (keeps the module loaded).
pub struct Kernel {
    inner: LoadedFunction,
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernel").finish_non_exhaustive()
    }
}

impl Kernel {
    /// Launch on `stream` (see [`Module::launch`] for what `args` must satisfy).
    ///
    /// # Errors
    ///
    /// [`KernelError::Cuda`] when the driver refuses the launch.
    pub fn launch(
        &self,
        cfg: &LaunchConfig,
        stream: &Stream,
        args: &[KernelArg],
    ) -> Result<(), KernelError> {
        Ok(self.inner.launch(cfg, stream, args)?)
    }
}

/// Grid and block dimensions and dynamic shared memory of one launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Blocks per grid in x, y and z.
    pub grid: (u32, u32, u32),
    /// Threads per block in x, y and z.
    pub block: (u32, u32, u32),
    /// Dynamic shared memory per block, in bytes.
    pub shared_bytes: u32,
}

/// One kernel parameter, passed by value. Device pointers are passed as `U64` addresses
/// ([`crate::cuda::DeviceRange::checked_addr`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelArg {
    /// A signed 32-bit value.
    I32(i32),
    /// An unsigned 32-bit value.
    U32(u32),
    /// A signed 64-bit value.
    I64(i64),
    /// An unsigned 64-bit value; a device pointer is passed this way.
    U64(u64),
    /// A 32-bit float.
    F32(f32),
    /// A 64-bit float.
    F64(f64),
}

impl KernelArg {
    pub(crate) fn as_param(&mut self) -> *mut c_void {
        match self {
            KernelArg::I32(v) => (v as *mut i32).cast(),
            KernelArg::U32(v) => (v as *mut u32).cast(),
            KernelArg::I64(v) => (v as *mut i64).cast(),
            KernelArg::U64(v) => (v as *mut u64).cast(),
            KernelArg::F32(v) => (v as *mut f32).cast(),
            KernelArg::F64(v) => (v as *mut f64).cast(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_option_list_is_cupys_in_cupys_order() {
        assert_eq!(
            cupy_options((8, 6), (13, 2)),
            [
                "-ftz=true",
                "-arch=sm_86",
                "--device-as-default-execution-space"
            ]
        );
    }

    #[test]
    fn the_arch_is_capped_by_what_nvrtc_targets() {
        let arch = |cc, v| cupy_options(cc, v)[1].clone();
        assert_eq!(arch((12, 0), (12, 4)), "-arch=sm_90");
        assert_eq!(arch((12, 0), (12, 8)), "-arch=sm_120");
        assert_eq!(arch((12, 1), (12, 8)), "-arch=sm_120");
        assert_eq!(arch((12, 1), (13, 0)), "-arch=sm_121");
        assert_eq!(arch((7, 5), (13, 2)), "-arch=sm_75");
        assert_eq!(arch((8, 7), (12, 4)), "-arch=sm_87", "a Tegra arch is kept");
        assert_eq!(
            nvrtc_max_arch((11, 8)),
            121,
            "versions CuPy does not name fall to its else branch"
        );
    }

    #[test]
    fn a_compile_failure_keeps_its_log() {
        let e = CudaError::Nvrtc {
            call: "nvrtcCompileProgram",
            code: 6,
            log: "error: x".into(),
        };
        assert_eq!(
            KernelError::from(e),
            KernelError::Compile {
                name: String::new(),
                log: "error: x".into()
            }
        );
        let e = CudaError::LibraryMissing {
            library: "NVRTC (libnvrtc)",
        };
        assert!(matches!(KernelError::from(e), KernelError::Cuda(_)));
    }
}
