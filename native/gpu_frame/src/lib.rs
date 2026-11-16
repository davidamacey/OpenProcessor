//! The general part of the native frame engine: nothing here knows the plan, the wire, a branch
//! or a product kernel (plan 17 section 2.0).
//!
//! | Plan 17 item | Here |
//! |---|---|
//! | `Lanes::decode` | [`nvjpeg::Lanes::decode`], [`nvjpeg::Lane::decode`] |
//! | `FramePool::alloc` | [`frame_pool::FramePool::alloc`] |
//! | `Admission::admit` | [`budget::Admission::admit`] |
//! | `Region::slice` | [`region::Region::slice`], [`region::Region::slice_blocking`] |
//! | `Kernels::load`, `Module::launch` | [`kernels::Kernels::load`], [`kernels::Module::launch`] |
//! | `ModelCalls::infer` | [`model_calls::ModelCalls::infer`], filled by [`model_server::ModelServer`] |
//!
//! Under them: [`cuda`] (context, streams, copies, errors with their fault class), [`pool`]
//! (OS-thread pools with panic isolation and cancellation at the hand-off) and [`slab`] (first fit,
//! coalescing). Every `unsafe` block is in `cuda.rs`, `nvjpeg/ffi.rs` or
//! `region.rs`.
//!
//! CUDA, NVRTC and nvJPEG are loaded at run time; nothing is linked at build time, so the crate
//! builds and its tests run on a machine without a GPU.

pub mod budget;
pub mod cuda;
pub mod frame_pool;
pub mod kernels;
pub mod model_calls;
pub mod model_server;
pub mod nvjpeg;
pub mod pool;
pub mod region;
pub mod slab;

pub use budget::{
    Admission, AdmitError, CapExceeded, Charge, DeviceCap, HostCount, OverloadCause, Permit,
};
pub use cuda::{Context, CudaError, DeviceRange, Fault, PinnedBuf, Stream, device_identity};
pub use frame_pool::{AllocError, DeviceBuf, FramePool};
pub use kernels::{Kernel, KernelArg, KernelError, Kernels, LaunchConfig, Module};
pub use model_calls::{CallError, InferCall, InputTensor, ModelCalls, OutputTensor};
pub use model_server::{ModelServer, ModelServerConfig};
pub use nvjpeg::{Backend, DecodeError, Decoded, DeviceFrame, Lane, Lanes};
pub use pool::{CancelToken, JobError, JobHandle, PoolError, WorkerPool};
pub use region::{Region, Slice};
pub use slab::SliceError;
