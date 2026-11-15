//! The general part of the native frame engine: nothing here knows the plan, the wire, a branch
//! or a product kernel (plan 17 section 2.0).
//!
//! | Plan 17 item | Here |
//! |---|---|
//! | `Lanes::decode` | [`nvjpeg::Lanes::decode`], [`nvjpeg::Lane::decode`] |
//! | `FramePool::alloc` | [`frame_pool::FramePool::alloc`] |
//! | `Admission::admit` | [`budget::Admission::admit`] |
//!
//! Under them: [`cuda`] (context, streams, copies, errors with their fault class) and [`pool`]
//! (OS-thread pools with panic isolation and cancellation at the hand-off). Every `unsafe` block is
//! in `cuda.rs` or `nvjpeg/ffi.rs`.
//!
//! CUDA and nvJPEG are loaded at run time; nothing is linked at build time, so the crate builds and
//! its tests run on a machine without a GPU.

pub mod budget;
pub mod cuda;
pub mod frame_pool;
pub mod nvjpeg;
pub mod pool;

pub use budget::{
    Admission, AdmitError, CapExceeded, Charge, DeviceCap, HostCount, OverloadCause, Permit,
};
pub use cuda::{Context, CudaError, DeviceRange, Fault, PinnedBuf, Stream};
pub use frame_pool::{AllocError, DeviceBuf, FramePool};
pub use nvjpeg::{Backend, DecodeError, Decoded, DeviceFrame, Lane, Lanes};
pub use pool::{CancelToken, JobError, JobHandle, PoolError, WorkerPool};
