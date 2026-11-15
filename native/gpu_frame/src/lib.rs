//! The general part of the native frame engine: nothing here knows the plan, the wire, a branch
//! or a product kernel (plan 17 section 2.0).
//!
//! Here: [`cuda`] (context, streams, copies, errors with their fault class). Every `unsafe` block
//! is in `cuda.rs`.
//!
//! CUDA is loaded at run time; nothing is linked at build time, so the crate builds and its tests
//! run on a machine without a GPU.

pub mod cuda;

pub use cuda::{Context, CudaError, DeviceRange, Fault, PinnedBuf, Stream};
