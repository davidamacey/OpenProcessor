//! The product part of the native frame engine: the KServe front of protocol version 1, the plan
//! vocabulary and its parser, settings, metrics, geometry and routing (plan 17 section 2.0).
//! Every CUDA or nvJPEG call goes through `gpu_frame`.
//!
//! The binary (`main.rs`) only wires these modules; they live in a library so that the contract
//! and golden tests can reach them.

pub mod admission;
pub mod branches;
pub mod cores;
pub mod engine;
pub mod fault;
pub mod front;
pub mod geometry;
pub mod health;
pub mod jpeg_header;
pub mod kernels;
pub mod metrics;
pub mod photo;
pub mod pipeline;
pub mod proto;
pub mod pyfmt;
pub mod settings;
pub mod stub;
pub mod wire;
