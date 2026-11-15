//! `op-fe-loadgen`: the measuring client of the native engine's first phase (plan 17 section 5,
//! N0). One closed-loop client speaks protocol version 1 to any engine at an address, so both
//! engines are measured by the same code, and a second mode drives the models alone through
//! `gpu_frame`'s model-call interface. Every row it writes carries the evidence the paper's
//! tables are built from (plan 17 section 5, "Evidence for the paper").
//!
//! | Module | Concern |
//! |---|---|
//! | [`arith`] | percentiles, rates, the model server's statistics deltas |
//! | [`cgroup`] | the measured engine's CPU from its cgroup's `cpu.stat` |
//! | [`host`] | load, CPU model, clock |
//! | [`list`], [`plan`] | the photos and the plan, with their SHA-256 |
//! | [`exchange`] | one photo's round trip and the closed loop around it |
//! | [`client`], [`statistics`] | protocol version 1 as a client; the model server's counters |
//! | [`z_plan`], [`z_mode`] | the models alone: which calls a photo makes, and making them |

pub mod arith;
pub mod cgroup;
pub mod client;
pub mod commands;
pub mod exchange;
pub mod gpu;
pub mod host;
pub mod list;
pub mod measure;
pub mod plan;
pub mod proto;
pub mod row;
pub mod statistics;
pub mod z_mode;
pub mod z_plan;
