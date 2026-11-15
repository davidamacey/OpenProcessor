//! Protocol version 1 as the Python engine's wire module defines it
//! (`src/domain/frame_engine_wire.py`): the plan and its parser, the BYTES packing, the
//! `branch_status` text, and the generated contract the rest is read from.
//!
//! The contract (`contracts/json/frame_engine_v1.json`, rendered from the Python module) is the
//! one copy of the tensor names, datatypes and shapes, the branch outputs and the vocabularies;
//! this crate embeds it at build time. What the contract does not carry (the numeric ranges of
//! the plan fields) is held to the Python parser by the generated plan cases.

pub mod branch_status;
pub mod bytes_tensor;
pub mod contract;
pub mod json;
pub mod plan;
pub mod request;

pub use contract::contract;
pub use plan::{EnginePlan, PlanError, parse_plan};
