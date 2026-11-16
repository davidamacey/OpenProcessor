//! A photo file on local disk, as the server and the frame engine both see it: the key rule
//! (plan 18 sections 4.3 and 4.4), the resolve of a key under a root that refuses every path
//! leaving it, and the content claim the server makes for a file (its size and its `imohash`).
//!
//! The crate has no dependency on the object store, the settings or the network, so the engine,
//! which reads local files only and never talks to S3 (plan 17 section 4), can use the one copy
//! of the rule the server's `op_storage` uses.
#![forbid(unsafe_code)]

pub mod claim;
pub mod file_ref;
pub mod imohash;
pub mod key;
pub mod resolve;

pub use claim::{ClaimMismatch, FileClaim};
pub use file_ref::FileRef;
pub use key::{KeyError, Root, validate_key};
pub use resolve::{CanonicalDirError, canonical_dir, resolve_under};
