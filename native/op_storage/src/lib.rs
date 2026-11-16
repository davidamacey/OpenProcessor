//! The storage layer: key layout, landing zone, local read paths, streaming SHA-256, the durable
//! object store (a local directory or S3 through one `object_store` interface), derived files with
//! a local cache, delete and erase primitives, and the reconcile listing.
#![forbid(unsafe_code)]

pub mod durable;
pub mod erase;
pub mod error;
pub mod hash;
pub mod keys;
pub mod landing;
pub mod local;
pub mod reconcile;
