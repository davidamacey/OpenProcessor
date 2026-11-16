//! The upload receiver, and the ingest pipeline with its commit. This crate holds the receiver's
//! bounds so far: the settings, the admission semaphores and the disk guard. The route that
//! streams a request to the landing zone comes next.
#![forbid(unsafe_code)]

pub mod admission;
pub mod disk_space;
pub mod settings;
