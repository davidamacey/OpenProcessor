//! The upload receiver, and the ingest pipeline with its commit. This slice holds the receiver:
//! `POST /curation/projects/{project}/ingest/upload` streams every photo of a request to the
//! landing zone and hands it on to a [`upload::PartSink`]. The pipeline and its commit come with
//! the slice that implements the sink.
#![forbid(unsafe_code)]

pub mod admission;
pub mod disk_space;
pub mod settings;
pub mod upload;
