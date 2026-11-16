//! `op-upload-loadgen`: sends photo uploads to the server the way the browser frontend does and
//! times them (plan 18 section 5.2, slice R0g of plan 18a). A closed loop keeps N requests in
//! flight over a photo list held in memory; each request is a browser-shaped multipart body
//! streamed from that memory, so the tool's own cost stays far below the rate it measures.
//!
//! | Module | Concern |
//! |---|---|
//! | [`pack`] | which photos go into which request (the two caps) |
//! | [`multipart`] | the browser-shaped body as a stream of frames, and the part stamps |
//! | [`send`] | one request on the shared connection pool, and how its answer is classified |
//! | [`record`] | what one request leaves behind, and the stamps file |
//! | [`summary`] | the row's arithmetic on those records, with no clock and no network |
//! | [`closed_loop`] | N requests in flight, the warm-up and the measured window |
//! | [`row`] | the JSON line (`op-upload-loadgen/1`) and its evidence |
//! | [`process`] | this process's own CPU, to show the tool was not the limit |
//! | [`measure`] | one measured row: the loop, the counters at the window's edges, the row built from both |
//! | [`commands`] | the command line and the wiring of the sources a row reads |
//!
//! The list, host and cgroup readers are `frame_engine_loadgen`'s, used as they are.

#![forbid(unsafe_code)]

pub mod closed_loop;
pub mod commands;
pub mod measure;
pub mod multipart;
pub mod pack;
pub mod process;
pub mod record;
pub mod row;
pub mod send;
pub mod summary;
