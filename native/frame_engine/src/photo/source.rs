//! Where a photo's frame comes from, decided before any budget is reserved
//! (`FrameEngine._source`): the GPU route (nvJPEG, then the orientation on the card) or the host
//! route (`crate::host`: decoded upright on the decode lane's CPU and uploaded), with the bytes the
//! frame holds and the `decode_route` the answer reports.

use bytes::Bytes;

use super::budget;
use crate::host;
use crate::jpeg_header::{Decoder, Route, oriented_size};

/// How the decode lane makes the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// nvJPEG, then the EXIF tag applied on the card.
    Gpu {
        /// The header's EXIF orientation.
        orientation: u32,
    },
    /// Decoded upright on the host (`host::decode_checked`), then copied to the card.
    Host,
}

/// One photo's decode, as decided from its header.
#[derive(Debug)]
pub struct Source {
    /// What the lane reads: the JPEG up to its end marker, or the whole file for the host route
    /// (the API's host decoders read the whole file too).
    pub image: Bytes,
    /// Which decoder makes the frame.
    pub origin: Origin,
    /// `(width, height)` of the upright frame; a decode of another size is refused.
    pub size: (u32, u32),
    /// Bytes against `OP_FRAME_ENGINE_BUDGET_MIB` that decide `over_budget` (Python's need).
    pub need: u64,
    /// Bytes reserved: the need, or the frame and its resample scratch when that is more.
    pub reserve: u64,
    /// `gpu` or `cpu_<reason>` (`src/domain/decode_route.py`).
    pub decode_route: String,
}

/// The source of `image` routed as `decided`, or the status that refuses it before any decode
/// (`corrupt` and `huge` from the route; `huge` and `decoder` from the host probe).
///
/// # Errors
///
/// The photo's status when it is refused.
pub fn decide(
    image: &Bytes,
    decided: Route,
    max_megapixels: f64,
    widths: &[u32],
) -> Result<Source, &'static str> {
    match decided.decoder {
        Decoder::Refused(status) => Err(status),
        Decoder::Gpu => {
            let h = decided.header;
            Ok(Source {
                image: image.slice(..h.end.min(image.len())),
                origin: Origin::Gpu {
                    orientation: h.orientation,
                },
                size: oriented_size(&h),
                need: budget::need_bytes(&h),
                reserve: budget::reserve_bytes(&h, widths),
                decode_route: "gpu".into(),
            })
        }
        Decoder::Cpu(reason) => {
            let admitted = host::admit(image, max_megapixels)?;
            Ok(Source {
                image: image.clone(),
                origin: Origin::Host,
                size: admitted.size,
                need: admitted.need,
                reserve: budget::host_reserve_bytes(admitted.size, widths),
                decode_route: format!("cpu_{reason}"),
            })
        }
    }
}
