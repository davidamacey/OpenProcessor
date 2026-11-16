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
    /// The embedded profile the colour step reads (`JpegHeader::icc`; on the host route an `RGB`
    /// or `RGBA` image's, from `host::admit`).
    pub icc: Option<Vec<u8>>,
    /// Colour components of the frame the profile describes (`JpegHeader::components`; 3 on the
    /// host route, whose profile is a colour image's only).
    pub components: u32,
}

/// The source of `image` routed as `decided`, or the status that refuses it before any decode
/// (`corrupt` and `huge` from the route; `huge` and `decoder` from the host probe). `sides` are
/// the plan's rendition sides; `widths` gives the output widths the model inputs of an upright
/// frame of that size write, for the resample scratch (the renditions' are added here).
///
/// # Errors
///
/// The photo's status when it is refused.
pub fn decide(
    image: &Bytes,
    decided: Route,
    max_megapixels: f64,
    sides: &[u32],
    widths: impl Fn((u32, u32)) -> Vec<u32>,
) -> Result<Source, &'static str> {
    let written = |(w, h): (u32, u32)| {
        let mut out = widths((w, h));
        out.extend(budget::rendition_widths(w, h, sides));
        out
    };
    match decided.decoder {
        Decoder::Refused(status) => Err(status),
        Decoder::Gpu => {
            let h = decided.header;
            let size = oriented_size(&h);
            Ok(Source {
                image: image.slice(..h.end.min(image.len())),
                origin: Origin::Gpu {
                    orientation: h.orientation,
                },
                size,
                need: budget::need_bytes(&h, sides),
                reserve: budget::reserve_bytes(&h, sides, &written(size)),
                decode_route: "gpu".into(),
                icc: h.icc,
                components: h.components,
            })
        }
        Decoder::Cpu(reason) => {
            let admitted = host::admit(image, max_megapixels)?;
            let need = budget::host_need_bytes(admitted.stored, sides);
            Ok(Source {
                image: image.clone(),
                origin: Origin::Host,
                size: admitted.size,
                need,
                reserve: budget::host_reserve_bytes(need, admitted.size, &written(admitted.size)),
                decode_route: format!("cpu_{reason}"),
                icc: admitted.icc,
                components: 3,
            })
        }
    }
}
