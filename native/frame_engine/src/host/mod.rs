//! The host route (plan 17 N1a): a photo nvJPEG does not take (`cpu_<reason>` of
//! `jpeg_header::route`) is decoded on the CPU of its decode lane and uploaded, then follows the
//! same branches. A port of `frame_engine/host_decode.py` over `src/utils/image_decode.py`:
//!
//! 1. [`admit`], before any budget is reserved (`FrameEngine._source`): the size from the header
//!    alone (`host_decode.probe`); past twice Pillow's `MAX_IMAGE_PIXELS` it is `huge` (Pillow's
//!    `DecompressionBombError`), a file no decoder here opens is `decoder`, and past
//!    `OP_FRAME_ENGINE_MAX_MEGAPIXELS` it is `huge`. The frame's need is
//!    [`host_need_bytes`]: the frame arrives upright, so no oriented copy is held next to it.
//! 2. [`decode_checked`], on the decode lane (`FrameEngine._decode_host`): the upright frame;
//!    a failed decode, or a pixel count other than the header's, is `decoder`.
//!
//! JPEG goes through libjpeg-turbo (`jpeg`), everything else through the `image` crate
//! (`formats`); both apply Pillow's mode conversions (`modes`) and its orientation rule
//! (`orientation`), the pixels moved by [`orient_host`], the host twin of `orient.cu`. The
//! profile of an `RGB` or `RGBA` image travels with the frame to the colour step, the one colour
//! path of both routes (plan 17 section 3, "Colour").

mod formats;
mod jpeg;
pub mod modes;
pub mod orientation;
mod png;

use crate::jpeg_header::{Kind, header_segments, jpeg_end, jpeg_header};
use crate::kernels::{orient_host, oriented_dims, reorients};

/// Pillow's `Image.MAX_IMAGE_PIXELS`; `Image.open` raises past twice this.
pub const MAX_IMAGE_PIXELS: u64 = 89_478_485;

const EXIF_TAG: &[u8] = b"Exif\0\0";
const XMP_TAG: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

/// A decoded frame before orientation: interleaved RGB and the profile Pillow would hand to the
/// colour step (an `RGB` or `RGBA` image's only).
struct Pixels {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    icc: Option<Vec<u8>>,
}

/// An upright frame decoded on the host.
#[derive(Debug)]
pub struct HostFrame {
    /// Interleaved RGB, `width * height * 3` bytes.
    pub rgb: Vec<u8>,
    /// Columns of the upright frame.
    pub width: u32,
    /// Rows of the upright frame.
    pub height: u32,
    /// The embedded profile of an `RGB` or `RGBA` image, for the colour step.
    pub icc: Option<Vec<u8>>,
}

/// A photo the host route admitted: its pixel count by the header and the bytes it reserves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Admitted {
    /// `(width, height)` of the upright frame.
    pub size: (u32, u32),
    /// Bytes of the frame against the budget (`prep_budget.host_need_bytes`).
    pub need: u64,
}

/// `prep_budget.host_need_bytes` without renditions: the one upright frame.
pub fn host_need_bytes(width: u32, height: u32) -> u64 {
    u64::from(width) * u64::from(height) * 3
}

/// Pillow's JPEG plugin takes a file that starts `FF D8 FF`.
fn is_jpeg(data: &[u8]) -> bool {
    data.starts_with(&[0xFF, 0xD8, 0xFF])
}

/// What Pillow's JPEG plugin keeps of the header segments: the `Exif` APP1 payloads joined (the
/// first with its tag), the last XMP packet.
fn jpeg_metadata(data: &[u8]) -> (Option<Vec<u8>>, Option<&[u8]>) {
    let mut exif: Option<Vec<u8>> = None;
    let mut xmp = None;
    for (marker, payload) in header_segments(data) {
        if marker != 0xE1 {
            continue;
        }
        if payload.starts_with(EXIF_TAG) {
            match &mut exif {
                Some(e) => e.extend_from_slice(&payload[EXIF_TAG.len()..]),
                None => exif = Some(payload.to_vec()),
            }
        } else if let Some(packet) = payload.strip_prefix(XMP_TAG) {
            xmp = Some(packet);
        }
    }
    (exif, xmp)
}

/// The stored size and the orientation the decode will apply. `host_decode.probe` reads the size
/// alone (the pixel count is what Python uses); the native flow cuts the round-1 inputs before
/// the decode, so it needs the upright size too, read by the same rule as [`decode`].
fn probe(data: &[u8]) -> Result<(u32, u32, u32), String> {
    let bad = |_| "bad EXIF block".to_owned();
    if is_jpeg(data) {
        let h = jpeg_header(data);
        if h.kind != Kind::Jpeg {
            return Err("no frame header".into());
        }
        let (exif, xmp) = jpeg_metadata(data);
        let tag = orientation::orientation(exif.as_deref(), xmp).map_err(bad)?;
        Ok((h.width, h.height, tag))
    } else if png::is_png(data) {
        let m = png::chunks(data)?;
        let tag = orientation::orientation(m.exif.as_deref(), m.xmp).map_err(bad)?;
        Ok((m.width, m.height, tag))
    } else {
        formats::probe(data)
    }
}

/// Step 1: the upright size and need of an admitted photo.
///
/// # Errors
///
/// The photo's status: `decoder` when no decoder here opens it, `huge` past the gates.
pub fn admit(data: &[u8], max_megapixels: f64) -> Result<Admitted, &'static str> {
    let (w, h, tag) = probe(data).map_err(|e| {
        tracing::debug!("host probe failed: {e}");
        "decoder"
    })?;
    let pixels = u64::from(w) * u64::from(h);
    if pixels > 2 * MAX_IMAGE_PIXELS {
        return Err("huge");
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "the Python gate divides a float"
    )]
    if pixels as f64 / 1e6 > max_megapixels {
        return Err("huge");
    }
    Ok(Admitted {
        size: oriented_dims(w, h, tag),
        need: host_need_bytes(w, h),
    })
}

/// `host_decode.decode`: the upright frame.
///
/// # Errors
///
/// Why there is none: the decoder's message (a damaged file, an unsupported format or mode).
pub fn decode(data: &[u8]) -> Result<HostFrame, String> {
    let (_, _, tag) = probe(data)?;
    let p = if is_jpeg(data) {
        let h = jpeg_header(data);
        let d = jpeg::decode(data, h.components, jpeg_end(data).is_some())?;
        Pixels {
            rgb: d.pixels,
            width: d.width,
            height: d.height,
            icc: if h.components == 3 { h.icc } else { None },
        }
    } else if png::is_png(data) {
        png::decode(data)?
    } else {
        formats::decode(data)?
    };
    let icc = p.icc;
    let (rgb, width, height) = if reorients(tag) {
        orient_host(&p.rgb, p.width, p.height, tag)
    } else {
        (p.rgb, p.width, p.height)
    };
    Ok(HostFrame {
        rgb,
        width,
        height,
        icc,
    })
}

/// Step 2: the frame decoded on the lane.
///
/// # Errors
///
/// `decoder` when it does not decode or its size disagrees with the probe's.
pub fn decode_checked(data: &[u8], size: (u32, u32)) -> Result<HostFrame, &'static str> {
    let frame = decode(data).map_err(|e| {
        tracing::debug!("host decode failed: {e}");
        "decoder"
    })?;
    if (frame.width, frame.height) != size {
        return Err("decoder");
    }
    Ok(frame)
}
