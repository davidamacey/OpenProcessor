//! `crops: inline` (`frame_engine/branches/crops.py`): each detection's rectangle of the frame as
//! a JPEG, encoded on the finish pool after the round-2 calls are out, so the copies and the
//! encodes overlap the calls in flight. Not a `branch_status` entry: a photo whose detector
//! failed answers no `crop_jpegs`; one with no detections an empty list.
//!
//! The Python engine encodes with OpenCV (`cv2.imencode` of the BGR pixels at
//! `OP_FRAME_ENGINE_JPEG_QUALITY`), which is libjpeg-turbo with libjpeg's defaults: 4:2:0
//! subsampling, Huffman tables not optimised, the accurate integer DCT, baseline. The TurboJPEG
//! API of the same library gives those settings by default with `TJPARAM_SUBSAMP` set to 4:2:0
//! (<https://libjpeg-turbo.org/Documentation/Documentation>, TurboJPEG 3 API: `TJPARAM_QUALITY`,
//! `TJPARAM_SUBSAMP`, `TJPARAM_FASTDCT` default 0, `TJPARAM_OPTIMIZE` default 0); RGB input is
//! converted to YCbCr by the same arithmetic as OpenCV's BGR input. A crop whose long side passes
//! `OP_FRAME_ENGINE_CROP_MAX_SIDE` is first shrunk with OpenCV's area filter ([`super::area`]).

use turbojpeg::{Compressor, Image, PixelFormat, Subsamp};

/// One crop's JPEG and the seconds its copy and encode took on their finish thread.
#[derive(Debug, Clone, PartialEq)]
pub struct CropJpeg {
    /// The encoded JPEG file.
    pub data: bytes::Bytes,
    /// Seconds the finish thread spent copying the rectangle out of the frame.
    pub copy_seconds: f64,
    /// Seconds the finish thread spent in the JPEG encode.
    pub encode_seconds: f64,
}

/// A finish thread's encoder at the engine's quality.
///
/// # Errors
///
/// The TurboJPEG compressor cannot be created or `quality` is not accepted by it.
pub fn encoder(quality: u32) -> Result<Compressor, String> {
    let mut c = Compressor::new().map_err(|e| e.to_string())?;
    c.set_quality(i32::try_from(quality).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    c.set_subsamp(Subsamp::Sub2x2).map_err(|e| e.to_string())?;
    Ok(c)
}

/// The JPEG of `width x height` interleaved RGB pixels, rows packed.
///
/// # Errors
///
/// The TurboJPEG encode fails.
pub fn encode(
    c: &mut Compressor,
    pixels: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, String> {
    let image = Image {
        pixels,
        width,
        pitch: width * 3,
        height,
        format: PixelFormat::RGB,
    };
    c.compress_to_vec(image)
        .map_err(|e| format!("crop JPEG encode failed: {e}"))
}

/// The JPEG of a crop's `width x height` RGB pixels, its long side first capped at `max_side`
/// (0: no cap) with the area filter, never enlarged (`CropJpegs._encode`).
///
/// # Errors
///
/// The resize's refusal ([`super::area::resize_area`]) or the encoder's failure.
pub fn encode_capped(
    c: &mut Compressor,
    pixels: &[u8],
    (width, height): (usize, usize),
    max_side: u32,
) -> Result<Vec<u8>, String> {
    match super::area::capped_size(width, height, max_side) {
        Some((w, h)) => {
            let shrunk = super::area::resize_area(pixels, (width, height), (w, h))?;
            encode(c, &shrunk, w, h)
        }
        None => encode(c, pixels, width, height),
    }
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    /// The test pattern both engines encode: `((7x + 3y), (x^2 + 11y), (5xy + 13))` mod 256.
    fn pattern(w: usize, h: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                out.extend(
                    [
                        (x * 7 + y * 3) % 256,
                        (x * x + y * 11) % 256,
                        (x * y * 5 + 13) % 256,
                    ]
                    .map(|v| v as u8),
                );
            }
        }
        out
    }

    #[test]
    fn the_crop_jpeg_is_byte_for_byte_opencvs() {
        // cv2.imencode('.jpg', rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q]) of the same 37 x 23
        // pattern, OpenCV 5.0.0 (libjpeg-turbo 3.1.2), as the Python engine's crops branch calls it
        for (quality, len, digest) in [
            (
                90,
                1514,
                "f10ef287c74ce9ca134884500d49d502bca9e1988f27065ff32c456954ec61d9",
            ),
            (
                75,
                1181,
                "2df63df012e747b093ebab63d8a530c65fcf9b34472f576e86da0175dfe4d0fb",
            ),
        ] {
            let mut c = encoder(quality).expect("encoder");
            let jpeg = encode(&mut c, &pattern(37, 23), 37, 23).expect("encode");
            let hex: String = Sha256::digest(&jpeg)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(
                (jpeg.len(), hex.as_str()),
                (len, digest),
                "quality {quality}"
            );
        }
    }

    #[test]
    fn a_crop_decodes_to_its_size_or_to_the_cap() {
        let mut c = encoder(90).expect("encoder");
        for (max_side, size) in [(0, (5, 9)), (9, (5, 9)), (6, (3, 6))] {
            let jpeg = encode_capped(&mut c, &pattern(5, 9), (5, 9), max_side).expect("encode");
            let image = turbojpeg::decompress(&jpeg, PixelFormat::RGB).expect("decode");
            assert_eq!((image.width, image.height), size, "cap {max_side}");
        }
    }
}
