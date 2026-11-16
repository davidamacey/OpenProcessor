//! PNG on the host route, read as Pillow 13.0.0's `PngImagePlugin` reads it.
//!
//! The pixels come from the `png` crate 0.18.1 (`Transformations::EXPAND`: palettes to RGB,
//! grey below 8 bits scaled to 8, `tRNS` to alpha; <https://docs.rs/png/0.18.1>). Pillow's rules
//! around them are kept here:
//!
//! * Checksums: Pillow checks the CRC of every chunk before the first `IDAT` ("broken PNG file")
//!   and of none after it, so the crate's own check is off (`set_ignore_crc`) and [`chunks`]
//!   checks the ones before the image data; zlib checks the data's Adler-32, so that check is on.
//! * Metadata: `getexif()` reads `info` as opened (chunks before `IDAT`) when that holds an
//!   `eXIf`; otherwise it loads the image, and the chunks after the data join `info`, a later
//!   chunk replacing an earlier one. The XMP packet is the `iTXt` chunk `XML:com.adobe.xmp`
//!   (uncompressed here; a compressed packet is not read).
//! * Modes: 16-bit grey (`I;16`, also with a `tRNS` level) is clipped; other 16-bit samples keep
//!   their high byte; alpha is dropped (`modes`).

use std::io::Cursor;

use super::{Pixels, modes};

const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
const XMP_KEY: &[u8] = b"XML:com.adobe.xmp";

/// What Pillow's open and `getexif()` read of the chunks.
#[derive(Debug, Default)]
pub struct Meta<'a> {
    pub width: u32,
    pub height: u32,
    pub exif: Option<Vec<u8>>,
    pub xmp: Option<&'a [u8]>,
}

/// Whether `data` is a PNG (Pillow's `_accept`).
pub fn is_png(data: &[u8]) -> bool {
    data.starts_with(SIGNATURE)
}

fn be32(data: &[u8], at: usize) -> Option<usize> {
    let b = data.get(at..at.checked_add(4)?)?;
    usize::try_from(u32::from_be_bytes([b[0], b[1], b[2], b[3]])).ok()
}

/// The text of an uncompressed `iTXt` chunk with the XMP keyword.
fn itxt_xmp(payload: &[u8]) -> Option<&[u8]> {
    let rest = payload.strip_prefix(XMP_KEY)?.strip_prefix(b"\0")?;
    let (&flag, rest) = rest.split_first()?;
    if flag != 0 {
        return None;
    }
    let rest = rest.get(1..)?;
    let after_lang = &rest[memchr::memchr(0, rest)? + 1..];
    Some(&after_lang[memchr::memchr(0, after_lang)? + 1..])
}

/// Metadata found on one side of the image data.
#[derive(Default)]
struct Side<'a> {
    exif: Option<&'a [u8]>,
    xmp: Option<&'a [u8]>,
}

impl<'a> Side<'a> {
    fn note(&mut self, kind: &[u8], payload: &'a [u8]) {
        if kind == b"eXIf" {
            self.exif = Some(payload);
        } else if kind == b"iTXt"
            && let Some(x) = itxt_xmp(payload)
        {
            self.xmp = Some(x);
        }
    }
}

/// Walk every chunk: the header and checksums before the data strictly, the rest leniently.
pub fn chunks(data: &[u8]) -> Result<Meta<'_>, String> {
    if !is_png(data) {
        return Err("not a PNG file".into());
    }
    let (mut at, mut seen_data) = (SIGNATURE.len(), false);
    let (mut before, mut after) = (Side::default(), Side::default());
    let mut size = None;
    while let (Some(len), Some(kind)) = (be32(data, at), data.get(at + 4..at + 8)) {
        let payload = data.get(at + 8..at + 8 + len);
        seen_data |= kind == b"IDAT";
        if !seen_data {
            let payload = payload.ok_or("truncated PNG file")?;
            let crc = be32(data, at + 8 + len).ok_or("truncated PNG file")?;
            if crc32fast::hash(&data[at + 4..at + 8 + len]) as usize != crc {
                return Err("broken PNG file (bad header checksum)".into());
            }
            if size.is_none() {
                if kind != b"IHDR" || len < 13 {
                    return Err("no IHDR chunk".into());
                }
                size = Some((be32(payload, 0).unwrap_or(0), be32(payload, 4).unwrap_or(0)));
            }
            before.note(kind, payload);
        } else if let Some(payload) = payload {
            after.note(kind, payload);
        }
        if kind == b"IEND" || payload.is_none() {
            break;
        }
        at += 12 + len;
    }
    let (w, h) = size.ok_or("no IHDR chunk")?;
    let (exif, xmp) = if before.exif.is_some() {
        (before.exif, before.xmp)
    } else {
        (after.exif, after.xmp.or(before.xmp))
    };
    Ok(Meta {
        width: u32::try_from(w).map_err(|e| e.to_string())?,
        height: u32::try_from(h).map_err(|e| e.to_string())?,
        exif: exif.map(|e| [b"Exif\0\0".as_slice(), e].concat()),
        xmp,
    })
}

/// The crate's reader past the chunks before the image data, with Pillow's checksum policy.
fn reader(data: &[u8]) -> Result<png::Reader<Cursor<&[u8]>>, String> {
    let mut options = png::DecodeOptions::default();
    options.set_ignore_crc(true);
    // zlib checks the stream's Adler-32 for Pillow; the crate skips it by default
    options.set_ignore_adler32(false);
    let mut decoder = png::Decoder::new_with_options(Cursor::new(data), options);
    // the engine's megapixel gate bounds the frame, not the crate's 64 MiB default
    decoder.set_limits(png::Limits { bytes: usize::MAX });
    decoder.set_transformations(png::Transformations::EXPAND);
    decoder.read_info().map_err(|e| e.to_string())
}

/// The profile of an `RGB` or `RGBA` image (`iCCP`, read before the image data as Pillow's open
/// reads it).
pub fn profile(data: &[u8]) -> Result<Option<Vec<u8>>, String> {
    let reader = reader(data)?;
    let info = reader.info();
    Ok(
        matches!(info.color_type, png::ColorType::Rgb | png::ColorType::Rgba)
            .then(|| info.icc_profile.as_ref().map(|p| p.to_vec()))
            .flatten(),
    )
}

/// The frame as Pillow converts it to RGB.
pub fn decode(data: &[u8]) -> Result<Pixels, String> {
    let mut reader = reader(data)?;
    let source = reader.info().color_type;
    let size = reader.output_buffer_size().ok_or("frame too large")?;
    let mut buf = vec![0u8; size];
    let frame = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    buf.truncate(frame.buffer_size());
    let rgb = to_rgb(&buf, frame.color_type, frame.bit_depth, source)?;
    Ok(Pixels {
        rgb,
        width: frame.width,
        height: frame.height,
    })
}

fn to_rgb(
    buf: &[u8],
    colour: png::ColorType,
    depth: png::BitDepth,
    source: png::ColorType,
) -> Result<Vec<u8>, String> {
    use png::ColorType as C;
    if depth == png::BitDepth::Eight {
        return Ok(match colour {
            C::Grayscale => modes::grey_to_rgb(buf, 1),
            C::GrayscaleAlpha => modes::grey_to_rgb(buf, 2),
            C::Rgb => buf.to_vec(),
            C::Rgba => modes::colour_to_rgb(buf, 4),
            C::Indexed => return Err("palette not expanded".into()),
        });
    }
    if depth != png::BitDepth::Sixteen {
        return Err(format!("unexpected depth {depth:?} after expansion"));
    }
    let wide: Vec<u16> = buf
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_be_bytes(*b))
        .collect();
    Ok(match (colour, source) {
        (C::Grayscale, _) => modes::grey16_to_rgb(&wide),
        (C::GrayscaleAlpha, C::Grayscale) => {
            let grey: Vec<u16> = wide.iter().step_by(2).copied().collect();
            modes::grey16_to_rgb(&grey)
        }
        (C::GrayscaleAlpha, _) => modes::high_bytes_to_rgb(&wide, 2, true),
        (C::Rgb, _) => modes::high_bytes_to_rgb(&wide, 3, false),
        (C::Rgba, _) => modes::high_bytes_to_rgb(&wide, 4, false),
        (C::Indexed, _) => return Err("palette not expanded".into()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const P3: &[u8] = include_bytes!("../../../../tests/fixtures/icc/DisplayP3-v4.icc");

    fn tagged(colour: png::ColorType) -> Vec<u8> {
        let mut info = png::Info::with_size(2, 1);
        info.color_type = colour;
        info.icc_profile = Some(P3.into());
        let mut out = Vec::new();
        let mut writer = png::Encoder::with_info(&mut out, info)
            .expect("encoder")
            .write_header()
            .expect("header");
        let channels = if colour == png::ColorType::Rgb { 3 } else { 1 };
        writer
            .write_image_data(&vec![0; 2 * channels])
            .expect("data");
        writer.finish().expect("finish");
        out
    }

    #[test]
    fn the_profile_is_read_before_the_data_for_colour_images_only() {
        assert_eq!(
            profile(&tagged(png::ColorType::Rgb)).expect("png"),
            Some(P3.to_vec())
        );
        assert_eq!(
            profile(&tagged(png::ColorType::Grayscale)).expect("png"),
            None
        );
        let admitted = crate::host::admit(&tagged(png::ColorType::Rgb), 100.0).expect("admitted");
        assert_eq!(admitted.icc.as_deref(), Some(P3));
    }
}
