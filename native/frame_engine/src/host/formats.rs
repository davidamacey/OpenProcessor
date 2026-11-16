//! The host route's formats other than JPEG and PNG, through the `image` crate 0.25.10, then Pillow's
//! mode conversion (`modes`) and orientation rule (`orientation`).
//!
//! Pillow identifies a file by its content, not its name, and so does
//! `ImageReader::with_guessed_format` (<https://docs.rs/image/0.25.10/image/struct.ImageReader.html>).
//! Only the formats in [`ACCEPTED`] are decoded: each one equals Pillow 13.0.0's pixels on the
//! corpus (`tests/golden_host.rs`). The size gate is the engine's own (`OP_FRAME_ENGINE_MAX_MEGAPIXELS`
//! and Pillow's bomb limit), so the crate's allocation limits are lifted.

use std::io::Cursor;

use image::{DynamicImage, ExtendedColorType, ImageDecoder, ImageFormat, ImageReader};

use super::orientation::{BadExif, orientation};
use super::{Pixels, modes};

/// The formats decoded here (others answer `decoder`; the corpus test lists which and why).
pub const ACCEPTED: [ImageFormat; 4] = [
    ImageFormat::Gif,
    ImageFormat::Bmp,
    ImageFormat::Tiff,
    ImageFormat::WebP,
];

type Reader<'a> = ImageReader<Cursor<&'a [u8]>>;

fn reader(data: &[u8]) -> Result<(Reader<'_>, ImageFormat), String> {
    let mut reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    reader.no_limits();
    match reader.format() {
        Some(f) if ACCEPTED.contains(&f) => Ok((reader, f)),
        _ => Err("cannot identify image file".into()),
    }
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// The orientation Pillow's `getexif()` gives: a TIFF's own directory, else the container's
/// EXIF block and XMP packet.
fn tag(decoder: &mut dyn ImageDecoder, format: ImageFormat) -> Result<u32, String> {
    if format == ImageFormat::Tiff {
        return Ok(decoder.orientation().map_err(err)?.to_exif().into());
    }
    let exif = decoder.exif_metadata().map_err(err)?;
    let xmp = decoder.xmp_metadata().map_err(err)?;
    orientation(exif.as_deref(), xmp.as_deref()).map_err(|BadExif| "bad EXIF block".to_owned())
}

/// The stored size from the header alone and the orientation.
pub fn probe(data: &[u8]) -> Result<(u32, u32, u32), String> {
    let (reader, format) = reader(data)?;
    let mut decoder = reader.into_decoder().map_err(err)?;
    let tag = tag(&mut decoder, format)?;
    let (w, h) = decoder.dimensions();
    Ok((w, h, tag))
}

/// Whether Pillow opens the image as `RGB` or `RGBA` (the modes whose profile it converts).
fn truecolour(original: ExtendedColorType) -> bool {
    matches!(
        original,
        ExtendedColorType::Rgb8
            | ExtendedColorType::Rgba8
            | ExtendedColorType::Rgb16
            | ExtendedColorType::Rgba16
    )
}

/// The frame and its profile.
pub fn decode(data: &[u8]) -> Result<Pixels, String> {
    let (reader, _) = reader(data)?;
    let mut decoder = reader.into_decoder().map_err(err)?;
    let original = decoder.original_color_type();
    if matches!(
        original,
        ExtendedColorType::Cmyk8 | ExtendedColorType::Cmyk16
    ) {
        return tiff_cmyk(data);
    }
    let icc = if truecolour(original) {
        decoder.icc_profile().map_err(err)?
    } else {
        None
    };
    let image = DynamicImage::from_decoder(decoder).map_err(err)?;
    let (width, height) = (image.width(), image.height());
    Ok(Pixels {
        rgb: to_rgb(image)?,
        width,
        height,
        icc,
    })
}

/// A CMYK TIFF: the `image` crate converts CMYK by its own rounding, so the ink samples are read
/// through the `tiff` crate it wraps (<https://docs.rs/tiff/0.11.3>) and converted as Pillow does
/// (16-bit samples by their high byte, raw mode `CMYK;16B`).
fn tiff_cmyk(data: &[u8]) -> Result<Pixels, String> {
    use tiff::decoder::{Decoder, DecodingResult, Limits};
    let mut decoder = Decoder::new(Cursor::new(data))
        .map_err(err)?
        .with_limits(Limits::unlimited());
    let (width, height) = decoder.dimensions().map_err(err)?;
    let cmyk: Vec<u8> = match decoder.read_image().map_err(err)? {
        DecodingResult::U8(v) => v,
        DecodingResult::U16(v) => v.iter().map(|s| s.to_be_bytes()[0]).collect(),
        _ => return Err("unsupported CMYK sample type".into()),
    };
    Ok(Pixels {
        rgb: modes::cmyk_to_rgb(&cmyk),
        width,
        height,
        icc: None,
    })
}

/// `Image.convert('RGB')` of what the decoder gave.
fn to_rgb(image: DynamicImage) -> Result<Vec<u8>, String> {
    Ok(match image {
        DynamicImage::ImageLuma8(b) => modes::grey_to_rgb(b.as_raw(), 1),
        DynamicImage::ImageLumaA8(b) => modes::grey_to_rgb(b.as_raw(), 2),
        DynamicImage::ImageRgb8(b) => b.into_raw(),
        DynamicImage::ImageRgba8(b) => modes::colour_to_rgb(b.as_raw(), 4),
        DynamicImage::ImageLuma16(b) => modes::grey16_to_rgb(b.as_raw()),
        DynamicImage::ImageLumaA16(b) => modes::high_bytes_to_rgb(b.as_raw(), 2, true),
        DynamicImage::ImageRgb16(b) => modes::high_bytes_to_rgb(b.as_raw(), 3, false),
        DynamicImage::ImageRgba16(b) => modes::high_bytes_to_rgb(b.as_raw(), 4, false),
        other => return Err(format!("unsupported sample type {:?}", other.color())),
    })
}
