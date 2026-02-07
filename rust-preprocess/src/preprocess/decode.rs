/// Image decoding: turbojpeg for JPEG, image crate for PNG/WebP/BMP fallback.

pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    /// RGB pixel data, row-major, 3 bytes per pixel
    pub data: Vec<u8>,
}

/// Decode image bytes into RGB pixel data.
pub fn decode_image(data: &[u8]) -> Result<DecodedImage, String> {
    // Check JPEG magic bytes
    if data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8 {
        decode_jpeg(data)
    } else {
        decode_fallback(data)
    }
}

fn decode_jpeg(data: &[u8]) -> Result<DecodedImage, String> {
    let mut decompressor = turbojpeg::Decompressor::new().map_err(|e| format!("turbojpeg init: {e}"))?;
    let header = decompressor
        .read_header(data)
        .map_err(|e| format!("JPEG header: {e}"))?;

    let width = header.width;
    let height = header.height;
    let pitch = width * 3; // RGB, 3 bytes per pixel

    let mut pixels = vec![0u8; (pitch * height) as usize];
    let image = turbojpeg::Image {
        pixels: pixels.as_mut_slice(),
        width: width as usize,
        pitch: pitch as usize,
        height: height as usize,
        format: turbojpeg::PixelFormat::RGB,
    };

    decompressor
        .decompress(data, image)
        .map_err(|e| format!("JPEG decode: {e}"))?;

    Ok(DecodedImage {
        width: width as u32,
        height: height as u32,
        data: pixels,
    })
}

fn decode_fallback(data: &[u8]) -> Result<DecodedImage, String> {
    let img = image::load_from_memory(data).map_err(|e| format!("Image decode: {e}"))?;
    let rgb = img.to_rgb8();
    let width = rgb.width();
    let height = rgb.height();
    Ok(DecodedImage {
        width,
        height,
        data: rgb.into_raw(),
    })
}
