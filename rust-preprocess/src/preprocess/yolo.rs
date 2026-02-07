/// YOLO letterbox preprocessing.
///
/// Exactly matches `src/services/cpu_preprocess.py:84-169`:
/// 1. scale = min(640/h, 640/w).min(1.0) (no upscale)
/// 2. new_w = round(w * scale), new_h = round(h * scale)
/// 3. Resize with bilinear interpolation
/// 4. Center padding with value 114
/// 5. Ultralytics rounding trick for padding
/// 6. Normalize /255.0, HWC -> CHW

use super::decode::DecodedImage;

pub struct LetterboxResult {
    /// CHW tensor, FP32, normalized [0,1], shape [3, 640, 640]
    pub tensor: Vec<f32>,
    pub scale: f64,
    pub pad_w: f64,
    pub pad_h: f64,
    pub orig_w: u32,
    pub orig_h: u32,
}

pub fn letterbox(img: &DecodedImage, target_size: u32) -> LetterboxResult {
    let orig_h = img.height as f64;
    let orig_w = img.width as f64;
    let target = target_size as f64;

    // Step 1: Calculate scale (no upscale)
    let scale = (target / orig_h).min(target / orig_w).min(1.0);

    // Step 2: New dimensions (Python round())
    let new_w = (orig_w * scale).round() as u32;
    let new_h = (orig_h * scale).round() as u32;

    // Step 3: Resize with bilinear interpolation
    let resized = if scale != 1.0 {
        resize_bilinear(&img.data, img.width, img.height, new_w, new_h)
    } else {
        img.data.clone()
    };

    // Step 4-5: Calculate padding (Ultralytics rounding trick)
    let pad_w = (target - new_w as f64) / 2.0;
    let pad_h = (target - new_h as f64) / 2.0;
    let top = (pad_h - 0.1).round() as u32;
    let bottom = (pad_h + 0.1).round() as u32;
    let left = (pad_w - 0.1).round() as u32;
    let right = (pad_w + 0.1).round() as u32;

    // Step 6: Create padded image (114, 114, 114)
    let padded_w = (left + new_w + right) as usize;
    let padded_h = (top + new_h + bottom) as usize;

    // Initialize with pad value 114
    let mut padded = vec![114u8; padded_h * padded_w * 3];

    // Copy resized image into center
    for y in 0..new_h as usize {
        let src_offset = y * new_w as usize * 3;
        let dst_offset = ((y + top as usize) * padded_w + left as usize) * 3;
        let row_bytes = new_w as usize * 3;
        padded[dst_offset..dst_offset + row_bytes]
            .copy_from_slice(&resized[src_offset..src_offset + row_bytes]);
    }

    // Handle size mismatch (rounding may cause 1px difference)
    let final_data = if padded_w != target_size as usize || padded_h != target_size as usize {
        resize_bilinear(
            &padded,
            padded_w as u32,
            padded_h as u32,
            target_size,
            target_size,
        )
    } else {
        padded
    };

    // Step 7: Normalize and HWC -> CHW
    let ts = target_size as usize;
    let mut tensor = vec![0.0f32; 3 * ts * ts];

    for y in 0..ts {
        for x in 0..ts {
            let src_idx = (y * ts + x) * 3;
            tensor[0 * ts * ts + y * ts + x] = final_data[src_idx] as f32 / 255.0; // R
            tensor[1 * ts * ts + y * ts + x] = final_data[src_idx + 1] as f32 / 255.0; // G
            tensor[2 * ts * ts + y * ts + x] = final_data[src_idx + 2] as f32 / 255.0; // B
        }
    }

    LetterboxResult {
        tensor,
        scale,
        pad_w,
        pad_h,
        orig_w: img.width,
        orig_h: img.height,
    }
}

/// Simple bilinear resize for RGB images.
fn resize_bilinear(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    // Use fast_image_resize crate for SIMD-optimized bilinear resize
    use fast_image_resize as fr;

    let src_image = fr::images::Image::from_vec_u8(
        src_w,
        src_h,
        src.to_vec(),
        fr::PixelType::U8x3,
    )
    .expect("Failed to create source image");

    let mut dst_image = fr::images::Image::new(
        dst_w,
        dst_h,
        fr::PixelType::U8x3,
    );

    let mut resizer = fr::Resizer::new();
    resizer
        .resize(
            &src_image,
            &mut dst_image,
            &fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Interpolation(
                fr::FilterType::Bilinear,
            )),
        )
        .expect("Resize failed");

    dst_image.into_vec()
}
