/// CLIP center-crop preprocessing.
///
/// Exactly matches `src/services/cpu_preprocess.py:172-209`:
/// 1. scale = 256.0 / min(h, w)
/// 2. new_w = int(w * scale) -- Python int() truncates
/// 3. new_h = int(h * scale)
/// 4. Resize with bilinear interpolation
/// 5. Center crop 256x256
/// 6. Normalize /255.0, HWC -> CHW

use super::decode::DecodedImage;

/// Returns CHW tensor, FP32, normalized [0,1], shape [3, 256, 256]
pub fn center_crop(img: &DecodedImage, target_size: u32) -> Vec<f32> {
    let orig_h = img.height as f64;
    let orig_w = img.width as f64;
    let target = target_size as f64;

    // Step 1: Scale shortest edge to target_size
    let scale = target / orig_h.min(orig_w);

    // Step 2: Python int() truncates (floor for positive numbers)
    let new_w = (orig_w * scale) as u32;
    let new_h = (orig_h * scale) as u32;

    // Step 3: Resize with bilinear
    let resized = resize_bilinear(&img.data, img.width, img.height, new_w, new_h);

    // Step 4: Center crop
    let start_x = ((new_w - target_size) / 2) as usize;
    let start_y = ((new_h - target_size) / 2) as usize;
    let ts = target_size as usize;

    // Step 5: Normalize and HWC -> CHW
    let mut tensor = vec![0.0f32; 3 * ts * ts];

    for y in 0..ts {
        for x in 0..ts {
            let src_idx = ((start_y + y) * new_w as usize + (start_x + x)) * 3;
            tensor[0 * ts * ts + y * ts + x] = resized[src_idx] as f32 / 255.0; // R
            tensor[1 * ts * ts + y * ts + x] = resized[src_idx + 1] as f32 / 255.0; // G
            tensor[2 * ts * ts + y * ts + x] = resized[src_idx + 2] as f32 / 255.0; // B
        }
    }

    tensor
}

/// Bilinear resize for RGB images using fast_image_resize.
fn resize_bilinear(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
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
