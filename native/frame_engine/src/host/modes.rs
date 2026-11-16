//! Pillow 13.0.0's `Image.convert('RGB')` for the modes the host route meets (the API calls it on
//! every frame that is not `RGB`, `src/utils/image_decode.py`), on the samples a decoder gave.
//!
//! Pillow's rules, which the `image` crate's own conversions do not follow (it scales 16-bit
//! samples and composites alpha by its own arithmetic), checked against Pillow's output by the
//! corpus test:
//! * `L`, `LA`: the grey level in all three channels; alpha is dropped, not composited.
//! * `RGBA`: alpha dropped.
//! * `I;16` (16-bit grey): clipped to 255, not scaled (`convert` goes through `I`).
//! * 16-bit `RGB`, `RGBA`, `LA` (raw modes `RGB;16B` and the like): the high byte.
//! * `CMYK`: `nk - MULDIV255(c, nk)` with `nk = 255 - k`, per channel (`cmyk2rgb` in
//!   `src/libImaging/Convert.c`), which is `((255 - c) * (255 - k) + 127) / 255` rounded as
//!   Pillow's `MULDIV255` rounds.

/// Pillow's `MULDIV255(a, b)`: `a * b / 255` rounded to nearest.
fn muldiv255(a: u32, b: u32) -> u32 {
    let t = a * b + 128;
    ((t >> 8) + t) >> 8
}

/// `CMYK` samples (four per pixel, not inverted) to `RGB`.
pub fn cmyk_to_rgb(cmyk: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(cmyk.len() / 4 * 3);
    for px in cmyk.as_chunks::<4>().0 {
        let nk = 255 - u32::from(px[3]);
        for &c in &px[..3] {
            out.push(u8::try_from(nk - muldiv255(u32::from(c), nk)).unwrap_or(u8::MAX));
        }
    }
    out
}

/// Grey (`L`, or the grey of `LA` when `stride` is 2) to `RGB`.
pub fn grey_to_rgb(samples: &[u8], stride: usize) -> Vec<u8> {
    samples
        .chunks_exact(stride)
        .flat_map(|px| [px[0]; 3])
        .collect()
}

/// `RGB` or `RGBA` (when `stride` is 4) to `RGB`.
pub fn colour_to_rgb(samples: &[u8], stride: usize) -> Vec<u8> {
    samples
        .chunks_exact(stride)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect()
}

/// `I;16` to `RGB`: each level clipped to 255.
pub fn grey16_to_rgb(samples: &[u16]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|&v| [u8::try_from(v).unwrap_or(u8::MAX); 3])
        .collect()
}

/// 16-bit samples with `stride` per pixel, of which the first `colours` are kept (3 for colour,
/// 1 for grey, replicated): the high byte of each.
pub fn high_bytes_to_rgb(samples: &[u16], stride: usize, grey: bool) -> Vec<u8> {
    let high = |v: u16| v.to_be_bytes()[0];
    samples
        .chunks_exact(stride)
        .flat_map(|px| {
            if grey {
                [high(px[0]); 3]
            } else {
                [high(px[0]), high(px[1]), high(px[2])]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The values Pillow 13.0.0 gave for `Image.fromarray(a, 'CMYK').convert('RGB')` (the corpus
    /// generator's check in `scripts/codegen/frame_engine_host_corpus.py` derives the formula).
    #[test]
    fn cmyk_follows_pillows_rounding() {
        let cases = [
            ([0, 0, 0, 0], [255, 255, 255]),
            ([255, 255, 255, 255], [0, 0, 0]),
            ([100, 0, 50, 0], [155, 255, 205]),
            ([100, 0, 50, 100], [94, 155, 125]),
            ([5, 250, 128, 10], [240, 5, 122]),
        ];
        for (cmyk, want) in cases {
            assert_eq!(cmyk_to_rgb(&cmyk), want, "{cmyk:?}");
        }
    }

    #[test]
    fn sixteen_bit_grey_clips_and_colour_keeps_the_high_byte() {
        assert_eq!(
            grey16_to_rgb(&[0, 100, 255, 256, 65535]),
            [
                0, 0, 0, 100, 100, 100, 255, 255, 255, 255, 255, 255, 255, 255, 255
            ]
        );
        assert_eq!(
            high_bytes_to_rgb(&[0x1234, 0xABCD, 0x00FF, 0xFFFF], 4, false),
            [0x12, 0xAB, 0x00]
        );
        assert_eq!(high_bytes_to_rgb(&[0x8001, 7], 2, true), [0x80; 3]);
    }

    #[test]
    fn alpha_is_dropped_not_composited() {
        assert_eq!(
            colour_to_rgb(&[1, 2, 3, 0, 4, 5, 6, 255], 4),
            [1, 2, 3, 4, 5, 6]
        );
        assert_eq!(grey_to_rgb(&[9, 0, 7, 128], 2), [9, 9, 9, 7, 7, 7]);
    }
}
