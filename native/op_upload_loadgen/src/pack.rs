//! Packing: which photos of the list go into one request. The server refuses a request over
//! `upload_max_images_per_request` photos or `upload_max_bytes_per_request` photo bytes with 413
//! (`src/config/curation.py`, defaults 128 and 512 MiB), so a pack never exceeds either.

use std::fmt;
use std::ops::Range;

/// Why a list cannot be packed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    /// The image cap is zero: no request could carry a photo.
    NoImagesAllowed,
    /// The list is empty.
    EmptyList,
    /// One photo alone is over the byte cap, so no request could carry it.
    PhotoOverByteCap {
        /// Position of the photo in the list.
        index: usize,
        /// Its size in bytes.
        size: u64,
        /// The byte cap.
        max_bytes: u64,
    },
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackError::NoImagesAllowed => write!(f, "--max-images must be at least 1"),
            PackError::EmptyList => write!(f, "the list holds no photo"),
            PackError::PhotoOverByteCap {
                index,
                size,
                max_bytes,
            } => write!(
                f,
                "photo {index} is {size} bytes, over --max-bytes {max_bytes}: no request could carry it"
            ),
        }
    }
}

impl std::error::Error for PackError {}

/// Split the list, given by its photo sizes in order, into consecutive requests: a request takes
/// photos while it holds fewer than `max_images` and the next photo keeps the photo bytes within
/// `max_bytes`. Every photo lands in exactly one pack, in list order.
///
/// # Errors
///
/// Returns `Err` for an empty list, a zero image cap, or a photo that alone exceeds `max_bytes`
/// (the run is refused rather than sending a request the server must answer with 413).
pub fn pack(
    sizes: &[u64],
    max_images: usize,
    max_bytes: u64,
) -> Result<Vec<Range<usize>>, PackError> {
    if max_images == 0 {
        return Err(PackError::NoImagesAllowed);
    }
    if sizes.is_empty() {
        return Err(PackError::EmptyList);
    }
    if let Some((index, &size)) = sizes.iter().enumerate().find(|(_, s)| **s > max_bytes) {
        return Err(PackError::PhotoOverByteCap {
            index,
            size,
            max_bytes,
        });
    }
    let mut packs = Vec::new();
    let (mut start, mut bytes) = (0, 0u64);
    for (i, &size) in sizes.iter().enumerate() {
        if i > start && (i - start >= max_images || bytes + size > max_bytes) {
            packs.push(start..i);
            (start, bytes) = (i, 0);
        }
        bytes += size;
    }
    packs.push(start..sizes.len());
    Ok(packs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_of(sizes: &[u64], r: &Range<usize>) -> u64 {
        sizes[r.clone()].iter().sum()
    }

    #[test]
    fn the_image_cap_closes_a_pack() {
        let packs = pack(&[1; 7], 3, 1000).unwrap();
        assert_eq!(packs, vec![0..3, 3..6, 6..7]);
    }

    #[test]
    fn the_byte_cap_closes_a_pack_and_an_exact_fit_stays() {
        let sizes = [40, 60, 50, 50, 1];
        let packs = pack(&sizes, 128, 100).unwrap();
        assert_eq!(packs, vec![0..2, 2..4, 4..5]);
        assert!(packs.iter().all(|r| bytes_of(&sizes, r) <= 100));
    }

    #[test]
    fn both_caps_hold_together_and_every_photo_is_in_one_pack_in_order() {
        let sizes: Vec<u64> = (0..200).map(|i| 1 + (i * 37) % 90).collect();
        let packs = pack(&sizes, 8, 300).unwrap();
        let mut next = 0;
        for r in &packs {
            assert_eq!(r.start, next);
            assert!((1..=8).contains(&r.len()));
            assert!(bytes_of(&sizes, r) <= 300);
            next = r.end;
        }
        assert_eq!(next, sizes.len());
    }

    #[test]
    fn a_photo_over_the_byte_cap_refuses_the_run() {
        assert_eq!(
            pack(&[10, 101, 5], 4, 100),
            Err(PackError::PhotoOverByteCap {
                index: 1,
                size: 101,
                max_bytes: 100
            })
        );
    }

    #[test]
    fn a_zero_cap_or_an_empty_list_refuses() {
        assert_eq!(pack(&[1], 0, 10), Err(PackError::NoImagesAllowed));
        assert_eq!(pack(&[], 1, 10), Err(PackError::EmptyList));
    }
}
