//! The crop side cap's area filter against OpenCV's own output (`tests/golden/area.json`,
//! `scripts/codegen/frame_engine_area_goldens.py`): the same capped size and the same pixels.

use frame_engine::branches::area::{capped_size, resize_area};
use serde_json::Value;
use sha2::{Digest, Sha256};

const AREA: &str = include_str!("golden/area.json");

/// The generator's `pattern`: a hash of `(x, y, channel)` in 32-bit arithmetic.
fn pattern(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h as u64 {
        for x in 0..w as u64 {
            for c in 0..3u64 {
                let v = (x * 73_856_093) ^ (y * 19_349_663) ^ (c * 83_492_791);
                let v = ((v & 0xFFFF_FFFF) * 2_654_435_761) & 0xFFFF_FFFF;
                out.push((v >> 24) as u8);
            }
        }
    }
    out
}

fn n(v: &Value) -> usize {
    usize::try_from(v.as_u64().expect("a size")).expect("fits")
}

#[test]
fn every_capped_crop_is_opencvs_to_the_byte() {
    let golden: Value = serde_json::from_str(AREA).expect("area goldens");
    let cases = golden["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());
    for case in cases {
        let (w, h) = (n(&case["w"]), n(&case["h"]));
        let max_side = u32::try_from(n(&case["max_side"])).expect("a side");
        let size = capped_size(w, h, max_side).expect("every case is capped");
        assert_eq!(
            size,
            (n(&case["out_w"]), n(&case["out_h"])),
            "{w} x {h} at {max_side}"
        );
        let out = resize_area(&pattern(w, h), (w, h), size).expect("resize");
        let hex: String = Sha256::digest(&out)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            hex,
            case["sha256"].as_str().expect("a digest"),
            "{w} x {h} at {max_side}"
        );
    }
}
