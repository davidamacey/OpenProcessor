//! The renditions' geometry against the goldens the Python engine wrote
//! (`tests/golden/renditions.json`, `scripts/codegen/frame_engine_text_goldens.py`): Pillow
//! `thumbnail` sizes and the Lanczos weight sets, bit for bit.

use frame_engine::geometry::{fit_size, lanczos_weights};
use serde_json::Value;
use sha2::{Digest, Sha256};

const RENDITIONS: &str = include_str!("golden/renditions.json");

fn int(v: &Value) -> i64 {
    v.as_i64().expect("an integer")
}

#[test]
fn thumbnail_sizes_equal_pillows() {
    let g: Value = serde_json::from_str(RENDITIONS).expect("JSON");
    let rows = g["fit_size"].as_array().expect("rows");
    assert!(rows.len() > 10_000);
    for r in rows {
        let r = r.as_array().expect("a row");
        assert_eq!(
            fit_size(int(&r[0]), int(&r[1]), int(&r[2])),
            (int(&r[3]), int(&r[4])),
            "{r:?}"
        );
    }
}

#[test]
fn lanczos_weight_sets_equal_pythons_to_the_bit() {
    let g: Value = serde_json::from_str(RENDITIONS).expect("JSON");
    let sets = g["lanczos"].as_array().expect("sets");
    assert!(sets.len() > 1000);
    let mut differ = Vec::new();
    for s in sets {
        let s = s.as_array().expect("a set");
        let (n_in, n_out) = (int(&s[0]), int(&s[1]));
        let w = lanczos_weights(n_in, n_out, 0, n_out).expect("weights");
        assert_eq!(
            [w.n as i64, w.kmax as i64, w.lo, w.hi],
            [int(&s[2]), int(&s[3]), int(&s[4]), int(&s[5])],
            "{n_in} -> {n_out}"
        );
        let digest: String = Sha256::digest(&w.blob)[..16]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if digest != s[6].as_str().expect("a digest") {
            differ.push((n_in, n_out));
        }
    }
    assert!(
        differ.is_empty(),
        "{} of {} sets differ: {differ:?}",
        differ.len(),
        sets.len()
    );
}

#[test]
fn the_frame_budget_with_renditions_equals_pythons_need_bytes() {
    use frame_engine::jpeg_header::{JpegHeader, Kind};
    use frame_engine::photo::budget::need_bytes;

    let g: Value = serde_json::from_str(RENDITIONS).expect("JSON");
    let rows = g["need_bytes"].as_array().expect("rows");
    assert!(rows.len() > 100);
    let mut scratch_won = 0;
    for r in rows {
        let r = r.as_array().expect("a row");
        let header = JpegHeader {
            kind: Kind::Jpeg,
            width: int(&r[0]) as u32,
            height: int(&r[1]) as u32,
            components: 3,
            precision: 8,
            progressive: false,
            orientation: int(&r[2]) as u32,
            lossless: false,
            end: 0,
            icc: None,
        };
        let sides: Vec<u32> = r[3]
            .as_array()
            .expect("sides")
            .iter()
            .map(|s| int(s) as u32)
            .collect();
        let frame = u64::from(header.width) * u64::from(header.height) * 3;
        let want = int(&r[4]) as u64;
        if want > frame * if header.orientation == 1 { 1 } else { 2 } {
            scratch_won += 1;
        }
        assert_eq!(need_bytes(&header, &sides), want, "{r:?}");
    }
    assert!(scratch_won > 10, "the rows exercise the rendition scratch");
}
