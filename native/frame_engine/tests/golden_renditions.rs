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
