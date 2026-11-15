//! The geometry and weight ports against the goldens the Python functions wrote
//! (`tests/golden/`, `scripts/codegen/frame_engine_goldens.py`): equal to the bit.

use frame_engine::geometry::{
    blur_rect, crop_rect, letterbox_geometry, pairwise_sum, stored_box_rects, undo_letterbox,
    weights, whole_square_geometry,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const GEOMETRY: &str = include_str!("golden/geometry.json");
const WEIGHTS: &str = include_str!("golden/weights.jsonl");

fn f(v: &Value) -> f64 {
    v.as_f64().expect("a number")
}

fn i(v: &Value) -> i64 {
    v.as_i64().expect("an integer")
}

fn four(v: &Value) -> [f64; 4] {
    let a = v.as_array().expect("four numbers");
    [f(&a[0]), f(&a[1]), f(&a[2]), f(&a[3])]
}

fn four_ints(v: &Value) -> [i64; 4] {
    let a = v.as_array().expect("four integers");
    [i(&a[0]), i(&a[1]), i(&a[2]), i(&a[3])]
}

fn bits(x: [f64; 4]) -> [u64; 4] {
    x.map(f64::to_bits)
}

fn geometry() -> Value {
    serde_json::from_str(GEOMETRY).expect("the geometry goldens are JSON")
}

#[test]
fn letterbox_and_square_equal_python_on_every_size_and_target() {
    let g = geometry();
    let rows = g["letterbox"].as_array().expect("rows");
    assert!(rows.len() > 4000);
    for row in rows {
        let (w, h, target) = (i(&row[0]), i(&row[1]), i(&row[2]));
        let lb = letterbox_geometry(w, h, target).expect("geometry");
        let got = [lb.new_w, lb.new_h, lb.left, lb.top];
        let want = [i(&row[3]), i(&row[4]), i(&row[5]), i(&row[6])];
        assert_eq!(got, want, "letterbox {w}x{h} -> {target}");
        assert_eq!(
            lb.scale.to_bits(),
            f(&row[7]).to_bits(),
            "scale of {w}x{h} -> {target}"
        );
        assert_eq!(lb.pad_w.to_bits(), f(&row[8]).to_bits());
        assert_eq!(lb.pad_h.to_bits(), f(&row[9]).to_bits());
    }
    for row in g["square"].as_array().expect("rows") {
        let (w, h, target) = (i(&row[0]), i(&row[1]), i(&row[2]));
        let sq = whole_square_geometry(w, h, target).expect("geometry");
        let want = [i(&row[3]), i(&row[4]), i(&row[5]), i(&row[6])];
        assert_eq!(
            [sq.new_w, sq.new_h, sq.left, sq.top],
            want,
            "square {w}x{h} -> {target}"
        );
    }
}

#[test]
fn unmapping_and_rectangles_equal_python_to_the_bit() {
    let g = geometry();
    for case in g["undo_letterbox"].as_array().expect("cases") {
        let pad = case["pad"].as_array().expect("pad");
        let got = undo_letterbox(
            four(&case["box"]),
            f(&case["scale"]),
            (f(&pad[0]), f(&pad[1])),
        );
        assert_eq!(bits(got), bits(four(&case["out"])), "{case}");
    }
    let cases = g["crop_rect"].as_array().expect("cases");
    assert!(cases.len() >= 900);
    for case in cases {
        let got = crop_rect(four(&case["box"]), i(&case["w"]), i(&case["h"]));
        assert_eq!(got, four_ints(&case["out"]), "{case}");
    }
    for case in g["blur_rect"].as_array().expect("cases") {
        let got = blur_rect(four(&case["box"]), i(&case["w"]), i(&case["h"]));
        let want = (!case["out"].is_null()).then(|| four_ints(&case["out"]));
        assert_eq!(got, want, "{case}");
    }
    for case in g["stored_box_rects"].as_array().expect("cases") {
        let boxes: Vec<[f64; 4]> = case["boxes"]
            .as_array()
            .expect("rows")
            .iter()
            .map(four)
            .collect();
        let got = stored_box_rects(&boxes, i(&case["w"]), i(&case["h"]));
        let want: Vec<[i64; 4]> = case["out"]
            .as_array()
            .expect("rows")
            .iter()
            .map(four_ints)
            .collect();
        assert_eq!(got, want, "{case}");
    }
}

#[test]
fn the_row_sum_is_numpys_pairwise_sum_to_the_bit() {
    let g = geometry();
    let rows = g["pairwise_sum"].as_array().expect("rows");
    assert!(rows.len() >= 50);
    for row in rows {
        let values: Vec<f64> = row["bits"]
            .as_array()
            .expect("values")
            .iter()
            .map(|b| f64::from_bits(b.as_u64().expect("bits")))
            .collect();
        let want = row["sum"].as_u64().expect("sum");
        assert_eq!(
            pairwise_sum(&values).to_bits(),
            want,
            "{} values",
            values.len()
        );
    }
}

#[test]
fn every_weight_set_equals_pythons_bit_for_bit() {
    let mut checked = 0;
    let mut plain = 0;
    for line in WEIGHTS.lines() {
        let row: Value = serde_json::from_str(line).expect("a row");
        let [n_in, n_out, left, take] = [0, 1, 2, 3].map(|k| i(&row[k]));
        let antialias = row[4].as_bool().expect("flag");
        let w = weights(n_in, n_out, left, take, antialias).expect("weights");
        let digest: String = Sha256::digest(&w.blob)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let head = [i(&row[5]), i(&row[6]), i(&row[7]), i(&row[8])];
        assert_eq!(
            [w.n as i64, w.kmax as i64, w.lo, w.hi],
            head,
            "n, kmax, lo, hi of ({n_in}, {n_out}, {left}, {take}, {antialias})"
        );
        assert_eq!(
            &digest[..32],
            row[9].as_str().expect("digest"),
            "bits of ({n_in}, {n_out}, {left}, {take}, {antialias})"
        );
        checked += 1;
        plain += usize::from(!antialias);
    }
    assert!(checked > 20_000, "{checked} weight sets");
    assert!(plain > 5_000, "{plain} of them the plain branch");
}
