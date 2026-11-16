//! The face branches' ports against the goldens the Python functions wrote (`tests/golden/
//! faces.json`, `scripts/codegen/frame_engine_face_goldens.py`): passes, decode, merge and NMS and
//! the normalisation equal to the bit; the alignment to the reference's precision.

use frame_engine::branches::faces::align::{estimate, warp_params};
use frame_engine::branches::faces::embed::normalise;
use frame_engine::branches::faces::passes::{DetPass, plan_passes};
use frame_engine::branches::faces::scrfd::{self, Dets, StrideOut};
use serde_json::Value;

const FACES: &str = include_str!("golden/faces.json");

fn goldens() -> Value {
    serde_json::from_str(FACES).expect("the face goldens are JSON")
}

fn int(v: &Value) -> i64 {
    v.as_i64().expect("an integer")
}

/// A float32 written as its shortest decimal, parsed as float32 (the text, not a float64 read).
fn f32_of(v: &Value) -> f32 {
    v.to_string().parse().expect("a float32")
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array().expect("numbers").iter().map(f32_of).collect()
}

fn rows<const N: usize>(v: &Value) -> Vec<[f32; N]> {
    v.as_array()
        .expect("rows")
        .iter()
        .map(|r| f32s(r).try_into().expect("a row of N"))
        .collect()
}

fn dets(v: &Value) -> Dets {
    Dets {
        scores: f32s(&v["scores"]),
        boxes: rows::<4>(&v["boxes"]),
        landmarks: rows::<10>(&v["landmarks"]),
    }
}

fn bits(d: &Dets) -> (Vec<u32>, Vec<[u32; 4]>, Vec<[u32; 10]>) {
    (
        d.scores.iter().map(|v| v.to_bits()).collect(),
        d.boxes.iter().map(|b| b.map(f32::to_bits)).collect(),
        d.landmarks.iter().map(|k| k.map(f32::to_bits)).collect(),
    )
}

fn decode_input(outputs: &Value, net: usize, threshold: f32) -> Dets {
    let get = |name: String| f32s(&outputs[name.as_str()]);
    let per: Vec<[Vec<f32>; 3]> = scrfd::FPN_STRIDES
        .iter()
        .map(|s| {
            [
                get(format!("score_{s}")),
                get(format!("bbox_{s}")),
                get(format!("kps_{s}")),
            ]
        })
        .collect();
    let view = |k: usize| StrideOut {
        scores: &per[k][0],
        boxes: &per[k][1],
        kps: &per[k][2],
    };
    scrfd::decode(&[view(0), view(1), view(2)], net, net, threshold).expect("decode")
}

#[test]
fn the_passes_equal_python_on_every_size_and_setting() {
    let g = goldens();
    let cases = g["passes"].as_array().expect("cases");
    assert!(cases.len() > 400);
    let mut tiled = 0;
    for case in cases {
        let s = case["settings"].as_array().expect("settings");
        let (w, h) = (int(&case["size"][0]), int(&case["size"][1]));
        let got =
            plan_passes(w, h, int(&s[0]), int(&s[1]), int(&s[2]), int(&s[3])).expect("passes");
        let want: Vec<DetPass> = case["passes"]
            .as_array()
            .expect("passes")
            .iter()
            .map(|p| {
                let v: Vec<i64> = p.as_array().expect("pass").iter().map(int).collect();
                DetPass {
                    scaled: [v[0], v[1]],
                    window: [v[2], v[3], v[4], v[5]],
                    inner: [v[6] == 1, v[7] == 1, v[8] == 1, v[9] == 1],
                }
            })
            .collect();
        tiled += usize::from(want.iter().any(|p| p.inner.contains(&true)));
        assert_eq!(got, want, "{w}x{h} under {s:?}");
    }
    assert!(tiled > 100, "only {tiled} tiled cases");
}

#[test]
fn the_decode_equals_python_to_the_bit_with_scores_at_the_threshold() {
    let g = goldens();
    for case in g["decode"].as_array().expect("cases") {
        let net = int(&case["net"]) as usize;
        let threshold = case["threshold"].as_f64().expect("threshold") as f32;
        let got = decode_input(&case["outputs"], net, threshold);
        assert_eq!(
            bits(&got),
            bits(&dets(&case["want"])),
            "net {net} at {threshold}"
        );
    }
}

#[test]
fn a_photo_of_passes_maps_and_merges_as_python_with_the_cap_counted() {
    let g = goldens();
    for case in g["photos"].as_array().expect("cases") {
        let s = case["settings"].as_array().expect("settings");
        let (w, h) = (int(&case["size"][0]), int(&case["size"][1]));
        let net = int(&s[0]);
        let passes = plan_passes(w, h, net, int(&s[1]), int(&s[2]), int(&s[3])).expect("passes");
        let outputs = case["outputs"].as_array().expect("outputs");
        assert_eq!(passes.len(), outputs.len());
        let parts: Vec<Dets> = passes
            .iter()
            .zip(outputs)
            .map(|(p, o)| scrfd::to_source(p, w, h, &decode_input(o, net as usize, 0.5)))
            .collect();
        for (k, (got, want)) in parts
            .iter()
            .zip(case["to_source"].as_array().expect("parts"))
            .enumerate()
        {
            assert_eq!(bits(got), bits(&dets(want)), "{w}x{h} pass {k}");
        }
        let nms = case["nms"].as_f64().expect("nms") as f32;
        let faces = scrfd::merge(&parts, nms, int(&case["cap"]) as usize);
        assert_eq!(
            bits(&faces.dets),
            bits(&dets(&case["want"])),
            "{w}x{h} merged"
        );
        assert_eq!(faces.overflow as i64, int(&case["want"]["overflow"]));
    }
}

#[test]
fn nms_keeps_pythons_rows_in_pythons_order_with_ties_and_the_threshold_itself() {
    let g = goldens();
    for case in g["nms"].as_array().expect("cases") {
        let threshold = case["threshold"].as_f64().expect("threshold") as f32;
        let got = scrfd::nms(
            &rows::<4>(&case["boxes"]),
            &f32s(&case["scores"]),
            threshold,
        );
        let want: Vec<usize> = case["keep"]
            .as_array()
            .expect("keep")
            .iter()
            .map(|v| int(v) as usize)
            .collect();
        assert_eq!(got, want, "at {threshold}");
    }
}

/// Float32 ulps between two values of the same sign.
fn ulps(a: f32, b: f32) -> u32 {
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
}

#[test]
fn the_alignment_matches_the_reference_to_its_precision() {
    let g = goldens();
    let (mut identical, mut total) = (0, 0);
    for case in g["align"].as_array().expect("cases") {
        let lmk: [f32; 10] = rows::<2>(&case["landmarks"])
            .concat()
            .try_into()
            .expect("five points");
        let got = warp_params(&lmk).expect("warp params");
        assert_eq!(got.ok, case["ok"].as_bool().expect("ok"), "{lmk:?}");
        if !got.ok {
            continue;
        }
        assert_eq!(i64::from(got.taps), int(&case["taps"]), "{lmk:?}");
        let src: [[f64; 2]; 5] =
            std::array::from_fn(|k| [f64::from(lmk[2 * k]), f64::from(lmk[2 * k + 1])]);
        let fwd = estimate(&src).expect("estimate");
        let want_fwd: Vec<f64> = case["forward"]
            .as_array()
            .expect("forward")
            .iter()
            .map(|v| v.as_str().expect("repr").parse().expect("f64"))
            .collect();
        let scale = want_fwd.iter().map(|v| v.abs()).fold(0.0, f64::max);
        for (g, w) in fwd.iter().flatten().zip(&want_fwd) {
            assert!(
                (g - w).abs() <= 1e-12 * scale,
                "forward {g} vs {w} for {lmk:?}"
            );
        }
        let want_inv = f32s(&case["inverse"]);
        for (g, w) in got.inv.iter().zip(&want_inv) {
            total += 1;
            identical += usize::from(g.to_bits() == w.to_bits());
            assert!(ulps(*g, *w) <= 2, "inverse {g} vs {w} for {lmk:?}");
        }
    }
    assert!(
        identical * 100 >= total * 95,
        "{identical} of {total} inverse values identical"
    );
}

#[test]
fn the_vectors_are_normalised_as_numpy_does_in_float32() {
    let g = goldens();
    for case in g["normalise"].as_array().expect("cases") {
        for (row, unit) in case["vectors"]
            .as_array()
            .expect("rows")
            .iter()
            .zip(case["unit"].as_array().expect("rows"))
        {
            let mut got = f32s(row);
            normalise(&mut got);
            let want = f32s(unit);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "a row of {}",
                got.len()
            );
        }
    }
}
