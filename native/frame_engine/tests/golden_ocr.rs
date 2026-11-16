//! The `ocr` branch's ports against the goldens the Python engine, OpenCV 5.0.0 and pyclipper
//! 1.4.0 wrote (`tests/golden/ocr_*.json`, `scripts/codegen/frame_engine_text_goldens.py`).

use frame_engine::branches::ocr::contours::find_contours;
use frame_engine::branches::ocr::ctc::CtcDecoder;
use frame_engine::branches::ocr::db::{DbParams, boxes_from_map, sort_reading_order};
use frame_engine::branches::ocr::fill::box_score;
use frame_engine::branches::ocr::lines::{
    crop_dims, det_size, line_map, plan_batches, rec_ratio, resized_width,
};
use frame_engine::branches::ocr::min_rect::mini_box;
use frame_engine::branches::ocr::offset::unclip;
use serde_json::Value;

const DB: &str = include_str!("golden/ocr_db.json");
const LINES: &str = include_str!("golden/ocr_lines.json");
const CTC: &str = include_str!("golden/ocr_ctc.json");
const DICTIONARY: &str = include_str!("../../../models/paddleocr_rec_trt/ppocrv6_dict.txt");

fn json(text: &str) -> Value {
    serde_json::from_str(text).expect("the goldens are JSON")
}

fn int(v: &Value) -> i64 {
    v.as_i64().expect("an integer")
}

fn bits32(v: &Value) -> f32 {
    f32::from_bits(u32::try_from(int(v)).expect("32 bits"))
}

fn bits64(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().expect("64 bits"))
}

fn arr(v: &Value) -> &Vec<Value> {
    v.as_array().expect("an array")
}

/// `raster` of the generator, line for line.
fn raster(case: &Value) -> Vec<f32> {
    let (w, h) = (int(&case["w"]), int(&case["h"]));
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (2 * x + 1, 2 * y + 1);
            let mut q = 0i64;
            for s in arr(&case["shapes"]) {
                let (rx, ry) = (px - int(&s["cx"]), py - int(&s["cy"]));
                let (a, b) = (int(&s["a"]), int(&s["b"]));
                let ellipse = |a: i64, b: i64| rx * rx * b * b + ry * ry * a * a <= a * a * b * b;
                let inside = match s["kind"].as_str().expect("a kind") {
                    "bar" => {
                        let (dx, dy) = (int(&s["dx"]), int(&s["dy"]));
                        let n2 = dx * dx + dy * dy;
                        let (u, v) = (rx * dx + ry * dy, -rx * dy + ry * dx);
                        u * u <= a * a * n2 && v * v <= b * b * n2
                    }
                    "ring" => {
                        let i = int(&s["inner"]);
                        ellipse(a, b) && !ellipse(a - i, b - i)
                    }
                    _ => ellipse(a, b),
                };
                if inside {
                    q = q.max(int(&s["level"]) - ((x + 2 * y) % 5) * 4);
                }
            }
            let noise = int(&case["noise"]);
            if noise != 0 {
                let hsh =
                    ((x * 73856093) ^ (y * 19349663) ^ (int(&case["seed"]) * 83492791)) & 0xFFFF;
                q = (q + hsh % noise).min(255);
            }
            out.push(q as f32 / 255.0);
        }
    }
    out
}

#[test]
fn contours_boxes_and_scores_equal_opencvs_on_every_map() {
    let g = json(DB);
    let maps = arr(&g["maps"]);
    assert!(maps.len() >= 160);
    let (mut contours, mut boxes) = (0usize, 0usize);
    for (k, case) in maps.iter().enumerate() {
        let (w, h) = (int(&case["w"]) as usize, int(&case["h"]) as usize);
        let pred = raster(case);
        let p = arr(&case["params"]);
        let params = DbParams {
            thresh: p[0].as_f64().expect("thresh"),
            box_thresh: p[1].as_f64().expect("box_thresh"),
            unclip_ratio: p[2].as_f64().expect("ratio"),
        };
        let thresh = params.thresh as f32;
        let mask: Vec<u8> = pred.iter().map(|v| u8::from(*v > thresh)).collect();
        let found = find_contours(&mask, w, h);
        let want = arr(&case["contours"]);
        assert_eq!(found.len(), want.len(), "map {k}: contour count");
        for (i, (got, c)) in found.iter().zip(want).enumerate() {
            let pts: Vec<[i32; 2]> = arr(&c["points"])
                .iter()
                .map(|p| [int(&p[0]) as i32, int(&p[1]) as i32])
                .collect();
            assert_eq!(got, &pts, "map {k} contour {i}");
            let (mini, side) = mini_box(got);
            let mini_want: Vec<f32> = arr(&c["mini"]).iter().map(bits32).collect();
            assert_eq!(
                mini.as_flattened(),
                &mini_want[..],
                "map {k} contour {i}: mini box"
            );
            assert_eq!(
                side.to_bits(),
                bits32(&c["side"]).to_bits(),
                "map {k} contour {i}: side"
            );
            let score = box_score(&pred, w, h, &mini);
            assert_eq!(
                score.to_bits(),
                bits64(&c["score"]).to_bits(),
                "map {k} contour {i}: score"
            );
            contours += 1;
        }
        let dest = arr(&case["dest"]);
        let got = boxes_from_map(
            &pred,
            w,
            h,
            (int(&dest[0]) as u32, int(&dest[1]) as u32),
            params,
        );
        let want_boxes: Vec<Vec<i64>> = arr(&case["boxes"])
            .iter()
            .map(|b| arr(b).iter().map(int).collect())
            .collect();
        let got_boxes: Vec<Vec<i64>> = got
            .iter()
            .map(|b| {
                b.quad
                    .as_flattened()
                    .iter()
                    .map(|v| i64::from(*v))
                    .collect()
            })
            .collect();
        assert_eq!(got_boxes, want_boxes, "map {k}: boxes");
        let scores: Vec<u32> = got.iter().map(|b| b.score.to_bits()).collect();
        let want_scores: Vec<u32> = arr(&case["scores"])
            .iter()
            .map(|s| bits32(s).to_bits())
            .collect();
        assert_eq!(scores, want_scores, "map {k}: scores");
        boxes += got.len();
    }
    assert!(
        contours > 1000 && boxes > 50,
        "{contours} contours, {boxes} boxes"
    );
}

#[test]
fn the_unclipped_rectangle_equals_pyclippers_after_the_union() {
    let g = json(DB);
    for (k, row) in arr(&g["unclip"]).iter().enumerate() {
        let c: Vec<f32> = arr(&row["corners"]).iter().map(bits32).collect();
        let corners = [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]], [c[6], c[7]]];
        let grown = unclip(&corners, row["ratio"].as_f64().expect("ratio"));
        match (&row.get("mini"), grown) {
            (Some(want), Some(grown)) => {
                let pts: Vec<[i32; 2]> = grown.iter().map(|p| [p[0] as i32, p[1] as i32]).collect();
                let (mini, side) = mini_box(&pts);
                let want: Vec<f32> = arr(want).iter().map(bits32).collect();
                assert_eq!(mini.as_flattened(), &want[..], "rectangle {k}");
                assert_eq!(
                    side.to_bits(),
                    bits32(&row["side"]).to_bits(),
                    "rectangle {k}"
                );
            }
            (None, None) => {}
            (want, got) => panic!("rectangle {k}: Python {want:?}, native {got:?}"),
        }
    }
}

#[test]
fn detector_sizes_equal_python() {
    let g = json(LINES);
    for row in arr(&g["det_size"]) {
        let r = arr(row);
        let got = det_size(
            int(&r[0]),
            int(&r[1]),
            int(&r[2]),
            r[3].as_str().expect("limit"),
            int(&r[4]),
        );
        assert_eq!(got, Some((int(&r[5]), int(&r[6]))), "{row}");
    }
}

#[test]
fn crop_dims_ratios_maps_and_taps_equal_python() {
    let g = json(LINES);
    let mut maps = 0;
    for (k, row) in arr(&g["quads"]).iter().enumerate() {
        let q: Vec<[i32; 2]> = arr(&row["quad"])
            .iter()
            .map(|p| [int(&p[0]) as i32, int(&p[1]) as i32])
            .collect();
        let quad = [q[0], q[1], q[2], q[3]];
        let dims = arr(&row["dims"]);
        assert_eq!(crop_dims(&quad), (int(&dims[0]), int(&dims[1])), "quad {k}");
        assert_eq!(
            rec_ratio(&quad).map(f64::to_bits),
            row["ratio"].as_u64(),
            "quad {k}"
        );
        for m in row.get("maps").map(arr).into_iter().flatten() {
            let width = int(&m["width"]);
            let want: Vec<f32> = arr(&m["map"]).iter().map(bits32).collect();
            for (cap, taps) in [1u32, 2, 4, 8].into_iter().zip(arr(&m["taps"])) {
                let got = line_map(&quad, width, cap).expect("a map");
                assert_eq!(
                    got.taps,
                    [int(&taps[0]) as i32, int(&taps[1]) as i32],
                    "quad {k} width {width} cap {cap}"
                );
                // the solve's rounding noise (|v| near 1e-17 where the map is 0) is not compared
                for (a, b) in got.map.iter().zip(&want) {
                    let close = a == b || (a - b).abs() <= 1e-9 + 2.0 * f32::EPSILON * b.abs();
                    assert!(
                        close,
                        "quad {k} width {width}: {:?} against {want:?}",
                        got.map
                    );
                }
            }
            maps += 1;
        }
    }
    assert!(maps > 1000, "{maps} maps");
}

#[test]
fn batches_and_reading_order_equal_python() {
    let g = json(LINES);
    for case in arr(&g["batches"]) {
        let ratios: Vec<f64> = arr(&case["ratios"]).iter().map(bits64).collect();
        let got: Vec<(Vec<usize>, i64)> = plan_batches(&ratios, int(&case["batch"]) as usize, 2048)
            .into_iter()
            .map(|b| (b.indices, b.width))
            .collect();
        let want: Vec<(Vec<usize>, i64)> = arr(&case["out"])
            .iter()
            .map(|b| {
                (
                    arr(&b[0]).iter().map(|i| int(i) as usize).collect(),
                    int(&b[1]),
                )
            })
            .collect();
        assert_eq!(got, want);
        for r in &ratios {
            assert!(resized_width(*r, 2048) >= 1);
        }
    }
    for case in arr(&g["reading_order"]) {
        let quads: Vec<[[i32; 2]; 4]> = arr(&case["boxes"])
            .iter()
            .map(|b| {
                let p = |i: usize| [int(&b[i][0]) as i32, int(&b[i][1]) as i32];
                [p(0), p(1), p(2), p(3)]
            })
            .collect();
        let want: Vec<usize> = arr(&case["order"])
            .iter()
            .map(|i| int(i) as usize)
            .collect();
        assert_eq!(sort_reading_order(&quads), want);
    }
}

#[test]
fn ctc_lines_equal_python_over_the_real_dictionary() {
    let g = json(CTC);
    let decoder = CtcDecoder::from_text(DICTIONARY);
    assert_eq!(decoder.len() as i64, int(&g["classes"]));
    for (k, row) in arr(&g["rows"]).iter().enumerate() {
        let idx: Vec<i32> = arr(&row["indices"]).iter().map(|v| int(v) as i32).collect();
        let scores: Vec<f32> = arr(&row["scores"]).iter().map(bits32).collect();
        let line = decoder.decode(&idx, &scores).expect("decoded");
        assert_eq!(line.text, row["text"].as_str().expect("text"), "row {k}");
        assert_eq!(
            line.mean.to_bits(),
            bits32(&row["mean"]).to_bits(),
            "row {k}: mean"
        );
        assert_eq!(
            line.lowest.to_bits(),
            bits32(&row["min"]).to_bits(),
            "row {k}: min"
        );
    }
}
