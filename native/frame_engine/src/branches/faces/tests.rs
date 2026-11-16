use bytes::Bytes;
use gpu_frame::{KernelArg, OutputTensor};

use super::embed::{self, normalise};
use super::scrfd::{Dets, Faces, OUTPUT_NAMES};
use super::warp::{self, FaceWarps};
use super::*;

fn spec(coarse: u32, tile_side: u32, overlap: u32) -> FaceSpec {
    FaceSpec {
        detector: "scrfd".into(),
        input_size: 64,
        recogniser: "arcface".into(),
        batch: 2,
        score_min: 0.5,
        nms: 0.4,
        max_faces: 3,
        coarse,
        tile_side,
        overlap,
    }
}

fn fp32(name: &str, shape: Vec<i64>, values: &[f32]) -> OutputTensor {
    OutputTensor {
        name: name.into(),
        datatype: "FP32".into(),
        shape,
        data: Bytes::from(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        ),
    }
}

/// A detector answer for `batch` inputs of 64: nothing scores, except one anchor of stride 8 of
/// input `fire` (`(input, anchor, score, ltrb)`).
fn answer(batch: usize, fire: Option<(usize, usize, f32, [f32; 4])>) -> Vec<OutputTensor> {
    OUTPUT_NAMES
        .iter()
        .enumerate()
        .map(|(k, name)| {
            let stride = scrfd::FPN_STRIDES[k % 3];
            let anchors = (64 / stride) * (64 / stride) * 2;
            let d = [1, 4, 10][k / 3];
            let mut v = vec![0f32; batch * anchors * d];
            if let (Some((input, anchor, score, ltrb)), 8) = (fire, stride) {
                let at = (input * anchors + anchor) * d;
                match k / 3 {
                    0 => v[at] = score,
                    1 => v[at..at + 4].copy_from_slice(&ltrb),
                    _ => {}
                }
            }
            fp32(name, vec![batch as i64, anchors as i64, d as i64], &v)
        })
        .collect()
}

#[test]
fn the_passes_go_in_calls_of_eight_each_at_the_top_left_of_its_own_zero_filled_input() {
    // 1000 x 600 at a tile side of 192 in a 64 input: whole, coarse and 4 x 2 tiles (10 passes)
    let calls = detector_calls(&spec(32, 192, 8), 1000, 600).expect("calls");
    assert_eq!(
        calls.iter().map(|c| c.passes.len()).collect::<Vec<_>>(),
        [8, 2]
    );
    assert_eq!(calls[1].input.shape, [2, 3, 64, 64]);
    let square = 3 * 64 * 64;
    for call in &calls {
        for (i, (r, p)) in call.resamples.iter().zip(&call.passes).enumerate() {
            assert_eq!(r.dst_offset, i * square, "pass {i} starts its own input");
            assert_eq!(
                (r.roi, r.size, r.window),
                ([0, 0, 1000, 600], p.scaled, p.window)
            );
            assert_eq!(
                (r.plane, r.row_stride, r.filter),
                (64 * 64, 64, Filter::Bilinear)
            );
        }
    }
    assert_eq!(calls[0].passes[0].scaled, [64, 38], "the whole frame first");
    assert_eq!(
        calls[0].passes[1].scaled,
        [32, 19],
        "then the coarse square"
    );
    assert_eq!(
        calls[0].passes[2].scaled,
        [192, 115],
        "then tiles of the frame at 192"
    );
    let small = detector_calls(&spec(0, 192, 8), 150, 100).expect("calls");
    assert_eq!(small.len(), 1);
    assert_eq!(
        small[0].passes.len(),
        1,
        "a photo within the tile side is one pass"
    );
}

#[test]
fn the_settle_reads_each_pass_from_its_batch_element_and_maps_it_to_source_pixels() {
    let spec = spec(32, 0, 0);
    let calls = detector_calls(&spec, 128, 64).expect("calls");
    assert_eq!(calls[0].passes.len(), 2);
    // input 1 is the coarse pass (the frame at 32 x 16, scale 0.25): anchor 2 sits at (8, 0)
    let fire = Some((1, 2, 0.9, [1.0, 0.0, 1.0, 1.0]));
    let faces = settle(&spec, 128, 64, &calls, &[answer(2, fire)]).expect("settle");
    assert_eq!(faces.dets.scores, [0.9]);
    assert_eq!(faces.dets.boxes, [[0.0, 0.0, 64.0, 32.0]]);
    assert_eq!(
        faces.dets.landmarks,
        [[32.0, 0.0, 32.0, 0.0, 32.0, 0.0, 32.0, 0.0, 32.0, 0.0]]
    );
    let none = settle(&spec, 128, 64, &calls, &[answer(2, None)]).expect("settle");
    assert!(none.dets.is_empty());
    assert_eq!(none.overflow, 0);
}

#[test]
fn a_detector_answer_that_does_not_fit_its_calls_is_an_error_not_a_guess() {
    let spec = spec(0, 0, 0);
    let calls = detector_calls(&spec, 64, 64).expect("calls");
    assert!(
        settle(&spec, 64, 64, &calls, &[]).is_err(),
        "an answer missing"
    );
    let mut short = answer(1, None);
    short.pop();
    assert!(
        settle(&spec, 64, 64, &calls, &[short]).is_err(),
        "an output missing"
    );
    let mut bad = answer(1, None);
    bad[0] = fp32("score_8", vec![1, 3, 1], &[0.0; 3]);
    assert!(
        settle(&spec, 64, 64, &calls, &[bad]).is_err(),
        "an anchor grid of another size"
    );
}

fn faces(n: usize, overflow: usize) -> Faces {
    let mut dets = Dets::default();
    for k in 0..n {
        let x = 10.0 * k as f32;
        dets.scores.push(0.9 - 0.1 * k as f32);
        dets.boxes.push([x, 5.0, x + 8.0, 13.0]);
        let mut lmk = [0f32; 10];
        let points = [(3.0, 3.0), (6.0, 3.0), (4.5, 5.0), (3.5, 7.0), (5.5, 7.0)];
        for (j, (dx, dy)) in points.iter().enumerate() {
            lmk[2 * j] = x + dx;
            lmk[2 * j + 1] = 5.0 + dy;
        }
        dets.landmarks.push(lmk);
    }
    Faces { dets, overflow }
}

#[test]
fn the_face_tensors_have_the_wire_layout_with_the_overflow_counted() {
    let out = outputs(&faces(2, 7));
    let names: Vec<&str> = out.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        names,
        [
            "face_boxes",
            "face_scores",
            "face_landmarks",
            "face_overflow"
        ]
    );
    assert!(
        matches!(&out[0].1, Tensor::Fp32 { shape, data } if shape == &[2, 4] && data.len() == 8)
    );
    assert!(matches!(&out[2].1, Tensor::Fp32 { shape, .. } if shape == &[2, 10]));
    assert_eq!(
        out[3].1,
        Tensor::Int32 {
            shape: vec![1],
            data: vec![7]
        }
    );
    let none = outputs(&Faces::default());
    assert!(matches!(&none[0].1, Tensor::Fp32 { shape, .. } if shape == &[0, 4]));
    assert_eq!(
        none[3].1,
        Tensor::Int32 {
            shape: vec![1],
            data: vec![0]
        }
    );
}

#[test]
fn face_blur_sums_cover_each_clamped_box_and_zero_an_empty_one() {
    let mut f = faces(2, 0);
    f.dets.boxes[1] = [200.0, 5.0, 210.0, 9.0];
    let clamped = blur_rects(&f, 100, 50);
    assert_eq!(clamped, [Some([0, 5, 8, 13]), None]);
    let sums = blur_sums(&clamped, &[[11, 22]]).expect("sums");
    assert_eq!(
        sums,
        Tensor::Int64 {
            shape: vec![2, 3],
            data: vec![64, 11, 22, 0, 0, 0]
        }
    );
    assert!(blur_sums(&clamped, &[]).is_err());
    assert!(blur_sums(&clamped, &[[1, 1], [2, 2]]).is_err());
    let none = blur_sums(&[], &[]).expect("none");
    assert_eq!(
        none,
        Tensor::Int64 {
            shape: vec![0, 3],
            data: vec![]
        }
    );
}

#[test]
fn the_recogniser_calls_take_the_usable_faces_in_chunks_of_the_batch() {
    let mut f = faces(5, 0);
    f.dets.landmarks[1] = [4.0; 10];
    let plan = embed::plan(&f, 2).expect("plan");
    assert_eq!(plan.embedded, [true, false, true, true, true]);
    let shapes: Vec<[i64; 4]> = plan.chunks.iter().map(|c| c.input.shape).collect();
    assert_eq!(shapes, [[2, 3, 112, 112], [2, 3, 112, 112]]);
    assert_eq!(plan.chunks[0].warps.bytes(), 2 * 3 * 112 * 112);
    assert!(
        plan.chunks
            .iter()
            .all(|c| c.warps.taps.iter().all(|t| *t == 1)),
        "small faces"
    );
    assert!(
        embed::plan(&Faces::default(), 2)
            .expect("plan")
            .chunks
            .is_empty()
    );
    let mut bad = faces(1, 0);
    bad.dets.landmarks[0][3] = f32::NAN;
    assert!(
        embed::plan(&bad, 2).is_err(),
        "the reference's SVD refuses a NaN landmark"
    );
}

#[test]
fn the_vectors_are_stacked_in_call_order_and_each_row_unit_length() {
    let a = fp32("output", vec![2, 2], &[3.0, 4.0, 0.0, 0.0]);
    let b = fp32("output", vec![1, 2], &[0.0, -2.0]);
    let [vectors, flags] =
        embed::outputs(&[true, true, false, true], &[vec![a], vec![b]]).expect("outputs");
    let unit = vec![0.6, 0.8, 0.0, 0.0, 0.0, -1.0];
    assert_eq!(
        vectors.1,
        Tensor::Fp32 {
            shape: vec![3, 2],
            data: unit
        }
    );
    assert_eq!(flags.0, "face_embedded");
    let [none, _] = embed::outputs(&[], &[]).expect("outputs");
    assert_eq!(
        none.1,
        Tensor::Fp32 {
            shape: vec![0, 0],
            data: vec![]
        }
    );
    let wide = fp32("output", vec![1, 3], &[1.0, 0.0, 0.0]);
    let b = fp32("output", vec![1, 2], &[0.0, -2.0]);
    assert!(embed::outputs(&[true, true], &[vec![wide], vec![b]]).is_err());
    let mut tiny = [1e-12f32, 0.0];
    normalise(&mut tiny);
    assert_eq!(tiny, [1e-12 / 1e-10, 0.0], "the norm is floored at 1e-10");
}

#[test]
fn the_warp_launch_is_warp_intos_grid_and_argument_order() {
    let cfg = warp::launch(3, 112).expect("launch");
    assert_eq!((cfg.grid, cfg.block), ((98, 3, 1), (128, 1, 1)));
    assert!(warp::launch(0, 112).is_err());
    let w = FaceWarps {
        inv: vec![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        taps: vec![3],
        size: 112,
    };
    let (inv, taps) = warp::blobs(&w);
    assert_eq!(inv.len(), 24);
    assert_eq!(f32::from_le_bytes(inv[20..24].try_into().expect("4")), 6.0);
    assert_eq!(taps, 3i32.to_le_bytes());
    let args = warp::args((0x1000, 640, 480), 0x2000, 0x3000, 1, 112, 0x4000).expect("args");
    assert_eq!(
        args,
        [
            KernelArg::U64(0x1000),
            KernelArg::I32(480),
            KernelArg::I32(640),
            KernelArg::U64(0x2000),
            KernelArg::U64(0x3000),
            KernelArg::I32(1),
            KernelArg::I32(112),
            KernelArg::U64(0x4000),
            KernelArg::I64(3 * 112 * 112),
        ]
    );
}
