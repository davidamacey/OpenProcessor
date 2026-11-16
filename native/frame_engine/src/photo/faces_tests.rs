//! The face branches in the photo flow on the host fake: the calls they make, their tensors, the
//! cap, a photo with no face, and each model failing.

use super::*;
use crate::photo::clock;

/// Faces in a 64 input, one recogniser batch of 2; `extra` is spliced into the `faces` object.
fn faces_plan(extra: &str) -> String {
    format!(
        r#"{{"version": 1, "branches": ["faces", "face_embed"], "crops": "none", "faces": {{"detector": "scrfd", "input_size": 64, "recogniser": "arcface", "batch": 2, "score_min": 0.5, "nms": 0.4, "max_faces": 50{extra}}}}}"#
    )
}

/// A face at the stride-8 centre `(cx, cy)` (in grid cells of 8 input pixels), one stride
/// across, its landmarks laid out as a face.
fn fire(cx: usize, cy: usize, score: f32) -> (usize, f32, [f32; 4], [f32; 10]) {
    let kps = [-0.4, -0.3, 0.4, -0.3, 0.0, 0.1, -0.3, 0.5, 0.3, 0.5];
    (2 * (cy * 8 + cx), score, [1.0, 1.0, 1.0, 1.0], kps)
}

fn calls_to(b: &Bed, model: &str) -> Vec<[i64; 4]> {
    b.fake
        .calls()
        .into_iter()
        .filter(|(m, _)| m == model)
        .map(|(_, s)| s)
        .collect()
}

#[tokio::test]
async fn a_face_is_found_aligned_and_embedded_with_every_face_tensor() {
    let b = bed();
    b.fake.set(|s| s.face_fires = vec![fire(3, 3, 0.9)]);
    let answer = b
        .flow
        .process(photo(&faces_plan("")))
        .await
        .expect("answer");
    assert_eq!(answer.branches["faces"], BranchState::Ok);
    assert_eq!(answer.branches["face_embed"], BranchState::Ok);
    assert_eq!(calls_to(&b, "scrfd"), [[1, 3, 64, 64]]);
    assert_eq!(calls_to(&b, "arcface"), [[1, 3, 112, 112]]);
    let events = b.fake.events();
    assert_eq!(
        events[0],
        Event::Decode {
            bytes: vec![3 * 64 * 64],
            fills: vec![Some(0)]
        },
        "the pass on a zero-filled input, written by the decode job"
    );
    assert!(events.contains(&Event::Warps(vec![1])));
    assert!(events.contains(&Event::Launch { writes: 0, blur: 1 }));
    let want = [
        ("face_boxes", vec![1, 4]),
        ("face_scores", vec![1]),
        ("face_landmarks", vec![1, 10]),
        ("face_blur_sums", vec![1, 3]),
        ("face_overflow", vec![1]),
        ("face_embeddings", vec![1, 8]),
        ("face_embedded", vec![1]),
    ];
    for (name, dims) in want {
        let t = output(&answer, name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(shape(t), dims, "{name}");
    }
    // the whole pass is the frame at 64 x 48: the box [16, 16, 32, 32] maps back through the
    // pass's float32 scale, 64 / 1000 (= 48 / 750)
    let Some(Tensor::Fp32 { data: boxes, .. }) = output(&answer, "face_boxes") else {
        panic!("boxes");
    };
    let scale = (64.0f64 / 1000.0) as f32;
    assert_eq!(
        boxes,
        &[16.0 / scale, 16.0 / scale, 32.0 / scale, 32.0 / scale]
    );
    let Some(Tensor::Fp32 { data: v, .. }) = output(&answer, "face_embeddings") else {
        panic!("vectors");
    };
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6, "unit length, got {norm}");
    let text = b.flow.metrics.render();
    for stage in clock::ALL.iter().filter(|s| s.starts_with("face_")) {
        assert!(
            text.contains(&format!(
                "op_frame_engine_stage_observations_total{{stage=\"{stage}\"}} 1"
            )),
            "{stage}"
        );
    }
    assert_released(&b).await;
}

#[tokio::test]
async fn a_large_photo_is_searched_in_tiles_in_calls_of_eight_passes() {
    let b = bed();
    b.fake.set(|s| s.face_fires = vec![fire(3, 3, 0.9)]);
    let plan = faces_plan(r#", "coarse": 32, "tile_side": 192, "overlap": 8"#);
    let answer = b.flow.process(photo(&plan)).await.expect("answer");
    // whole, coarse, then the frame at 192 x 144 in 4 x 3 tiles: 14 passes
    assert_eq!(calls_to(&b, "scrfd"), [[8, 3, 64, 64], [6, 3, 64, 64]]);
    let Event::Decode { bytes, fills } = &b.fake.events()[0] else {
        panic!("a decode first");
    };
    assert_eq!(bytes, &[8 * 3 * 64 * 64, 6 * 3 * 64 * 64]);
    assert_eq!(fills, &[Some(0), Some(0)]);
    assert_eq!(answer.branches["faces"], BranchState::Ok);
    assert_released(&b).await;
}

#[tokio::test]
async fn faces_past_the_cap_are_counted_not_answered() {
    let b = bed();
    b.fake
        .set(|s| s.face_fires = vec![fire(1, 1, 0.9), fire(5, 1, 0.8), fire(1, 5, 0.7)]);
    let answer = b
        .flow
        .process(photo(
            &faces_plan("").replace(r#""max_faces": 50"#, r#""max_faces": 2"#),
        ))
        .await
        .expect("answer");
    assert_eq!(
        output(&answer, "face_overflow"),
        Some(&Tensor::Int32 {
            shape: vec![1],
            data: vec![1]
        })
    );
    let Some(Tensor::Fp32 { data: scores, .. }) = output(&answer, "face_scores") else {
        panic!("scores");
    };
    assert_eq!(scores, &[0.9, 0.8], "best first");
    assert_eq!(calls_to(&b, "arcface"), [[2, 3, 112, 112]]);
    assert_released(&b).await;
}

#[tokio::test]
async fn a_photo_without_a_face_makes_no_recogniser_call_and_answers_empty_tensors() {
    let b = bed();
    let answer = b
        .flow
        .process(photo(&faces_plan("")))
        .await
        .expect("answer");
    assert_eq!(answer.branches["face_embed"], BranchState::Ok);
    assert!(calls_to(&b, "arcface").is_empty());
    assert!(
        !b.fake
            .events()
            .iter()
            .any(|e| matches!(e, Event::Launch { .. }))
    );
    assert_eq!(shape(output(&answer, "face_boxes").expect("boxes")), [0, 4]);
    assert_eq!(
        shape(output(&answer, "face_blur_sums").expect("blur")),
        [0, 3]
    );
    assert_eq!(
        shape(output(&answer, "face_embeddings").expect("vectors")),
        [0, 0]
    );
    assert_eq!(shape(output(&answer, "face_embedded").expect("flags")), [0]);
    assert_released(&b).await;
}

#[tokio::test]
async fn a_face_whose_landmarks_give_no_transform_is_not_embedded() {
    let b = bed();
    b.fake.set(|s| {
        let (anchor, score, ltrb, _) = fire(3, 3, 0.9);
        s.face_fires = vec![(anchor, score, ltrb, [0.0; 10])];
    });
    let answer = b
        .flow
        .process(photo(&faces_plan("")))
        .await
        .expect("answer");
    assert!(calls_to(&b, "arcface").is_empty());
    assert_eq!(
        output(&answer, "face_embedded"),
        Some(&Tensor::Bool {
            shape: vec![1],
            data: vec![false]
        })
    );
    assert_eq!(
        shape(output(&answer, "face_embeddings").expect("vectors")),
        [0, 0]
    );
    assert_eq!(
        shape(output(&answer, "face_blur_sums").expect("blur")),
        [1, 3]
    );
    assert_released(&b).await;
}

#[tokio::test]
async fn a_failed_detector_call_fails_faces_and_the_vectors_that_need_them() {
    let b = bed();
    b.fake.set(|s| {
        s.face_fires = vec![fire(3, 3, 0.9)];
        s.fail.insert(
            "scrfd".into(),
            CallError::Model {
                code: "INVALID_ARGUMENT".into(),
                message: "bad input".into(),
            },
        );
    });
    let answer = b
        .flow
        .process(photo(&faces_plan("")))
        .await
        .expect("answer");
    assert_eq!(answer.branches["faces"], BranchState::Error(MODEL_ERROR));
    assert_eq!(
        answer.branches["face_embed"],
        BranchState::Error(DEPENDENCY_FAILED)
    );
    assert!(answer.outputs.iter().all(|(n, _)| !n.starts_with("face")));
    assert!(calls_to(&b, "arcface").is_empty());
    assert_released(&b).await;
}

#[tokio::test]
async fn a_failed_recogniser_call_fails_the_vectors_only() {
    let b = bed();
    b.fake.set(|s| {
        s.face_fires = vec![fire(3, 3, 0.9)];
        s.fail.insert(
            "arcface".into(),
            CallError::Model {
                code: "INVALID_ARGUMENT".into(),
                message: "bad input".into(),
            },
        );
    });
    let answer = b
        .flow
        .process(photo(&faces_plan("")))
        .await
        .expect("answer");
    assert_eq!(answer.branches["faces"], BranchState::Ok);
    assert_eq!(
        answer.branches["face_embed"],
        BranchState::Error(MODEL_ERROR)
    );
    assert!(output(&answer, "face_boxes").is_some());
    assert!(output(&answer, "face_embeddings").is_none());
    assert_released(&b).await;
}

#[tokio::test]
async fn a_panic_in_host_work_fails_that_job_and_the_finish_thread_goes_on() {
    let fake = Fake::new(MIB, 1);
    let cancel = gpu_frame::CancelToken::new();
    let failed = fake
        .host(|| -> u32 { panic!("a decode of garbage") }, &cancel)
        .expect("queued")
        .wait()
        .await;
    assert!(matches!(failed, Err(gpu_frame::JobError::Panicked(m)) if m.contains("garbage")));
    let next = fake.host(|| 7u32, &cancel).expect("queued").wait().await;
    assert_eq!(
        next.expect("ran").expect("value"),
        7,
        "the one thread still serves"
    );
    cancel.cancel();
    let dropped = fake.host(|| 8u32, &cancel).expect("queued").wait().await;
    assert_eq!(dropped.err(), Some(gpu_frame::JobError::Cancelled));
}
