//! The photo flow on the host fake (`fake.rs`): stages, bounds, cancellation, failures. What the
//! card does (pixels, nvJPEG, kernels) is owed to the GPU holds.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use gpu_frame::CallError;
use prometheus_client::metrics::gauge::Gauge;

use super::budget::GpuBudget;
use super::fake::{Event, Fake};
use super::*;
use crate::admission::RequestClock;
use crate::colour::ColourStep;
use crate::pipeline::RequestState;
use crate::wire::branch_status::MODEL_ERROR;
use crate::wire::bytes_tensor::{pack, unpack};
use crate::wire::parse_plan;

const MIB: usize = 1 << 20;

/// A baseline JPEG's markers (no entropy-coded image): `w x h`, 3 components, an EXIF
/// orientation when not 1, a scan of two bytes and the end marker.
fn jpeg(w: u16, h: u16, orientation: u16) -> Bytes {
    let mut b = vec![0xFF, 0xD8];
    if orientation != 1 {
        let mut exif = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec();
        exif.extend_from_slice(&orientation.to_le_bytes());
        exif.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        b.extend_from_slice(&[0xFF, 0xE1]);
        b.extend_from_slice(&u16::try_from(exif.len() + 2).expect("short").to_be_bytes());
        b.extend_from_slice(&exif);
    }
    b.extend_from_slice(&[0xFF, 0xC0, 0, 17, 8]);
    b.extend_from_slice(&h.to_be_bytes());
    b.extend_from_slice(&w.to_be_bytes());
    b.extend_from_slice(&[3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    b.extend_from_slice(&[0xFF, 0xDA, 0, 12, 3, 1, 0, 2, 0x11, 3, 0x11, 0, 0x3F, 0]);
    b.extend_from_slice(&[0x12, 0x34, 0xFF, 0xD9]);
    Bytes::from(b)
}

const INGEST: &str = r#"{"version": 1, "branches": ["detect", "embed_frame", "embed_crops"], "crops": "inline", "detector": {"model": "det", "input_size": 640, "letterbox_fill": 114}, "embed": {"model": "pe", "size": 336}, "crop_embed": {"model": "pe_crop", "size": 224, "batch": 32}}"#;
const FRAME: &str = r#"{"version": 1, "branches": ["detect", "embed_frame"], "crops": "none", "detector": {"model": "det", "input_size": 640, "letterbox_fill": 114}, "embed": {"model": "pe", "size": 336}}"#;

fn ingest_with(batch: u32, classes: &str) -> String {
    INGEST
        .replace(r#""batch": 32"#, &format!(r#""batch": {batch}"#))
        .replace(
            r#""crops": "inline""#,
            &format!(r#""crops": "inline"{classes}"#),
        )
}

/// Three detections in the canvas of a 1000 x 750 photo (classes 2, 7, 16).
const ROWS: [[f32; 6]; 3] = [
    [101.37, 133.71, 300.5, 401.25, 0.91, 2.0],
    [-20.25, 70.0, 30.5, 90.0, 0.5, 7.9],
    [600.0, 500.0, 700.0, 600.0, 0.25, 16.0],
];

/// The fake recogniser's classes: blank, `a` to `e`, then the space.
const DICTIONARY: &str = "a\nb\nc\nd\ne\n";

struct Bed {
    fake: Arc<Fake>,
    flow: PhotoFlow<Fake, GpuBudget>,
}

fn bed_with(region: usize, window: usize, budget: u64, workers: u32) -> Bed {
    let fake = Fake::new(region, 2);
    fake.set(|s| s.detections = ROWS.to_vec());
    let flow = PhotoFlow {
        device: Arc::clone(&fake),
        budget: GpuBudget::new(workers, budget, 0.2, Gauge::default()),
        settings: PhotoSettings {
            min_megapixels: 0.0,
            max_megapixels: 100.0,
            budget_bytes: budget,
            budget_wait: Duration::from_millis(200),
            slab_wait: Duration::from_millis(100),
            window,
        },
        metrics: Metrics::new(),
        colour: ColourStep::new(2, Gauge::default()),
        dictionary: Arc::new(CtcDecoder::from_text(DICTIONARY)),
    };
    Bed { fake, flow }
}

fn bed() -> Bed {
    bed_with(64 * MIB, 4 * MIB, 512 * MIB as u64, 4)
}

fn photo_of(image: Bytes, plan: &str) -> Photo {
    Photo {
        image,
        plan: parse_plan(plan.as_bytes(), false).expect("plan"),
        boxes: None,
        request: Arc::new(RequestState::new(RequestClock::start(Instant::now(), 30.0))),
    }
}

fn photo(plan: &str) -> Photo {
    photo_of(jpeg(1000, 750, 1), plan)
}

fn output<'a>(a: &'a PhotoAnswer, name: &str) -> Option<&'a Tensor> {
    a.outputs.iter().find(|(n, _)| n == name).map(|(_, t)| t)
}

fn shape(t: &Tensor) -> Vec<i64> {
    match t {
        Tensor::Bool { shape, .. }
        | Tensor::Int32 { shape, .. }
        | Tensor::Int64 { shape, .. }
        | Tensor::Fp32 { shape, .. }
        | Tensor::Fp64 { shape, .. } => shape.clone(),
        Tensor::Bytes(items) => vec![items.len() as i64],
    }
}

/// Every count back at zero: frames, region bytes, budget bytes and worker slots.
async fn assert_released(b: &Bed) {
    // jobs still running hold the frame to their own wait; give them a moment, bounded
    for _ in 0..200 {
        if b.fake.frames.load(Ordering::SeqCst) == 0 && b.fake.slab.usage().in_use == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(b.fake.frames.load(Ordering::SeqCst), 0, "frames alive");
    assert_eq!(b.fake.slab.usage().in_use, 0, "region bytes held");
    assert_eq!(b.flow.budget.bytes_in_use(), 0, "budget bytes held");
    assert_eq!(b.flow.budget.workers_in_use(), 0, "worker slots held");
}

#[tokio::test]
async fn the_ingest_plan_answers_every_output_in_wire_order_and_releases_everything() {
    let b = bed();
    let answer = b.flow.process(photo(INGEST)).await.expect("answer");
    assert_eq!(
        (answer.status.as_str(), answer.decode_route.as_str()),
        ("ok", "gpu")
    );
    let names: Vec<&str> = answer.outputs.iter().map(|(n, _)| n.as_str()).collect();
    let plan = parse_plan(INGEST.as_bytes(), false).expect("plan");
    let mut wanted = vec!["size"];
    wanted.extend(plan.expected_outputs());
    assert_eq!(names, wanted);
    assert_eq!(
        output(&answer, "size"),
        Some(&Tensor::Int32 {
            shape: vec![2],
            data: vec![1000, 750]
        })
    );
    assert_eq!(shape(output(&answer, "det_boxes").expect("boxes")), [3, 4]);
    assert_eq!(shape(output(&answer, "blur_sums").expect("blur")), [4, 3]);
    assert_eq!(
        shape(output(&answer, "whole_embedding").expect("vector")),
        [8]
    );
    assert_eq!(
        shape(output(&answer, "crop_embeddings").expect("crops")),
        [3, 8]
    );
    assert_eq!(shape(output(&answer, "crop_jpegs").expect("jpegs")), [3]);
    assert!(answer.branches.values().all(|s| *s == BranchState::Ok));
    assert_eq!(answer.branches.len(), 3);
    assert_released(&b).await;
}

#[tokio::test]
async fn the_hand_offs_come_in_the_python_order_and_each_stage_is_observed() {
    let b = bed();
    b.flow.process(photo(INGEST)).await.expect("answer");
    let events = b.fake.events();
    let kinds: Vec<String> = events
        .iter()
        .map(|e| match e {
            Event::Decode { .. } => "decode".into(),
            Event::Launch { .. } => "launch".into(),
            Event::Crop(_) => "crop".into(),
            Event::Warps(_) => "warps".into(),
            Event::ColourTable(_) => "colour table".into(),
            Event::Gather => "gather".into(),
            Event::Renditions(_) => "renditions".into(),
            Event::Call { model, .. } => format!("call {model}"),
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "decode",
            "call det",
            "call pe",
            "launch",
            "call pe_crop",
            "crop",
            "crop",
            "crop"
        ]
    );
    // round 1: the canvas on the detector's fill, then the frame square, one decode job
    assert_eq!(
        events[0],
        Event::Decode {
            bytes: vec![3 * 640 * 640, 3 * 336 * 336],
            fills: vec![Some(114), None]
        }
    );
    // round 2: one launch for the crop squares and the blur sums of the frame and each box
    assert_eq!(events[3], Event::Launch { writes: 1, blur: 4 });
    let text = b.flow.metrics.render();
    // the ingest plan runs no face, text or rendition branch
    let other = ["face_", "ocr_", "derivative_"];
    for stage in clock::ALL
        .iter()
        .filter(|s| !other.iter().any(|o| s.starts_with(o)))
    {
        assert!(
            text.contains(&format!(
                "op_frame_engine_stage_observations_total{{stage=\"{stage}\"}} 1"
            )),
            "{stage}"
        );
    }
    assert!(text.contains("op_frame_engine_images_total{status=\"ok\",decode_route=\"gpu\"} 1"));
    assert!(text.contains("op_frame_engine_budget_waits_total{outcome=\"admitted\"} 1"));
}

#[tokio::test]
async fn crops_go_to_the_encoder_in_chunks_of_the_plan_batch() {
    let b = bed();
    b.fake.set(|s| {
        let more = ROWS.iter().chain(ROWS.iter()).copied();
        s.detections = more.collect();
    });
    let answer = b
        .flow
        .process(photo(&ingest_with(4, "")))
        .await
        .expect("answer");
    let crops: Vec<[i64; 4]> = b
        .fake
        .calls()
        .into_iter()
        .filter(|(m, _)| m == "pe_crop")
        .map(|(_, s)| s)
        .collect();
    assert_eq!(crops, [[4, 3, 224, 224], [2, 3, 224, 224]]);
    assert_eq!(
        shape(output(&answer, "crop_embeddings").expect("crops")),
        [6, 8]
    );
    assert_released(&b).await;
}

#[tokio::test]
async fn embed_classes_picks_which_detections_get_a_vector() {
    let b = bed();
    let plan = ingest_with(32, r#", "embed_classes": [2, 16]"#);
    let answer = b.flow.process(photo(&plan)).await.expect("answer");
    assert_eq!(
        output(&answer, "crop_embedded"),
        Some(&Tensor::Bool {
            shape: vec![3],
            data: vec![true, false, true]
        })
    );
    assert_eq!(
        shape(output(&answer, "crop_embeddings").expect("crops")),
        [2, 8]
    );
    assert_eq!(
        shape(output(&answer, "crop_jpegs").expect("jpegs")),
        [3],
        "every detection's JPEG"
    );
}

#[tokio::test]
async fn a_photo_without_detections_answers_empty_rows_and_an_empty_jpeg_list() {
    let b = bed();
    b.fake.set(|s| s.detections.clear());
    let answer = b.flow.process(photo(INGEST)).await.expect("answer");
    assert_eq!(shape(output(&answer, "det_boxes").expect("boxes")), [0, 4]);
    assert_eq!(
        shape(output(&answer, "crop_embeddings").expect("crops")),
        [0, 0]
    );
    assert_eq!(output(&answer, "crop_jpegs"), Some(&Tensor::Bytes(vec![])));
    assert_eq!(
        shape(output(&answer, "blur_sums").expect("blur")),
        [1, 3],
        "the frame's own row"
    );
    assert!(!b.fake.calls().iter().any(|(m, _)| m == "pe_crop"));
}

#[tokio::test]
async fn the_frame_plan_sends_no_crop_call_and_encodes_no_jpeg() {
    let b = bed();
    let answer = b.flow.process(photo(FRAME)).await.expect("answer");
    let models: Vec<String> = b.fake.calls().into_iter().map(|(m, _)| m).collect();
    assert_eq!(models, ["det", "pe"]);
    assert!(output(&answer, "crop_jpegs").is_none());
    assert!(
        b.fake
            .events()
            .contains(&Event::Launch { writes: 0, blur: 4 })
    );
    assert_released(&b).await;
}

#[tokio::test]
async fn every_model_call_carries_the_request_deadline() {
    let b = bed();
    let p = photo(INGEST);
    let deadline = tokio::time::Instant::from_std(p.request.clock.deadline);
    b.flow.process(p).await.expect("answer");
    for e in b.fake.events() {
        if let Event::Call { deadline: d, .. } = e {
            assert_eq!(d, deadline);
        }
    }
}

#[tokio::test]
async fn the_window_collects_the_photos_oldest_call_before_the_next_chunk() {
    // a window of one crop square: each chunk is staged only after the call before it was
    // collected, so a crop call never finds more than its own chunk in the region
    let chunk = 3 * 224 * 224;
    let b = bed_with(64 * MIB, chunk, 512 * MIB as u64, 4);
    b.flow
        .process(photo(&ingest_with(1, "")))
        .await
        .expect("answer");
    let mut crops = 0;
    for e in b.fake.events() {
        if let Event::Call {
            model,
            region_in_use,
            ..
        } = e
            && model == "pe_crop"
        {
            crops += 1;
            assert_eq!(
                region_in_use,
                chunk.next_multiple_of(256),
                "only this call's chunk"
            );
        }
    }
    assert_eq!(crops, 3);
    let launches = b
        .fake
        .events()
        .iter()
        .filter(|e| matches!(e, Event::Launch { .. }))
        .count();
    assert_eq!(
        launches, 3,
        "a chunk that does not fit beside the others goes in a launch of its own"
    );
    assert_released(&b).await;
}

#[tokio::test]
async fn a_full_region_is_an_overload_after_its_wait_and_releases_the_budget() {
    let b = bed();
    let held = b
        .fake
        .slab
        .take(64 * MIB, Duration::ZERO)
        .await
        .expect("all of it");
    let failed = b.flow.process(photo(INGEST)).await.expect_err("overload");
    assert!(
        matches!(&failed, PipelineError::Overload(o) if o.cause == "slab"),
        "{failed:?}"
    );
    b.fake.slab.give(held, 64 * MIB);
    assert_released(&b).await;
}

#[tokio::test]
async fn an_input_larger_than_the_whole_region_answers_over_budget() {
    let b = bed_with(MIB, MIB, 512 * MIB as u64, 4);
    let answer = b.flow.process(photo(INGEST)).await.expect("answer");
    assert_eq!(
        (answer.status.as_str(), answer.decode_route.as_str()),
        ("over_budget", "none")
    );
    assert!(answer.outputs.is_empty());
    assert_released(&b).await;
}

#[tokio::test]
async fn a_budget_wait_past_its_bound_is_an_overload_and_a_photo_above_the_budget_is_over_budget() {
    // one worker slot: the second photo waits for it, and the wait is bounded
    let b = bed_with(64 * MIB, 4 * MIB, 512 * MIB as u64, 1);
    let gate = Arc::new(AtomicBool::new(false));
    b.fake.set(|s| s.decode_gate = Some(Arc::clone(&gate)));
    let first = {
        let flow = &b.flow;
        async move { flow.process(photo(INGEST)).await }
    };
    let second = async {
        tokio::time::sleep(Duration::from_millis(20)).await;
        let started = Instant::now();
        let r = b.flow.process(photo(INGEST)).await;
        gate.store(true, Ordering::SeqCst);
        (r, started.elapsed())
    };
    let (one, (two, waited)) = tokio::join!(first, second);
    one.expect("the first photo is served");
    let two = two.expect_err("overload");
    assert!(
        matches!(&two, PipelineError::Overload(o) if o.cause == "budget"),
        "{two:?}"
    );
    assert!(waited >= Duration::from_millis(200) && waited < Duration::from_secs(2));
    assert_released(&b).await;
    // 1000 x 750 x 3 bytes (2.25 MB) against a 2 MB budget
    let small = bed_with(64 * MIB, 4 * MIB, 2 * MIB as u64, 4);
    let answer = small.flow.process(photo(INGEST)).await.expect("answer");
    assert_eq!(answer.status, "over_budget");
    assert_eq!(small.fake.events(), [], "nothing was decoded");
}

#[tokio::test]
async fn a_rotated_photo_reserves_its_oriented_copy_and_answers_its_oriented_size() {
    let b = bed();
    let answer = b
        .flow
        .process(photo_of(jpeg(1000, 750, 6), INGEST))
        .await
        .expect("answer");
    assert_eq!(
        output(&answer, "size"),
        Some(&Tensor::Int32 {
            shape: vec![2],
            data: vec![750, 1000]
        })
    );
    // 2 x 2.25 MB for the frame and its copy is above 4 MB; 1 x is not
    let small = bed_with(64 * MIB, 4 * MIB, 4 * MIB as u64, 4);
    let answer = small
        .flow
        .process(photo_of(jpeg(1000, 750, 6), INGEST))
        .await
        .expect("answer");
    assert_eq!(answer.status, "over_budget");
    let answer = small.flow.process(photo(INGEST)).await.expect("answer");
    assert_eq!(answer.status, "ok");
}

#[tokio::test]
async fn a_cancelled_photo_is_dropped_at_the_decode_hand_off_and_holds_nothing() {
    let b = bed();
    let p = photo(INGEST);
    p.request.cancel();
    let failed = b.flow.process(p).await.expect_err("cancelled");
    assert!(
        matches!(&failed, PipelineError::Engine(m) if m.contains("cancelled")),
        "{failed:?}"
    );
    assert_released(&b).await;
}

#[tokio::test]
async fn a_photo_dropped_while_its_calls_are_in_flight_gives_every_slice_back() {
    let b = bed();
    b.fake.set(|s| s.call_hold = Some(Duration::from_secs(30)));
    let p = photo(INGEST);
    let request = Arc::clone(&p.request);
    let gave_up = tokio::time::timeout(Duration::from_millis(100), b.flow.process(p)).await;
    assert!(gave_up.is_err(), "the calls were still running");
    request.cancel();
    assert_released(&b).await;
}

#[tokio::test]
async fn a_photo_dropped_while_its_decode_runs_releases_once_the_lane_is_done() {
    let b = bed();
    let gate = Arc::new(AtomicBool::new(false));
    b.fake.set(|s| s.decode_gate = Some(Arc::clone(&gate)));
    let p = photo(INGEST);
    let request = Arc::clone(&p.request);
    let gave_up = tokio::time::timeout(Duration::from_millis(50), b.flow.process(p)).await;
    assert!(gave_up.is_err());
    request.cancel();
    let held = b.flow.budget.workers_in_use();
    // opened before the assertion: a failed one must not leave the lane blocked
    gate.store(true, Ordering::SeqCst);
    assert_eq!(held, 1, "the running decode holds the frame's permit");
    assert_released(&b).await;
}

#[tokio::test]
async fn a_rejected_detector_call_fails_detect_and_embed_crops_only() {
    let b = bed();
    b.fake.set(|s| {
        s.fail.insert(
            "det".into(),
            CallError::Model {
                code: "Internal".into(),
                message: "boom".into(),
            },
        );
    });
    let answer = b.flow.process(photo(INGEST)).await.expect("answer");
    assert_eq!(answer.status, "ok");
    assert_eq!(answer.branches["detect"], BranchState::Error(MODEL_ERROR));
    assert_eq!(answer.branches["embed_frame"], BranchState::Ok);
    assert_eq!(
        answer.branches["embed_crops"],
        BranchState::Error(DEPENDENCY_FAILED)
    );
    let names: Vec<&str> = answer.outputs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        ["size", "whole_embedding"],
        "no crop JPEGs without rectangles"
    );
    let text = b.flow.metrics.render();
    assert!(text.contains("model=\"det\",code=\"StatusCode.INTERNAL\",reason=\"rejected\""));
    assert_released(&b).await;
}

#[tokio::test]
async fn each_encoder_fails_alone() {
    let b = bed();
    b.fake.set(|s| {
        s.fail.insert(
            "pe".into(),
            CallError::Model {
                code: "Internal".into(),
                message: "x".into(),
            },
        );
    });
    let answer = b.flow.process(photo(INGEST)).await.expect("answer");
    assert_eq!(
        answer.branches["embed_frame"],
        BranchState::Error(MODEL_ERROR)
    );
    assert_eq!(answer.branches["embed_crops"], BranchState::Ok);
    assert!(output(&answer, "whole_embedding").is_none());
    assert_eq!(
        shape(output(&answer, "crop_embeddings").expect("crops")),
        [3, 8]
    );
    let b = bed();
    b.fake.set(|s| {
        s.fail.insert(
            "pe_crop".into(),
            CallError::Model {
                code: "Internal".into(),
                message: "x".into(),
            },
        );
    });
    let answer = b.flow.process(photo(INGEST)).await.expect("answer");
    assert_eq!(
        answer.branches["embed_crops"],
        BranchState::Error(MODEL_ERROR)
    );
    assert_eq!(answer.branches["embed_frame"], BranchState::Ok);
    assert!(output(&answer, "crop_embeddings").is_none());
    assert_eq!(shape(output(&answer, "crop_jpegs").expect("jpegs")), [3]);
    assert_released(&b).await;
}

#[tokio::test]
async fn a_gone_model_server_fails_the_photo_and_asks_for_a_probe() {
    for e in [
        CallError::Unavailable("reset".into()),
        CallError::Deadline,
        CallError::RegionUnknown("gone".into()),
    ] {
        let b = bed();
        let probe = e.suspects_server();
        b.fake.set(|s| {
            s.fail.insert("det".into(), e.clone());
        });
        let failed = b
            .flow
            .process(photo(INGEST))
            .await
            .expect_err("unavailable");
        assert!(
            matches!(failed, PipelineError::ModelServerUnavailable(_)),
            "{e:?}"
        );
        assert_eq!(
            b.fake.wakes.load(Ordering::SeqCst),
            usize::from(probe),
            "{e:?}"
        );
        assert_released(&b).await;
    }
}

#[tokio::test]
async fn decode_and_kernel_faults_answer_as_the_python_engine_does() {
    let b = bed();
    b.fake.set(|s| s.refuse = Some("decoder"));
    let answer = b.flow.process(photo(INGEST)).await.expect("answer");
    assert_eq!(
        (answer.status.as_str(), answer.decode_route.as_str()),
        ("decoder", "none")
    );
    assert!(
        answer
            .branches
            .values()
            .all(|s| *s == BranchState::Error(IMAGE_FAILED))
    );
    assert_released(&b).await;
    for (fault, want) in [
        (GpuFault::OutOfMemory("cap".into()), "gpu_memory"),
        (GpuFault::ContextLost("700".into()), "lost"),
        (GpuFault::Failed("launch".into()), "engine"),
    ] {
        for at_launch in [false, true] {
            let b = bed();
            b.fake.set(|s| {
                if at_launch {
                    s.launch_fault = Some(fault.clone());
                } else {
                    s.decode_fault = Some(fault.clone());
                }
            });
            let got = b.flow.process(photo(INGEST)).await;
            match (want, got) {
                ("gpu_memory", Ok(a)) => assert_eq!(a.status, "gpu_memory"),
                ("lost", Err(PipelineError::ContextLost(_)))
                | ("engine", Err(PipelineError::Engine(_))) => {}
                (want, got) => panic!("{want}: {got:?}"),
            }
            assert_released(&b).await;
        }
    }
    let b = bed();
    b.fake.lost.store(true, Ordering::SeqCst);
    assert!(matches!(
        b.flow.process(photo(INGEST)).await,
        Err(PipelineError::ContextLost(_))
    ));
    assert_eq!(b.fake.events(), [], "a lost context starts nothing");
}

#[tokio::test]
async fn a_panic_in_a_lane_job_fails_that_photo_and_the_lane_serves_the_next() {
    let fake = Fake::new(64 * MIB, 1);
    fake.set(|s| {
        s.detections = ROWS.to_vec();
        s.panic_in_decode = true;
    });
    let b = Bed {
        flow: PhotoFlow {
            device: Arc::clone(&fake),
            ..bed().flow
        },
        fake,
    };
    let failed = b.flow.process(photo(INGEST)).await.expect_err("panicked");
    assert!(
        matches!(&failed, PipelineError::Engine(m) if m.contains("a decode blew up")),
        "{failed:?}"
    );
    b.fake.set(|s| s.panic_in_decode = false);
    assert_eq!(
        b.flow.process(photo(INGEST)).await.expect("served").status,
        "ok"
    );
    assert_released(&b).await;
}

#[tokio::test]
async fn photos_no_decoder_takes_answer_without_a_decode() {
    let b = bed();
    for (image, status) in [
        (Bytes::from_static(b"\x89PNG\r\n\x1a\n...."), "decoder"),
        (Bytes::from_static(b"\xff\xd8\xff\xd9"), "corrupt"),
        (jpeg(12000, 9000, 1), "huge"),
    ] {
        let answer = b
            .flow
            .process(photo_of(image, INGEST))
            .await
            .expect("answer");
        assert_eq!(
            (answer.status.as_str(), answer.decode_route.as_str()),
            (status, "none")
        );
    }
    assert_eq!(b.fake.events(), []);
}

#[test]
fn every_branch_the_contract_builds_is_built_here() {
    let text = include_str!("../../../../contracts/json/frame_engine_v1.json");
    let c: serde_json::Value = serde_json::from_str(text).expect("the contract");
    let mut want: Vec<&str> = c["built_branches"]
        .as_array()
        .expect("built_branches")
        .iter()
        .map(|b| b.as_str().expect("a name"))
        .collect();
    let mut built = crate::branches::BUILT.to_vec();
    want.sort_unstable();
    built.sort_unstable();
    assert_eq!(built, want);
}

#[tokio::test]
async fn the_crop_jpegs_are_packed_as_bytes_elements() {
    let b = bed();
    let answer = b.flow.process(photo(INGEST)).await.expect("answer");
    let Some(Tensor::Bytes(items)) = output(&answer, "crop_jpegs") else {
        panic!("crop_jpegs is BYTES");
    };
    let packed = pack(items);
    // each element: a 4-byte little-endian length, then the bytes
    let first = &items[0];
    assert_eq!(
        &packed[..4],
        &u32::try_from(first.len()).expect("small").to_le_bytes()
    );
    assert_eq!(&packed[4..4 + first.len()], &first[..]);
    assert_eq!(unpack(&packed).expect("unpack"), *items);
    assert_eq!(
        &items[0][..],
        b"jpeg [158, 84, 470, 502]",
        "in detection order"
    );
}

#[path = "tests_combined.rs"]
mod combined;
#[path = "faces_tests.rs"]
mod faces;
#[path = "host_tests.rs"]
mod host;
#[path = "tests_late.rs"]
mod late_round;
#[path = "tests_sources.rs"]
mod sources;
#[path = "text_tests.rs"]
mod text;
