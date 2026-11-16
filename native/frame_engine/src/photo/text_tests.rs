//! The text branch and the renditions in the photo flow on the host fake: the calls, the batches
//! by width, the tensors, a photo with no text, the line cap, each model failing, and the
//! renditions with a failed encode.

use super::fake::text_map;
use super::*;
use crate::branches::ocr;
use crate::wire::branch_status::DEPENDENCY_FAILED;
use crate::wire::plan::OcrSpec;

/// Text in a 64-pixel detector input, recogniser batches of 2; `extra` goes into the plan.
fn text_plan(max_lines: u32, extra: &str) -> String {
    format!(
        r#"{{"version": 1, "branches": ["ocr"], "crops": "none", "ocr": {{"detector": "tdet", "recogniser": "trec", "batch": 2, "max_lines": {max_lines}, "det_side": 64, "det_limit": "max"}}{extra}}}"#
    )
}

/// Two bars on one line and one below (scores 0.9, 0.8, 0.6), in map pixels.
const BARS: [(usize, usize, usize, usize, f32); 3] = [
    (30, 60, 10, 18, 0.9),
    (2, 26, 11, 19, 0.8),
    (4, 50, 34, 40, 0.6),
];

fn calls_to(b: &Bed, model: &str) -> Vec<[i64; 4]> {
    b.fake
        .calls()
        .into_iter()
        .filter(|(m, _)| m == model)
        .map(|(_, s)| s)
        .collect()
}

fn spec_of(plan: &str) -> OcrSpec {
    parse_plan(plan.as_bytes(), false)
        .expect("plan")
        .ocr
        .expect("ocr")
}

/// What the branch functions make of the fake's map: the detector input and the recogniser
/// call shapes, in order.
fn expected(plan: &str, bars: &[(usize, usize, usize, usize, f32)]) -> ([i64; 4], Vec<[i64; 4]>) {
    let spec = spec_of(plan);
    let (input, _) = ocr::det_canvas(&spec, 1000, 750).expect("canvas");
    let [_, _, h, w] = input.shape.map(|d| d as usize);
    let settled = ocr::settle(&[text_map(h, w, bars)], &spec, 1000, 750).expect("settled");
    let batches = ocr::line_batches(&settled, &spec).expect("batches");
    (input.shape, batches.iter().map(|b| b.input.shape).collect())
}

fn texts(a: &PhotoAnswer) -> Vec<String> {
    match output(a, "ocr_texts") {
        Some(Tensor::Bytes(items)) => items
            .iter()
            .map(|t| String::from_utf8(t.to_vec()).expect("utf-8"))
            .collect(),
        other => panic!("ocr_texts is {other:?}"),
    }
}

#[tokio::test]
async fn text_is_found_read_in_width_batches_and_answered_with_every_text_tensor() {
    let b = bed();
    b.fake.set(|s| s.text_bars = BARS.to_vec());
    let plan = text_plan(10, "");
    let answer = b.flow.process(photo(&plan)).await.expect("answer");
    assert_eq!(answer.branches["ocr"], BranchState::Ok);
    let (det, rec) = expected(&plan, &BARS);
    assert_eq!(calls_to(&b, "tdet"), [det]);
    assert!(rec.len() >= 2, "three lines in calls of two: {rec:?}");
    assert_eq!(
        calls_to(&b, "trec"),
        rec,
        "one call per width batch, in order"
    );
    let events = b.fake.events();
    assert_eq!(
        events[0],
        Event::Decode {
            bytes: vec![(det[1] * det[2] * det[3]) as usize],
            fills: vec![None]
        },
        "the detector input is written by the decode job"
    );
    assert!(events.iter().any(|e| matches!(e, Event::Warps(_))));
    for (name, want) in [
        ("ocr_quads", vec![3, 8]),
        ("ocr_texts", vec![3]),
        ("ocr_scores", vec![3]),
        ("ocr_char_min", vec![3]),
        ("ocr_det_scores", vec![3]),
        ("ocr_overflow", vec![1]),
    ] {
        assert_eq!(shape(output(&answer, name).expect(name)), want, "{name}");
    }
    let read = texts(&answer);
    assert!(read.iter().all(|t| t.len() == 1), "{read:?}");
    assert_eq!(
        output(&answer, "ocr_overflow"),
        Some(&Tensor::Int32 {
            shape: vec![1],
            data: vec![0]
        })
    );
    assert_released(&b).await;
}

#[tokio::test]
async fn a_photo_without_text_makes_only_the_detector_call() {
    let b = bed();
    let plan = text_plan(10, "");
    let answer = b.flow.process(photo(&plan)).await.expect("answer");
    assert_eq!(answer.branches["ocr"], BranchState::Ok);
    assert_eq!(calls_to(&b, "tdet").len(), 1);
    assert!(calls_to(&b, "trec").is_empty());
    assert!(
        !b.fake
            .events()
            .iter()
            .any(|e| matches!(e, Event::Launch { .. }))
    );
    assert_eq!(shape(output(&answer, "ocr_quads").expect("quads")), [0, 8]);
    assert_eq!(texts(&answer), Vec::<String>::new());
    assert_released(&b).await;
}

#[tokio::test]
async fn lines_past_max_lines_are_counted_not_read() {
    let b = bed();
    b.fake.set(|s| s.text_bars = BARS.to_vec());
    let plan = text_plan(2, "");
    let answer = b.flow.process(photo(&plan)).await.expect("answer");
    assert_eq!(
        output(&answer, "ocr_overflow"),
        Some(&Tensor::Int32 {
            shape: vec![1],
            data: vec![1]
        })
    );
    assert_eq!(shape(output(&answer, "ocr_texts").expect("texts")), [2]);
    let read: i64 = calls_to(&b, "trec").iter().map(|s| s[0]).sum();
    assert_eq!(read, 2);
}

#[tokio::test]
async fn a_failed_text_model_fails_ocr_alone_and_answers_no_text_tensor() {
    for model in ["tdet", "trec"] {
        let b = bed();
        b.fake.set(|s| {
            s.text_bars = BARS.to_vec();
            s.fail.insert(
                model.into(),
                CallError::Model {
                    code: "INVALID_ARGUMENT".into(),
                    message: "bad".into(),
                },
            );
        });
        let plan = text_plan(10, "").replace(r#"["ocr"]"#, r#"["ocr", "detect"]"#).replace(
            r#""crops": "none","#,
            r#""crops": "none", "detector": {"model": "det", "input_size": 640, "letterbox_fill": 114},"#,
        );
        let answer = b.flow.process(photo(&plan)).await.expect("answer");
        assert_eq!(answer.status, "ok", "{model}");
        assert_eq!(
            answer.branches["ocr"],
            BranchState::Error(MODEL_ERROR),
            "{model}"
        );
        assert_eq!(answer.branches["detect"], BranchState::Ok, "{model}");
        assert!(
            answer.outputs.iter().all(|(n, _)| !n.starts_with("ocr_")),
            "{model}: no text tensor"
        );
        assert_ne!(
            answer.branches["ocr"],
            BranchState::Error(DEPENDENCY_FAILED)
        );
        assert_released(&b).await;
    }
}

const RENDITIONS: &str = r#", "derivatives": [{"name": "thumb", "side": 256, "quality": 80}, {"name": "full", "side": 4000, "quality": 90}]"#;

#[tokio::test]
async fn renditions_are_made_after_the_round_two_calls_and_answered_in_plan_order() {
    let b = bed();
    b.fake.set(|s| s.text_bars = BARS.to_vec());
    let plan = text_plan(10, RENDITIONS);
    let answer = b.flow.process(photo(&plan)).await.expect("answer");
    assert_eq!(answer.branches["derivatives"], BranchState::Ok);
    let events = b.fake.events();
    let job = events
        .iter()
        .position(|e| matches!(e, Event::Renditions(_)))
        .expect("a rendition job");
    let last_call = events
        .iter()
        .rposition(|e| matches!(e, Event::Call { .. }))
        .expect("calls");
    assert!(job > last_call, "after the recogniser calls went out");
    assert_eq!(
        events[job],
        Event::Renditions(vec![((256, 192), true), ((1000, 750), false)]),
        "a thumbnail resample, and the frame itself when it fits"
    );
    match output(&answer, "derivative_jpegs") {
        Some(Tensor::Bytes(items)) => assert_eq!(
            items,
            &[
                Bytes::from_static(b"jpeg (256, 192) q80"),
                Bytes::from_static(b"jpeg (1000, 750) q90")
            ]
        ),
        other => panic!("derivative_jpegs is {other:?}"),
    }
    assert_released(&b).await;
}

#[tokio::test]
async fn one_failed_encode_fails_the_photo_under_the_fault_rule() {
    let b = bed();
    b.fake
        .set(|s| s.encode_fault = Some(GpuFault::Failed("nvJPEG encode: refused".into())));
    let failed = b
        .flow
        .process(photo(&text_plan(10, RENDITIONS)))
        .await
        .expect_err("the photo fails");
    assert!(
        matches!(&failed, PipelineError::Engine(m) if m.contains("refused")),
        "{failed:?}"
    );
    assert_released(&b).await;

    let b = bed();
    b.fake
        .set(|s| s.encode_fault = Some(GpuFault::OutOfMemory("cap".into())));
    let answer = b
        .flow
        .process(photo(&text_plan(10, RENDITIONS)))
        .await
        .expect("an answer");
    assert_eq!(answer.status, "gpu_memory");
    assert_released(&b).await;
}

#[tokio::test]
async fn the_budget_reserves_the_renditions_scratch() {
    // 1000 x 750: frame 2,250,000 bytes; a 4000-side rendition is the frame, a 512 one has
    // 512 columns: 750 x 512 x 12 = 4,608,000 bytes of scratch
    let plan = text_plan(10, r#", "derivatives": [{"name": "m", "side": 512}]"#);
    let b = bed_with(64 * MIB, 4 * MIB, 6_000_000, 4);
    assert_eq!(
        b.flow.process(photo(&plan)).await.expect("answer").status,
        "over_budget",
        "2,250,000 + 4,608,000 > 6,000,000"
    );
    let b = bed_with(64 * MIB, 4 * MIB, 7_000_000, 4);
    assert_eq!(
        b.flow.process(photo(&plan)).await.expect("answer").status,
        "ok"
    );
}
