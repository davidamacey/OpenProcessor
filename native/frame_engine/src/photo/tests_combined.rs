//! What only the lanes together do, on the host fake: text held back by `if_detected`, and a
//! host-route photo with a converting profile, text and renditions.

use super::sources::{P3, PROGRESSIVE, with_icc};
use super::text::{BARS, RENDITIONS, text_plan};
use super::*;

/// Detect, then text late (64-pixel detector input), if class 16 was found.
fn late_text() -> String {
    r#"{"version": 1, "branches": ["detect", "ocr"], "crops": "none", "detector": {"model": "det", "input_size": 640, "letterbox_fill": 114}, "ocr": {"detector": "tdet", "recogniser": "trec", "batch": 2, "max_lines": 10, "det_side": 64, "det_limit": "max"}, "if_detected": {"ocr": [16]}}"#.into()
}

fn models_called(b: &Bed) -> Vec<String> {
    b.fake.calls().into_iter().map(|(m, _)| m).collect()
}

#[tokio::test]
async fn late_text_is_read_after_the_detector_found_its_class() {
    let b = bed();
    b.fake.set(|s| {
        s.detections = vec![[100.0, 100.0, 300.0, 300.0, 0.9, 16.0]];
        s.text_bars = BARS.to_vec();
    });
    let answer = b.flow.process(photo(&late_text())).await.expect("answer");
    assert_eq!(answer.branches["detect"], BranchState::Ok);
    assert_eq!(answer.branches["ocr"], BranchState::Ok);
    let events = b.fake.events();
    assert_eq!(
        events[0],
        Event::Decode {
            bytes: vec![3 * 640 * 640],
            fills: vec![Some(114)]
        },
        "the decode writes the detector's input only: the text input waits for its condition"
    );
    assert!(
        events.contains(&Event::Launch { writes: 1, blur: 0 }),
        "the late round writes the text input by a launch job"
    );
    let models = models_called(&b);
    assert_eq!(models[..2], ["det", "tdet"]);
    assert!(models[2..].iter().all(|m| m == "trec") && models.len() > 2);
    assert!(output(&answer, "ocr_texts").is_some());
    assert_released(&b).await;
}

#[tokio::test]
async fn late_text_does_not_run_when_its_class_is_missing() {
    let b = bed();
    b.fake.set(|s| {
        s.detections = vec![[100.0, 100.0, 300.0, 300.0, 0.9, 2.0]];
        s.text_bars = BARS.to_vec();
    });
    let answer = b.flow.process(photo(&late_text())).await.expect("answer");
    assert_eq!(answer.branches["detect"], BranchState::Ok);
    assert_eq!(answer.branches["ocr"], BranchState::NotRun);
    assert_eq!(models_called(&b), ["det"]);
    assert!(answer.outputs.iter().all(|(n, _)| !n.starts_with("ocr")));
    assert_released(&b).await;
}

#[tokio::test]
async fn a_host_route_photo_is_converted_read_and_rendered_from_its_upright_frame() {
    let b = bed();
    b.fake.set(|s| s.text_bars = BARS.to_vec());
    let image = with_icc(PROGRESSIVE, P3);
    let answer = b
        .flow
        .process(photo_of(image, &text_plan(10, RENDITIONS)))
        .await
        .expect("answer");
    assert_eq!(
        (answer.status.as_str(), answer.decode_route.as_str()),
        ("ok", "cpu_progressive")
    );
    assert_eq!(answer.branches["ocr"], BranchState::Ok);
    assert_eq!(answer.branches["derivatives"], BranchState::Ok);
    let events = b.fake.events();
    let gather = events.iter().position(|e| *e == Event::Gather);
    let renditions = events
        .iter()
        .position(|e| matches!(e, Event::Renditions(_)));
    assert!(
        matches!((gather, renditions), (Some(g), Some(r)) if g < r),
        "the colour step before the renditions: {events:?}"
    );
    assert_eq!(
        events.iter().find(|e| matches!(e, Event::Renditions(_))),
        Some(&Event::Renditions(vec![
            ((23, 15), false),
            ((23, 15), false)
        ])),
        "both sides hold the 23 x 15 upright frame: the frame itself"
    );
    assert!(output(&answer, "ocr_texts").is_some());
    assert_released(&b).await;
}
