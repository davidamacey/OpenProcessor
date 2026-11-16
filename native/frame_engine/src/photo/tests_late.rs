//! The late round of `if_detected` branches (`late.rs`) on the host fake.

use super::*;

const FACES_LATE: &str = r#"{"version": 1, "branches": ["detect", "faces", "face_embed"], "crops": "none", "detector": {"model": "det", "input_size": 640, "letterbox_fill": 114}, "faces": {"detector": "scrfd", "input_size": 640, "recogniser": "arcface", "batch": 32, "score_min": 0.5, "nms": 0.4, "max_faces": 50}, "if_detected": {"faces": [0, 16]}}"#;

#[test]
fn a_conditional_branch_and_what_needs_it_run_late_and_follow_the_detectors_classes() {
    use crate::photo::late::{Verdict, dependents, late, verdict};
    let plan = parse_plan(FACES_LATE.as_bytes(), false).expect("plan");
    assert_eq!(dependents("faces"), ["face_embed"]);
    assert_eq!(dependents("detect"), ["embed_crops"]);
    let built = [
        "detect",
        "embed_frame",
        "embed_crops",
        "faces",
        "face_embed",
    ];
    assert_eq!(late(&plan, &built), ["faces", "face_embed"]);
    assert_eq!(
        late(&plan, &crate::branches::BUILT),
        ["faces", "face_embed"],
        "faces is built"
    );
    assert_eq!(verdict(&plan, "faces", Some(&[2, 16])), Some(Verdict::Run));
    assert_eq!(
        verdict(&plan, "faces", Some(&[2, 7])),
        Some(Verdict::NotRun)
    );
    assert_eq!(verdict(&plan, "faces", Some(&[])), Some(Verdict::NotRun));
    assert_eq!(
        verdict(&plan, "faces", None),
        Some(Verdict::DependencyFailed)
    );
    assert_eq!(
        verdict(&plan, "face_embed", Some(&[0])),
        None,
        "follows faces"
    );
}

/// A photo's [`Work`] on the fake, with its frame from a decode with no writes.
async fn late_bed<'a>(
    b: &'a Bed,
    clock: &'a mut StageClock,
) -> (Work<'a, Fake>, Arc<super::fake::Frame>) {
    let request = Arc::new(RequestState::new(RequestClock::start(Instant::now(), 30.0)));
    let calls = PhotoCalls::new(
        Arc::clone(&b.fake),
        Arc::clone(&b.flow.metrics),
        b.flow.settings.window,
        b.flow.settings.slab_wait,
        tokio::time::Instant::from_std(request.clock.deadline),
    );
    let frame = b.fake.frame();
    let work = Work {
        clock,
        calls,
        answers: Answers {
            states: BTreeMap::new(),
            stopped: Vec::new(),
            out: Vec::new(),
        },
        device: Arc::clone(&b.fake),
        cancel: request.cancel_token().clone(),
        request,
        size: (1000, 750),
    };
    (work, frame)
}

fn late_call(owner: &'static str, n: i64) -> crate::photo::late::LateCall<'static> {
    let input = crate::branches::InputSpec {
        name: "images",
        shape: [n, 3, 640, 640],
    };
    crate::photo::late::LateCall {
        call: (owner, "faces", "scrfd", input, &["output"]),
        fill: Some(0),
        resamples: Vec::new(),
        stage: "face_letterbox",
    }
}

#[tokio::test]
async fn a_late_round_writes_its_inputs_in_launches_within_the_window_and_releases_them() {
    let b = bed_with(64 * MIB, 4 * MIB, 512 * MIB as u64, 4);
    let mut clock = StageClock::new(
        Arc::new(RequestState::new(RequestClock::start(Instant::now(), 30.0))),
        Arc::clone(&b.flow.metrics),
    );
    let (mut work, frame) = late_bed(&b, &mut clock).await;
    // five inputs of 1.2 MB in a 4 MiB window: three fit together; with their calls in flight
    // each later input is staged after the photo's oldest call is collected, one at a time
    let calls = (0..5).map(|_| late_call("faces", 1)).collect();
    let sent = work.late_round1(&frame, calls).await.expect("late round");
    assert_eq!(sent.len(), 5);
    let launches: Vec<Event> = b
        .fake
        .events()
        .into_iter()
        .filter(|e| matches!(e, Event::Launch { .. }))
        .collect();
    assert_eq!(
        launches,
        [
            Event::Launch { writes: 3, blur: 0 },
            Event::Launch { writes: 1, blur: 0 },
            Event::Launch { writes: 1, blur: 0 }
        ]
    );
    for e in b.fake.events() {
        if let Event::Call { region_in_use, .. } = e {
            assert!(
                region_in_use <= 4 * MIB,
                "{region_in_use} bytes held past the window"
            );
        }
    }
    for (_, id) in sent {
        let _ = work.calls.result(work.clock, id).await.expect("answer");
    }
    drop(frame);
    drop(work);
    assert_released(&b).await;
}

#[tokio::test]
async fn a_late_conditions_verdict_ends_the_branch_and_what_needs_it() {
    let b = bed();
    let plan = parse_plan(FACES_LATE.as_bytes(), false).expect("plan");
    let mut clock = StageClock::new(
        Arc::new(RequestState::new(RequestClock::start(Instant::now(), 30.0))),
        Arc::clone(&b.flow.metrics),
    );
    let (mut work, frame) = late_bed(&b, &mut clock).await;
    let late = ["faces", "face_embed"];
    assert_eq!(work.check_conditions(&plan, &late, Some(&[16])), late);
    assert_eq!(
        work.check_conditions(&plan, &late, Some(&[2])),
        Vec::<&str>::new()
    );
    assert_eq!(work.answers.states["faces"], BranchState::NotRun);
    assert_eq!(work.answers.states["face_embed"], BranchState::NotRun);
    let (mut other, _) = late_bed(&b, &mut clock).await;
    assert!(other.check_conditions(&plan, &late, None).is_empty());
    assert_eq!(
        other.answers.states["face_embed"],
        BranchState::Error(DEPENDENCY_FAILED)
    );
    drop(frame);
}

#[tokio::test]
async fn a_late_job_that_fails_fails_the_photo_and_gives_its_slices_back() {
    let b = bed();
    b.fake
        .set(|s| s.launch_fault = Some(GpuFault::Failed("late kernel".into())));
    let mut clock = StageClock::new(
        Arc::new(RequestState::new(RequestClock::start(Instant::now(), 30.0))),
        Arc::clone(&b.flow.metrics),
    );
    let (mut work, frame) = late_bed(&b, &mut clock).await;
    let failed = work.late_round1(&frame, vec![late_call("faces", 1)]).await;
    assert!(
        matches!(failed, Err(Stop::Fail(PipelineError::Engine(m))) if m.contains("late kernel"))
    );
    drop(frame);
    drop(work);
    assert_released(&b).await;
}

/// Detect, then faces late in a 64 input (one recogniser batch of 2), if class 16 was found.
const FACES_LATE_SMALL: &str = r#"{"version": 1, "branches": ["detect", "faces", "face_embed"], "crops": "none", "detector": {"model": "det", "input_size": 640, "letterbox_fill": 114}, "faces": {"detector": "scrfd", "input_size": 64, "recogniser": "arcface", "batch": 2, "score_min": 0.5, "nms": 0.4, "max_faces": 50}, "if_detected": {"faces": [16]}}"#;

fn models_called(b: &Bed) -> Vec<String> {
    b.fake.calls().into_iter().map(|(m, _)| m).collect()
}

#[tokio::test]
async fn late_faces_run_after_the_detector_found_their_class() {
    let b = bed();
    let kps = [-0.4, -0.3, 0.4, -0.3, 0.0, 0.1, -0.3, 0.5, 0.3, 0.5];
    b.fake.set(|s| {
        s.detections = vec![[100.0, 100.0, 300.0, 300.0, 0.9, 16.0]];
        s.face_fires = vec![(2 * (3 * 8 + 3), 0.9, [1.0; 4], kps)];
    });
    let answer = b
        .flow
        .process(photo(FACES_LATE_SMALL))
        .await
        .expect("answer");
    assert_eq!(answer.branches["detect"], BranchState::Ok);
    assert_eq!(answer.branches["faces"], BranchState::Ok);
    assert_eq!(answer.branches["face_embed"], BranchState::Ok);
    let events = b.fake.events();
    assert_eq!(
        events[0],
        Event::Decode {
            bytes: vec![3 * 640 * 640],
            fills: vec![Some(114)]
        },
        "the decode writes the detector's input only: the face pass waits for its condition"
    );
    assert!(
        events.contains(&Event::Launch { writes: 1, blur: 0 }),
        "the late round writes the face pass by a launch job"
    );
    assert_eq!(models_called(&b), ["det", "scrfd", "arcface"]);
    assert!(output(&answer, "face_embeddings").is_some());
    assert_released(&b).await;
}

#[tokio::test]
async fn late_faces_do_not_run_when_their_class_is_missing() {
    let b = bed();
    b.fake.set(|s| {
        s.detections = vec![[100.0, 100.0, 300.0, 300.0, 0.9, 2.0]];
        s.face_fires = vec![(0, 0.9, [1.0; 4], [0.0; 10])];
    });
    let answer = b
        .flow
        .process(photo(FACES_LATE_SMALL))
        .await
        .expect("answer");
    assert_eq!(answer.branches["detect"], BranchState::Ok);
    assert_eq!(answer.branches["faces"], BranchState::NotRun);
    assert_eq!(answer.branches["face_embed"], BranchState::NotRun);
    assert_eq!(models_called(&b), ["det"]);
    assert!(answer.outputs.iter().all(|(n, _)| !n.starts_with("face")));
    assert_released(&b).await;
}
