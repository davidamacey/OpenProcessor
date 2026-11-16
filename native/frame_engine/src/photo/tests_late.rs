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
        Vec::<&str>::new(),
        "faces is not built yet"
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
        call: (owner, "faces", "scrfd", input, "output"),
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
