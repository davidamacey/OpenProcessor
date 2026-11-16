//! Which failures are sent again, against a scripted server: the statuses the engine and the
//! transport can answer, one photo at a time, and the call timeout.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use op_engine_client::retry::{Class, DEADLINE, RETRIES, Verdict, backoff, decide};
use op_engine_client::{Branch, EngineClient, EngineError, EnginePlan, ProbeOutcome};
use tonic::{Code, Status};

use crate::answers::{case_named, plan_of, response_of};
use crate::support::{Counts, Fake, Reply, serve_fake, settings};

const PHOTO: &[u8] = b"photo";

async fn scripted() -> (
    Arc<Fake>,
    EnginePlan,
    op_engine_client::proto::ModelInferResponse,
) {
    let case = case_named("frame_only");
    let fake = Arc::new(Fake::default());
    (fake, plan_of(&case), response_of(&case))
}

#[tokio::test]
async fn a_call_the_transport_lost_is_sent_again_until_it_goes_through() {
    let (fake, plan, answer) = scripted().await;
    for message in ["Stream removed", "Received RST_STREAM with error code 1"] {
        fake.then(Reply::Fail(Status::internal(message)));
    }
    fake.then(Reply::Answer(answer));
    let addr = serve_fake(Arc::clone(&fake)).await;
    let counts = Arc::new(Counts::default());
    let client = EngineClient::new(&settings(addr), Some(counts.clone())).unwrap();
    let answer = client.process(PHOTO, &plan, None).await.unwrap();
    assert!(answer.ok());
    assert_eq!(fake.calls.load(Ordering::SeqCst), 3);
    assert_eq!(*counts.retried.lock().unwrap(), ["INTERNAL", "INTERNAL"]);
    assert!(counts.exhausted.lock().unwrap().is_empty());
}

#[tokio::test]
async fn an_overload_that_clears_is_ridden_out_however_many_calls_it_takes() {
    let (fake, plan, answer) = scripted().await;
    for _ in 0..5 {
        fake.then(Reply::Fail(Status::resource_exhausted("full")));
    }
    fake.then(Reply::Answer(answer));
    let addr = serve_fake(Arc::clone(&fake)).await;
    let mut settings = settings(addr);
    // five waits of 0.05 s doubling, with jitter, are over two seconds
    settings.deadline = Duration::from_secs(30);
    let client = EngineClient::new(&settings, None).unwrap();
    assert!(client.process(PHOTO, &plan, None).await.unwrap().ok());
    assert_eq!(fake.calls.load(Ordering::SeqCst), 6);
}

#[tokio::test]
async fn statuses_that_a_second_call_cannot_change_are_not_sent_again() {
    let (fake, plan, _) = scripted().await;
    let addr = serve_fake(Arc::clone(&fake)).await;
    let client = EngineClient::new(&settings(addr), None).unwrap();
    let cases = [
        (
            Status::internal("frame engine: decode went wrong"),
            "failed",
        ),
        (
            Status::deadline_exceeded("past the engine's deadline"),
            "failed",
        ),
        (
            Status::invalid_argument("bad_field: embed.size"),
            "rejected",
        ),
        (Status::not_found("unknown model 'x'"), "rejected"),
        (Status::unimplemented("LoadModel"), "failed"),
        (Status::cancelled("gone"), "failed"),
    ];
    for (status, kind) in cases {
        let code = status.code();
        fake.calls.store(0, Ordering::SeqCst);
        fake.then(Reply::Fail(status));
        let error = client.process(PHOTO, &plan, None).await.unwrap_err();
        assert_eq!(fake.calls.load(Ordering::SeqCst), 1, "{code:?}");
        match (kind, &error) {
            (
                "failed",
                EngineError::Failed {
                    code: got,
                    attempts: 1,
                    ..
                },
            )
            | ("rejected", EngineError::Rejected { code: got, .. }) => assert_eq!(*got, code),
            _ => panic!("{code:?}: {error}"),
        }
        assert!(!error.transient(), "{code:?}");
    }
}

#[tokio::test]
async fn a_call_with_no_answer_in_time_fails_the_photo_without_a_second_call() {
    let (fake, plan, _) = scripted().await;
    *fake.hold.lock().unwrap() = Duration::from_millis(600);
    let addr = serve_fake(Arc::clone(&fake)).await;
    let mut settings = settings(addr);
    settings.call_timeout = Duration::from_millis(200);
    let client = EngineClient::new(&settings, None).unwrap();
    let error = client.process(PHOTO, &plan, None).await.unwrap_err();
    assert!(
        matches!(
            error,
            EngineError::Failed {
                code: Code::DeadlineExceeded,
                attempts: 1,
                ..
            }
        ),
        "{error}"
    );
    assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn the_request_is_protocol_version_1_exactly() {
    let (fake, _, answer) = scripted().await;
    fake.answering(answer);
    let addr = serve_fake(Arc::clone(&fake)).await;
    let client = EngineClient::new(&settings(addr), None).unwrap();
    let plan = EnginePlan::stored_boxes_embed(&op_engine_client::EncoderModels {
        image_model: "pe".into(),
        crop_model: "pe_crop".into(),
        crop_size: 224,
    });
    // an answer for another plan is refused, but the request is what is looked at here
    let _ = client
        .process(b"abc", &plan, Some(&[[0.5, 0.25, 0.75, 1.0]]))
        .await;
    let request = fake.requests.lock().unwrap().remove(0);
    assert_eq!(request.model_name, "frame_engine");
    assert!(
        request.outputs.is_empty(),
        "no outputs filter: the whole answer"
    );
    let names: Vec<_> = request
        .inputs
        .iter()
        .map(|i| (i.name.as_str(), i.datatype.as_str()))
        .collect();
    assert_eq!(
        names,
        [("image", "BYTES"), ("plan", "BYTES"), ("boxes", "FP64")]
    );
    assert_eq!(request.inputs[0].shape, [1]);
    assert_eq!(request.inputs[2].shape, [1, 4]);
    assert_eq!(
        request.raw_input_contents.len(),
        3,
        "every input in raw_input_contents"
    );
    let image = op_engine_client::codec::unpack_bytes(&request.raw_input_contents[0]).unwrap();
    assert_eq!(image, [&b"abc"[..]]);
    let text = op_engine_client::codec::unpack_bytes(&request.raw_input_contents[1]).unwrap();
    assert_eq!(
        std::str::from_utf8(text[0]).unwrap(),
        plan.to_json().unwrap()
    );
    let expected: Vec<u8> = [0.5f64, 0.25, 0.75, 1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(request.raw_input_contents[2].as_ref(), expected.as_slice());
}

#[test]
fn an_overload_is_retried_to_the_deadline_and_an_unavailable_engine_three_times() {
    let wait = |class, attempt, elapsed| {
        decide(class, attempt, Duration::from_secs(elapsed), DEADLINE, 0.0)
    };
    for attempt in 1..=RETRIES {
        assert!(
            matches!(wait(Class::Transient, attempt, 0), Verdict::After(_)),
            "{attempt}"
        );
    }
    assert_eq!(wait(Class::Transient, RETRIES + 1, 0), Verdict::Stop);
    assert!(matches!(wait(Class::Overload, 50, 119), Verdict::After(_)));
    assert_eq!(wait(Class::Overload, 50, 120), Verdict::Stop);
    assert_eq!(wait(Class::Transient, 1, 121), Verdict::Stop);
    assert_eq!(wait(Class::Final, 1, 0), Verdict::Stop);
}

#[test]
fn the_wait_doubles_from_50_ms_with_up_to_200_ms_of_jitter_and_is_capped_at_two_seconds() {
    // tenacity wait_exponential_jitter(multiplier=0.05, max=2.0, jitter=0.2) at jitter 0 and 1
    let secs = |attempt, unit| backoff(attempt, unit).as_secs_f64();
    for (attempt, base) in [(1, 0.05), (2, 0.1), (3, 0.2), (4, 0.4), (5, 0.8), (6, 1.6)] {
        assert!((secs(attempt, 0.0) - base).abs() < 1e-9, "{attempt}");
        assert!(
            (secs(attempt, 1.0) - (base + 0.2).min(2.0)).abs() < 1e-9,
            "{attempt}"
        );
    }
    assert!((secs(7, 0.0) - 2.0).abs() < 1e-9);
    assert!((secs(500, 1.0) - 2.0).abs() < 1e-9);
}

#[tokio::test]
async fn readiness_is_asked_on_the_next_channel_when_the_transport_refuses_one() {
    // a probe answered "not ready" is an answer; only an undelivered one is asked again
    let (fake, _, _) = scripted().await;
    let addr = serve_fake(Arc::clone(&fake)).await;
    let client = EngineClient::new(&settings(addr), None).unwrap();
    let readiness = client.probe_ready("frame_engine").await;
    assert_eq!(readiness.outcome, ProbeOutcome::NotReady);
    assert_eq!(readiness.attempts, 1);
    assert_eq!(fake.probes.load(Ordering::SeqCst), 1);
    fake.ready
        .lock()
        .unwrap()
        .insert("frame_engine".into(), true);
    assert!(client.probe_ready("frame_engine").await.ready());
    let _ = Branch::Detect;
}
