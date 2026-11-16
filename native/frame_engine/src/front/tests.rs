//! The front's behaviour, called through the service trait without a network: the check order,
//! the outputs filter, the status of each failure class, the deadline and the cancelling of a
//! photo whose caller went away. The wire is the contract test's (`tests/native/`).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tonic::{Code, Request, Response, Status};

use super::{Front, FrontConfig, FrontHandle};
use crate::metrics::Metrics;
use crate::pipeline::{Photo, PhotoAnswer, Pipeline, PipelineError, RequestState, Tensor};
use crate::proto::grpc_inference_service_server::GrpcInferenceService;
use crate::proto::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use crate::proto::model_metadata_response::TensorMetadata;
use crate::proto::{
    ModelInferRequest, ModelInferResponse, ModelMetadataRequest, ModelReadyRequest,
    ServerLiveRequest, ServerReadyRequest,
};
use crate::stub::StubPipeline;
use crate::wire::branch_status::BranchState;
use crate::wire::bytes_tensor::pack;
use crate::wire::contract::contract;

const FRAME: &str = r#"{"version": 1, "branches": ["detect", "embed_frame"], "detector": {"model": "d", "input_size": 640, "letterbox_fill": 114}, "embed": {"model": "pe", "size": 336}}"#;

fn request(image: &[u8], plan: &str) -> ModelInferRequest {
    let mut r = ModelInferRequest {
        model_name: "frame_engine".into(),
        id: "photo-7".into(),
        ..Default::default()
    };
    for (name, value) in [("image", image), ("plan", plan.as_bytes())] {
        r.inputs.push(InferInputTensor {
            name: name.into(),
            datatype: "BYTES".into(),
            shape: vec![1],
            ..Default::default()
        });
        r.raw_input_contents.push(pack(&[value]));
    }
    r
}

fn front(max_in_flight: u32, deadline: f64) -> (Front<StubPipeline>, FrontHandle<StubPipeline>) {
    let config = FrontConfig {
        max_in_flight,
        request_deadline_seconds: deadline,
    };
    Front::new(Arc::new(StubPipeline::new(None)), Metrics::new(), config)
}

async fn infer<P: Pipeline>(
    front: &Front<P>,
    r: ModelInferRequest,
) -> Result<ModelInferResponse, Status> {
    front
        .model_infer(Request::new(r))
        .await
        .map(Response::into_inner)
}

fn refused(result: Result<ModelInferResponse, Status>) -> Status {
    result.expect_err("refused")
}

#[tokio::test]
async fn the_checks_run_in_the_contract_order() {
    let (front, _) = front(1, 30.0);
    let _full = front.state.cap.try_admit().expect("fill the cap");
    let blocked = std::env::temp_dir().join(format!("fe-front-order-{}", std::process::id()));
    std::fs::write(&blocked, b"").expect("create");
    let (not_ready, _) = {
        let config = FrontConfig {
            max_in_flight: 0,
            request_deadline_seconds: 30.0,
        };
        Front::new(
            Arc::new(StubPipeline::new(Some(blocked.clone()))),
            Metrics::new(),
            config,
        )
    };
    not_ready.state.pipeline.check().await;
    std::fs::remove_file(&blocked).expect("remove");

    // every check fails: the first one answers, and each request fails the one after it too
    let mut wrong_model = request(b"x", "{");
    wrong_model.model_name = "other".into();
    let order = [
        (
            refused(infer(&not_ready, wrong_model).await),
            Code::NotFound,
        ),
        (
            refused(infer(&not_ready, request(b"x", "{")).await),
            Code::InvalidArgument,
        ),
        (
            refused(infer(&not_ready, request(b"x", FRAME)).await),
            Code::Unavailable,
        ),
        (
            refused(infer(&front, request(b"x", FRAME)).await),
            Code::ResourceExhausted,
        ),
    ];
    for (status, code) in order {
        assert_eq!(status.code(), code, "{status:?}");
    }
    let status = refused(infer(&front, request(b"x", FRAME)).await);
    assert_eq!(status.message(), "overload: 1 requests in flight (cap 1)");
    let status = refused(infer(&not_ready, request(b"x", FRAME)).await);
    assert_eq!(status.message(), "frame engine not ready");
    let mut other = request(b"x", FRAME);
    other.model_name = "some_other_model".into();
    assert_eq!(
        refused(infer(&front, other).await).message(),
        "unknown model 'some_other_model'"
    );
    let status = refused(infer(&front, request(b"x", r#"{"version": 2}"#)).await);
    assert_eq!(
        (status.code(), status.message()),
        (Code::InvalidArgument, "bad_version: 2")
    );
}

#[tokio::test]
async fn an_overload_is_counted_and_the_slot_comes_back() {
    let (front, handle) = front(1, 30.0);
    let held = tokio::spawn({
        let r = request(b"stub:hold_ms=200", FRAME);
        let front = Front {
            state: Arc::clone(&front.state),
        };
        async move { infer(&front, r).await }
    });
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert_eq!(handle.in_flight(), 1);
    assert_eq!(
        refused(infer(&front, request(b"x", FRAME)).await).code(),
        Code::ResourceExhausted
    );
    held.await
        .expect("joined")
        .expect("the held photo is answered");
    assert_eq!(handle.in_flight(), 0);
    infer(&front, request(b"x", FRAME))
        .await
        .expect("the slot came back");
    let text = front.state.metrics.render();
    assert!(
        text.contains("op_frame_engine_overloads_total{cause=\"in_flight\"} 1"),
        "{text}"
    );
    assert!(text.contains("op_frame_engine_in_flight 0"), "{text}");
}

#[tokio::test]
async fn a_plain_photo_is_answered_with_its_id_the_version_and_the_four_always_outputs() {
    let (front, _) = front(8, 30.0);
    let response = infer(&front, request(b"\xff\xd8x", FRAME))
        .await
        .expect("answer");
    assert_eq!(
        (
            response.model_name.as_str(),
            response.model_version.as_str(),
            response.id.as_str()
        ),
        ("frame_engine", "1", "photo-7")
    );
    let names: Vec<&str> = response.outputs.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "status",
            "decode_route",
            "branch_status",
            "size",
            "det_boxes",
            "det_scores",
            "det_classes",
            "letterbox",
            "blur_sums",
            "whole_embedding"
        ]
    );
    assert_eq!(response.raw_output_contents.len(), names.len());
    // status is one BYTES element "ok"
    assert_eq!(
        response.raw_output_contents[0].as_ref(),
        b"\x02\x00\x00\x00ok"
    );
    // size is two little-endian int32s
    assert_eq!(
        response.raw_output_contents[3].as_ref(),
        [96, 0, 0, 0, 64, 0, 0, 0]
    );
    let letterbox = &response.raw_output_contents[7];
    assert_eq!(letterbox.len(), 24);
    assert_eq!(
        f64::from_le_bytes(letterbox[16..24].try_into().expect("8 bytes")),
        106.0
    );
    let blur = response
        .outputs
        .iter()
        .find(|o| o.name == "blur_sums")
        .expect("blur_sums");
    assert_eq!(
        (blur.datatype.as_str(), blur.shape.clone()),
        ("INT64", vec![3, 3])
    );
}

#[tokio::test]
async fn the_outputs_filter_applies_to_every_output_and_empty_means_all() {
    let (front, _) = front(8, 30.0);
    let mut r = request(b"x", FRAME);
    for name in ["det_scores", "status", "branch_status", "nonexistent"] {
        r.outputs.push(InferRequestedOutputTensor {
            name: name.into(),
            ..Default::default()
        });
    }
    let response = infer(&front, r).await.expect("answer");
    let names: Vec<&str> = response.outputs.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["status", "branch_status", "det_scores"]);
    assert_eq!(response.outputs.len(), response.raw_output_contents.len());
    let mut only_route = request(b"x", FRAME);
    only_route.outputs.push(InferRequestedOutputTensor {
        name: "decode_route".into(),
        ..Default::default()
    });
    let response = infer(&front, only_route).await.expect("answer");
    assert_eq!(response.outputs.len(), 1);
    assert_eq!(
        response.raw_output_contents[0].as_ref(),
        b"\x03\x00\x00\x00gpu"
    );
}

#[tokio::test]
async fn each_failure_class_is_its_status_and_internal_carries_the_prefix() {
    let (front, handle) = front(8, 30.0);
    let cases = [
        (
            "overload",
            Code::ResourceExhausted,
            "overload: budget: stub",
        ),
        (
            "unavailable",
            Code::Unavailable,
            "model server unavailable: stub: down",
        ),
        (
            "context_lost",
            Code::Unavailable,
            "frame engine restarting: stub: lost",
        ),
        ("internal", Code::Internal, "frame engine: stub failure"),
    ];
    for (word, code, message) in cases {
        let status = refused(
            infer(
                &front,
                request(format!("stub:fail={word}").as_bytes(), FRAME),
            )
            .await,
        );
        assert_eq!((status.code(), status.message()), (code, message), "{word}");
        assert_eq!(
            handle.in_flight(),
            0,
            "the slot is free before the status is sent"
        );
    }
    let text = front.state.metrics.render();
    assert!(
        text.contains("op_frame_engine_overloads_total{cause=\"budget\"} 1"),
        "{text}"
    );
}

#[tokio::test]
async fn a_photo_past_the_request_deadline_is_answered_with_the_stage_it_was_in() {
    let (front, handle) = front(8, 0.3);
    let started = Instant::now();
    let status = refused(infer(&front, request(b"stub:hold_ms=5000,stage=embed_pe", FRAME)).await);
    assert_eq!(status.code(), Code::DeadlineExceeded);
    assert_eq!(
        status.message(),
        "frame engine: no answer within 0.3 s (stage embed_pe)"
    );
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(handle.in_flight(), 0);
    let text = front.state.metrics.render();
    assert!(
        text.contains("op_frame_engine_request_deadlines_total{stage=\"embed_pe\"} 1"),
        "{text}"
    );
    infer(&front, request(b"x", FRAME))
        .await
        .expect("the next photo is served");
}

#[tokio::test]
async fn a_caller_that_goes_away_cancels_its_photo() {
    struct Watching {
        seen: Arc<AtomicBool>,
        wake: tokio::sync::Notify,
    }
    impl Pipeline for Watching {
        fn ready(&self) -> bool {
            true
        }
        fn lost(&self) -> bool {
            false
        }
        fn wake(&self) -> &tokio::sync::Notify {
            &self.wake
        }
        async fn check(&self) {}
        async fn process(&self, photo: Photo) -> Result<PhotoAnswer, PipelineError> {
            // a worker thread checks the flag at its next hand-off, after the handler is gone
            let seen = Arc::clone(&self.seen);
            let request = Arc::clone(&photo.request);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(150));
                seen.store(request.cancelled(), Ordering::SeqCst);
            });
            std::future::pending::<()>().await;
            Err(PipelineError::Engine("unreachable".into()))
        }
    }
    let seen = Arc::new(AtomicBool::new(false));
    let pipeline = Watching {
        seen: Arc::clone(&seen),
        wake: tokio::sync::Notify::new(),
    };
    let config = FrontConfig {
        max_in_flight: 4,
        request_deadline_seconds: 30.0,
    };
    let (front, handle) = Front::new(Arc::new(pipeline), Metrics::new(), config);
    let call = tokio::spawn(async move { infer(&front, request(b"x", FRAME)).await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(handle.in_flight(), 1);
    call.abort();
    let _ = call.await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        seen.load(Ordering::SeqCst),
        "the photo saw its request cancelled"
    );
    assert_eq!(handle.in_flight(), 0, "and its slot is free");
}

#[tokio::test]
async fn an_answered_photo_is_not_cancelled_when_the_handler_returns() {
    struct Capturing(
        tokio::sync::Notify,
        std::sync::Mutex<Option<Arc<RequestState>>>,
    );
    impl Pipeline for Capturing {
        fn ready(&self) -> bool {
            true
        }
        fn lost(&self) -> bool {
            false
        }
        fn wake(&self) -> &tokio::sync::Notify {
            &self.0
        }
        async fn check(&self) {}
        async fn process(&self, photo: Photo) -> Result<PhotoAnswer, PipelineError> {
            *self.1.lock().expect("lock") = Some(Arc::clone(&photo.request));
            Ok(PhotoAnswer {
                status: "ok".into(),
                decode_route: "gpu".into(),
                branches: BTreeMap::new(),
                outputs: Vec::new(),
            })
        }
    }
    let pipeline = Arc::new(Capturing(
        tokio::sync::Notify::new(),
        std::sync::Mutex::new(None),
    ));
    let config = FrontConfig {
        max_in_flight: 4,
        request_deadline_seconds: 30.0,
    };
    let (front, _) = Front::new(Arc::clone(&pipeline), Metrics::new(), config);
    infer(&front, request(b"x", FRAME)).await.expect("answer");
    let state = pipeline
        .1
        .lock()
        .expect("lock")
        .take()
        .expect("the photo ran");
    assert!(
        !state.cancelled(),
        "an answered photo is done, not abandoned"
    );
}

#[tokio::test]
async fn the_readiness_calls_answer_as_the_python_front_does() {
    let (front, handle) = front(8, 30.0);
    let live = || async {
        front
            .server_live(Request::new(ServerLiveRequest {}))
            .await
            .expect("live")
            .into_inner()
            .live
    };
    let ready = || async {
        front
            .server_ready(Request::new(ServerReadyRequest {}))
            .await
            .expect("ready")
            .into_inner()
            .ready
    };
    let model = |name: &str, version: &str| {
        let r = ModelReadyRequest {
            name: name.into(),
            version: version.into(),
        };
        async {
            front
                .model_ready(Request::new(r))
                .await
                .expect("model ready")
                .into_inner()
                .ready
        }
    };
    assert!(live().await && ready().await);
    assert!(model("frame_engine", "").await);
    assert!(model("frame_engine", "1").await);
    assert!(!model("frame_engine", "2").await, "an unknown version");
    assert!(!model("some_other_model", "").await, "an unknown model");
    handle.begin_drain();
    assert!(live().await, "draining is still alive");
    assert!(!ready().await);
    assert!(!model("frame_engine", "").await);
    let status = refused(infer(&front, request(b"x", FRAME)).await);
    assert_eq!(
        (status.code(), status.message()),
        (Code::Unavailable, "frame engine not ready")
    );
}

#[tokio::test]
async fn metadata_is_the_contracts_tensors_and_an_unknown_model_is_not_found() {
    let (front, _) = front(8, 30.0);
    let answer = front
        .model_metadata(Request::new(ModelMetadataRequest {
            name: "frame_engine".into(),
            version: String::new(),
        }))
        .await
        .expect("metadata")
        .into_inner();
    let c = contract();
    assert_eq!(
        (answer.name.as_str(), answer.platform.as_str()),
        ("frame_engine", "frame_engine")
    );
    assert_eq!(answer.versions, ["1"]);
    let table = |tensors: &[TensorMetadata]| -> Vec<(String, String, Vec<i64>)> {
        tensors
            .iter()
            .map(|t| (t.name.clone(), t.datatype.clone(), t.shape.clone()))
            .collect()
    };
    let want = |names: Vec<(&String, &String, &Vec<i64>)>| -> Vec<(String, String, Vec<i64>)> {
        names
            .into_iter()
            .map(|(n, d, s)| (n.clone(), d.clone(), s.clone()))
            .collect()
    };
    assert_eq!(
        table(&answer.inputs),
        want(
            c.inputs
                .iter()
                .chain([&c.read_by_key.input])
                .map(|t| (&t.name, &t.datatype, &t.shape))
                .collect()
        ),
        "the protocol's inputs, then the read-by-key extension's"
    );
    assert_eq!(
        table(&answer.outputs),
        want(
            c.outputs
                .iter()
                .map(|t| (&t.name, &t.datatype, &t.shape))
                .collect()
        )
    );
    assert_eq!(answer.outputs.len(), 27);
    let status = front
        .model_metadata(Request::new(ModelMetadataRequest {
            name: "x".into(),
            version: String::new(),
        }))
        .await
        .expect_err("unknown");
    assert_eq!(
        (status.code(), status.message()),
        (Code::NotFound, "unknown model 'x'")
    );
}

#[tokio::test]
async fn a_tensor_of_the_wrong_type_is_the_engines_own_failure() {
    struct Wrong(tokio::sync::Notify, Vec<(String, Tensor)>);
    impl Pipeline for Wrong {
        fn ready(&self) -> bool {
            true
        }
        fn lost(&self) -> bool {
            false
        }
        fn wake(&self) -> &tokio::sync::Notify {
            &self.0
        }
        async fn check(&self) {}
        async fn process(&self, _: Photo) -> Result<PhotoAnswer, PipelineError> {
            Ok(PhotoAnswer {
                status: "ok".into(),
                decode_route: "gpu".into(),
                branches: BTreeMap::from([("detect".to_owned(), BranchState::Ok)]),
                outputs: self.1.clone(),
            })
        }
    }
    let config = FrontConfig {
        max_in_flight: 4,
        request_deadline_seconds: 30.0,
    };
    let wrong_type = vec![(
        "size".to_owned(),
        Tensor::Fp32 {
            shape: vec![2],
            data: vec![1.0, 2.0],
        },
    )];
    let (front, _) = Front::new(
        Arc::new(Wrong(tokio::sync::Notify::new(), wrong_type)),
        Metrics::new(),
        config,
    );
    let status = refused(infer(&front, request(b"x", FRAME)).await);
    assert_eq!(status.code(), Code::Internal);
    assert_eq!(
        status.message(),
        "frame engine: output size is FP32, the contract says INT32"
    );
    let unknown = vec![(
        "nope".to_owned(),
        Tensor::Bytes(vec![Bytes::from_static(b"a")]),
    )];
    let (front, _) = Front::new(
        Arc::new(Wrong(tokio::sync::Notify::new(), unknown)),
        Metrics::new(),
        config,
    );
    let status = refused(infer(&front, request(b"x", FRAME)).await);
    assert_eq!(
        status.message(),
        "frame engine: output nope is not in protocol version 1"
    );
}

fn keyed_request(key: &str, bytes: &[u8]) -> ModelInferRequest {
    use op_local_file::{FileClaim, FileRef};
    let file = FileRef::new(key, FileClaim::of(bytes)).expect("a local key");
    let claim = file.claim();
    let document = format!(
        r#"{{"key": "{key}", "size": {}, "imohash": "{}"}}"#,
        claim.size,
        claim.imohash.to_hex()
    );
    let mut r = request(b"", FRAME);
    r.inputs[0].name = "source".into();
    r.raw_input_contents[0] = pack(&[document.as_bytes()]);
    r
}

#[tokio::test]
async fn a_keyed_request_is_refused_as_invalid_when_the_engine_has_no_root_for_it() {
    let (front, _) = front(8, 30.0);
    let status = refused(infer(&front, keyed_request("landing/7/a.0", b"x")).await);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(
        status
            .message()
            .starts_with("root_not_configured: 'landing'"),
        "{}",
        status.message()
    );
}

#[tokio::test]
async fn a_keyed_request_to_an_engine_with_a_root_reads_the_file_and_answers_like_bytes() {
    use crate::photo::read::KeyedRoots;
    let dir = tempfile::tempdir().expect("dir");
    let landing = dir.path().join("landing");
    std::fs::create_dir_all(landing.join("7")).expect("landing");
    let bytes = b"\xff\xd8 a plain photo";
    std::fs::write(landing.join("7/a.0"), bytes).expect("file");
    let roots = KeyedRoots::new(Some(&landing), None, 1 << 20).expect("roots");
    let config = FrontConfig {
        max_in_flight: 8,
        request_deadline_seconds: 30.0,
    };
    let (front, _) = Front::new(
        Arc::new(StubPipeline::new(None).with_roots(roots)),
        Metrics::new(),
        config,
    );
    let keyed = infer(&front, keyed_request("landing/7/a.0", bytes))
        .await
        .expect("keyed answer");
    let plain = infer(&front, request(bytes, FRAME))
        .await
        .expect("bytes answer");
    assert_eq!(keyed.raw_output_contents, plain.raw_output_contents);
    assert_eq!(
        keyed.outputs.iter().map(|o| &o.name).collect::<Vec<_>>(),
        plain.outputs.iter().map(|o| &o.name).collect::<Vec<_>>()
    );
    // a file that is gone is a photo status, not a gRPC error
    std::fs::remove_file(landing.join("7/a.0")).expect("remove");
    let gone = infer(&front, keyed_request("landing/7/a.0", bytes))
        .await
        .expect("answered");
    let status = gone
        .outputs
        .iter()
        .position(|o| o.name == "status")
        .expect("a status");
    assert_eq!(&gone.raw_output_contents[status][4..], b"missing");
    // both, then neither
    let mut both = keyed_request("landing/7/a.0", bytes);
    both.inputs.push(request(bytes, FRAME).inputs.remove(0));
    both.raw_input_contents.push(pack(&[bytes.as_slice()]));
    let status = refused(infer(&front, both).await);
    assert_eq!(
        (status.code(), status.message()),
        (
            Code::InvalidArgument,
            "image and source are exclusive: send one"
        )
    );
}
