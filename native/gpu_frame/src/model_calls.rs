//! The model call, behind one interface (plan 17 sections 1.5 and 2.0): inputs are slices of the
//! registered region, the answer is the named output tensors or a typed error. The model server
//! over CUDA shared memory is the one filling (`model_server.rs`); an in-process runtime would be
//! a second filling only if a measurement reopens decision 4.1, and the loser is then deleted.
//!
//! A call's future owns nothing but borrows: the slices it reads stay with the caller, who
//! releases them only after the future completed or was dropped (dropping it cancels the call).

use std::fmt;
use std::future::Future;
use std::time::Duration;

use bytes::Bytes;
use tokio::time::Instant;

use crate::region::Slice;

/// One input tensor read from a region slice.
#[derive(Debug)]
pub struct InputTensor<'a> {
    /// The input's name in the model's configuration.
    pub name: &'a str,
    /// The KServe datatype (`UINT8`, `FP32`, ...).
    pub datatype: &'a str,
    /// The shape, batch first.
    pub shape: &'a [i64],
    /// The region bytes sent as this input; the slice stays borrowed for the call.
    pub slice: &'a Slice,
    /// The slice's bytes copied to the host, sent in the request body when the inputs travel in
    /// the body ([`crate::model_server::InputMode::Body`]); `None` over shared memory.
    pub body: Option<&'a Bytes>,
}

/// One model call.
#[derive(Debug)]
pub struct InferCall<'a> {
    /// The model's name on the server.
    pub model: &'a str,
    /// The input tensors, in the model's input order.
    pub inputs: &'a [InputTensor<'a>],
    /// The outputs to return, in this order.
    pub outputs: &'a [&'a str],
    /// The photo's request deadline; the call's own bound is the smaller of this and the
    /// filling's per-call timeout.
    pub deadline: Instant,
}

/// One output tensor: its raw little-endian bytes as the server sent them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputTensor {
    /// The output's name in the model's configuration.
    pub name: String,
    /// The KServe datatype (`UINT8`, `FP32`, ...).
    pub datatype: String,
    /// The shape, batch first.
    pub shape: Vec<i64>,
    /// The raw little-endian tensor bytes, `shape` times the datatype's size.
    pub data: Bytes,
}

/// Why a call gave no outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallError {
    /// The server or the channel is gone, or the call was cancelled by the channel: retryable.
    Unavailable(String),
    /// No answer within the call's bound.
    Deadline,
    /// The server answered the call with an error: the branch fails.
    Model {
        /// The server's error code.
        code: String,
        /// The server's error message.
        message: String,
    },
    /// The server does not know the region and it could not be registered again.
    RegionUnknown(String),
}

impl CallError {
    /// Whether the server may be gone, so readiness should be probed now (a deadline or an
    /// unavailable channel is evidence about that call only; the probe decides).
    pub fn suspects_server(&self) -> bool {
        matches!(self, CallError::Unavailable(_) | CallError::Deadline)
    }
}

impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallError::Unavailable(m) => write!(f, "model server unavailable: {m}"),
            CallError::Deadline => write!(f, "no answer within the call's deadline"),
            CallError::Model { code, message } => write!(f, "model error {code}: {message}"),
            CallError::RegionUnknown(m) => write!(f, "region unknown to the model server: {m}"),
        }
    }
}

impl std::error::Error for CallError {}

/// The model call interface.
pub trait ModelCalls: Send + Sync {
    /// Run one call; dropping the future cancels it.
    fn infer<'a>(
        &'a self,
        call: InferCall<'a>,
    ) -> impl Future<Output = Result<Vec<OutputTensor>, CallError>> + Send + 'a;
}

/// The bound of one call: the per-call timeout or what is left of the request, whichever is
/// sooner; `None` once the request deadline passed (plan 17 section 2.2, "Model calls").
pub fn call_timeout(
    per_call: Duration,
    request_deadline: Instant,
    now: Instant,
) -> Option<Duration> {
    let left = request_deadline.checked_duration_since(now)?;
    if left.is_zero() {
        return None;
    }
    Some(per_call.min(left))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_call_bound_is_the_sooner_of_its_timeout_and_the_request() {
        let now = Instant::now();
        let s = Duration::from_secs;
        assert_eq!(call_timeout(s(10), now + s(60), now), Some(s(10)));
        assert_eq!(call_timeout(s(10), now + s(3), now), Some(s(3)));
        assert_eq!(call_timeout(s(10), now, now), None, "nothing left");
        assert_eq!(call_timeout(s(10), now, now + s(1)), None, "already past");
    }

    #[test]
    fn only_a_gone_server_or_a_deadline_asks_for_a_probe() {
        assert!(CallError::Deadline.suspects_server());
        assert!(CallError::Unavailable("reset".into()).suspects_server());
        assert!(
            !CallError::Model {
                code: "InvalidArgument".into(),
                message: "bad shape".into()
            }
            .suspects_server()
        );
        assert!(!CallError::RegionUnknown("x".into()).suspects_server());
    }

    /// A filling without a server, standing in for the model server in engine tests.
    struct Echo;

    impl ModelCalls for Echo {
        async fn infer<'a>(&'a self, call: InferCall<'a>) -> Result<Vec<OutputTensor>, CallError> {
            let left = call_timeout(Duration::from_secs(1), call.deadline, Instant::now())
                .ok_or(CallError::Deadline)?;
            tokio::time::sleep(Duration::from_millis(500)).await;
            if left < Duration::from_millis(500) {
                return Err(CallError::Deadline);
            }
            Ok(call
                .outputs
                .iter()
                .map(|n| OutputTensor {
                    name: (*n).to_string(),
                    datatype: "FP32".into(),
                    shape: vec![1],
                    data: Bytes::from_static(&[0; 4]),
                })
                .collect())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_dropped_call_is_cancelled_and_a_filling_can_stand_in() {
        let echo = Echo;
        let outputs = ["a", "b"];
        let call = InferCall {
            model: "m",
            inputs: &[],
            outputs: &outputs,
            deadline: Instant::now() + Duration::from_secs(5),
        };
        let got = echo.infer(call).await.unwrap();
        assert_eq!(
            got.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        let call = InferCall {
            model: "m",
            inputs: &[],
            outputs: &outputs,
            deadline: Instant::now() + Duration::from_secs(5),
        };
        let raced = tokio::time::timeout(Duration::from_millis(100), echo.infer(call)).await;
        assert!(
            raced.is_err(),
            "the caller gave up first; dropping the future ends the call"
        );
        let call = InferCall {
            model: "m",
            inputs: &[],
            outputs: &outputs,
            deadline: Instant::now() + Duration::from_millis(200),
        };
        assert_eq!(echo.infer(call).await, Err(CallError::Deadline));
    }
}
