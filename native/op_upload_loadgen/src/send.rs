//! One upload request on a shared connection pool, and how its answer is classified.
//!
//! The client is hyper-util's legacy `Client` over hyper 1.12's HTTP/1 client
//! (<https://docs.rs/hyper-util/0.1.21/hyper_util/client/legacy/struct.Client.html>), whose pool
//! keeps idle connections per host (`pool_max_idle_per_host`, set to the in-flight count so a
//! closed loop of N requests reuses N connections and never reconnects). HTTP/1.1 is what a browser
//! uses through the frontend's nginx (`frontend/nginx.conf` proxies to the API upstream over
//! HTTP/1.1 with `proxy_request_buffering off`), so no HTTP/2 is negotiated; the connector speaks
//! plain `http` only.
//! The body is `http_body_util::StreamBody` over [`Frames`]
//! (<https://docs.rs/http-body-util/0.1.5/http_body_util/struct.StreamBody.html>).

use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::stream::{self, Iter};
use http_body_util::{BodyExt, Limited, StreamBody};
use hyper::header::{ACCEPT, CONTENT_LENGTH, CONTENT_TYPE};
use hyper::{Method, Request, StatusCode, Uri};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use serde::Deserialize;
use std::collections::BTreeMap;

use crate::multipart::{self, FieldOrder, Fields, Frames, Photo};
use crate::record::{Outcome, RequestRecord};

/// The largest answer body read; an answer over it is a `client:response_body` failure.
const MAX_ANSWER_BYTES: usize = 16 * 1024 * 1024;

/// The request body type of this client.
pub type UploadBody = StreamBody<Iter<Frames>>;

/// A client whose pool holds up to `in_flight` idle connections to the host.
pub fn pooled_client(in_flight: usize) -> Client<HttpConnector, UploadBody> {
    let mut connector = HttpConnector::new();
    connector.set_nodelay(true);
    connector.set_connect_timeout(Some(Duration::from_secs(10)));
    Client::builder(TokioExecutor::new())
        .pool_max_idle_per_host(in_flight)
        .build(connector)
}

#[derive(Debug, Deserialize)]
struct Reply {
    results: Option<Vec<Item>>,
}

#[derive(Debug, Deserialize)]
struct Item {
    status: String,
    #[serde(default)]
    error_kind: Option<String>,
}

/// Classify an answer: a non-2xx status is `http:<status>`; a 2xx whose body has a `results`
/// list of one entry per photo is counted by each entry's `status` (`success`, `duplicate`,
/// `failed` by `error_kind`); a 2xx without `results` (a sink) is `read: false`; a `results` list
/// of the wrong length is `protocol:results_count`.
pub fn classify(status: StatusCode, body: &[u8], photos: usize) -> Outcome {
    if !status.is_success() {
        return Outcome::Failed {
            key: format!("http:{}", status.as_u16()),
        };
    }
    let Some(results) = serde_json::from_slice::<Reply>(body)
        .ok()
        .and_then(|r| r.results)
    else {
        return Outcome::Answered {
            success: 0,
            duplicate: 0,
            failed_by_kind: BTreeMap::new(),
            read: false,
        };
    };
    if results.len() != photos {
        return Outcome::Failed {
            key: "protocol:results_count".into(),
        };
    }
    let (mut success, mut duplicate) = (0, 0);
    let mut failed_by_kind: BTreeMap<String, u64> = BTreeMap::new();
    for item in results {
        match item.status.as_str() {
            "success" => success += 1,
            "duplicate" => duplicate += 1,
            "failed" => {
                let kind = item.error_kind.unwrap_or_else(|| "unknown".into());
                *failed_by_kind.entry(kind).or_default() += 1;
            }
            other => *failed_by_kind.entry(format!("status:{other}")).or_default() += 1,
        }
    }
    Outcome::Answered {
        success,
        duplicate,
        failed_by_kind,
        read: true,
    }
}

/// Everything a request needs besides its number.
pub struct Sender {
    /// The pool-backed client.
    pub client: Client<HttpConnector, UploadBody>,
    /// The upload route's URI.
    pub uri: Uri,
    /// One request's time limit, body and answer included.
    pub timeout: Duration,
    /// The field order of the form.
    pub order: FieldOrder,
    /// The text fields.
    pub fields: Fields,
    /// Largest frame of a photo part, in bytes.
    pub chunk: usize,
    /// The list, in memory.
    pub photos: Arc<Vec<Photo>>,
    /// Which photos each request carries.
    pub packs: Vec<Range<usize>>,
    /// The loop's start: every record's times are milliseconds after it.
    pub t0: Instant,
}

enum Sent {
    Answer(StatusCode, Vec<u8>),
    Failed(&'static str),
}

impl Sender {
    /// Send request `seq` (its pack is `seq` modulo the number of packs) and record how it went.
    pub async fn send(&self, seq: u64) -> RequestRecord {
        let pack = usize::try_from(seq % self.packs.len() as u64).unwrap_or(0);
        let range = self.packs[pack].clone();
        let photos = &self.photos[range];
        let built = multipart::build(
            self.order,
            &multipart::boundary(seq),
            &self.fields,
            photos,
            self.chunk,
        );
        let (content_type, content_length) = (built.content_type.clone(), built.content_length);
        let stamps = Arc::clone(&built.stamps);
        let request = Request::builder()
            .method(Method::POST)
            .uri(self.uri.clone())
            .header(CONTENT_TYPE, content_type)
            .header(CONTENT_LENGTH, content_length)
            .header(ACCEPT, "application/json")
            .body(StreamBody::new(stream::iter(built.frames)));
        let started = Instant::now();
        let sent = match request {
            Ok(request) => self.exchange(request).await,
            Err(_) => Sent::Failed("client:send"),
        };
        let ended = Instant::now();
        let ms =
            |at: Instant, from: Instant| at.saturating_duration_since(from).as_secs_f64() * 1e3;
        let photo_count = photos.len();
        let outcome = match sent {
            Sent::Answer(status, body) => classify(status, &body, photo_count),
            Sent::Failed(key) => Outcome::Failed { key: key.into() },
        };
        RequestRecord {
            seq,
            pack,
            photos: photo_count,
            photo_bytes: photos.iter().map(|p| p.data.len() as u64).sum(),
            body_bytes: content_length,
            started_ms: ms(started, self.t0),
            finished_ms: ms(ended, self.t0),
            parts_taken_ms: stamps.taken().iter().map(|&t| ms(t, started)).collect(),
            response_ms: ms(ended, started),
            outcome,
        }
    }

    async fn exchange(&self, request: Request<UploadBody>) -> Sent {
        let call = async {
            let response = match self.client.request(request).await {
                Ok(r) => r,
                Err(e) if e.is_connect() => return Sent::Failed("client:connect"),
                Err(_) => return Sent::Failed("client:send"),
            };
            let status = response.status();
            match Limited::new(response.into_body(), MAX_ANSWER_BYTES)
                .collect()
                .await
            {
                Ok(collected) => Sent::Answer(status, collected.to_bytes().to_vec()),
                Err(_) => Sent::Failed("client:response_body"),
            }
        };
        tokio::time::timeout(self.timeout, call)
            .await
            .unwrap_or(Sent::Failed("client:timeout"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(items: &str) -> Vec<u8> {
        format!("{{\"status\":\"success\",\"results\":[{items}]}}").into_bytes()
    }

    #[test]
    fn a_refusal_is_keyed_by_its_status() {
        for (code, key) in [(413, "http:413"), (503, "http:503"), (422, "http:422")] {
            let status = StatusCode::from_u16(code).unwrap();
            assert_eq!(
                classify(status, b"{}", 1),
                Outcome::Failed { key: key.into() }
            );
        }
    }

    #[test]
    fn per_photo_statuses_are_counted_and_failures_keep_their_kind() {
        let b = body(
            r#"{"status":"success"},{"status":"duplicate"},{"status":"duplicate"},
               {"status":"failed","error_kind":"decode_failed"},{"status":"failed"}"#,
        );
        let Outcome::Answered {
            success,
            duplicate,
            failed_by_kind,
            read,
        } = classify(StatusCode::OK, &b, 5)
        else {
            panic!("expected an answer");
        };
        assert_eq!((success, duplicate, read), (1, 2, true));
        assert_eq!(failed_by_kind["decode_failed"], 1);
        assert_eq!(failed_by_kind["unknown"], 1);
    }

    #[test]
    fn a_wrong_number_of_results_is_a_protocol_failure() {
        assert_eq!(
            classify(StatusCode::OK, &body(r#"{"status":"success"}"#), 2),
            Outcome::Failed {
                key: "protocol:results_count".into()
            }
        );
    }

    #[test]
    fn a_2xx_without_results_is_unread() {
        for b in [&b""[..], b"ok", b"{}", b"{\"results\":null}"] {
            assert!(matches!(
                classify(StatusCode::OK, b, 3),
                Outcome::Answered {
                    read: false,
                    success: 0,
                    ..
                }
            ));
        }
    }
}
