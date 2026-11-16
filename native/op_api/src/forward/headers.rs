//! What the forwarder does to the header sets of a request and its response.
//!
//! Hop-by-hop fields describe one connection, not the message, so a proxy removes them and each
//! connection gets its own (RFC 9110 section 7.6.1,
//! <https://www.rfc-editor.org/rfc/rfc9110#section-7.6.1>): `Connection`, `Keep-Alive`,
//! `Proxy-Authenticate`, `Proxy-Authorization`, `TE`, `Trailer`, `Transfer-Encoding`, `Upgrade`
//! (and the older `Proxy-Connection`), plus every field name the `Connection` field lists.
//!
//! A body streams through without being measured, so its framing is each hop's own:
//!
//! - `Transfer-Encoding` is removed on both sides; hyper's server has already decoded a chunked
//!   request body, and the client side chunks it again when the length is not known.
//! - `Content-Length` is kept as it came. hyper sends a body whose length is known with that
//!   length and checks the byte count while streaming, so a body that is longer or shorter than
//!   it says fails the request; nothing is buffered to count.

use std::net::IpAddr;

use http::HeaderMap;
use http::header::{
    CONNECTION, CONTENT_LENGTH, HOST, HeaderName, HeaderValue, PROXY_AUTHENTICATE,
    PROXY_AUTHORIZATION, TE, TRAILER, TRANSFER_ENCODING, UPGRADE,
};

use crate::middleware::request_id::X_REQUEST_ID;

const KEEP_ALIVE: HeaderName = HeaderName::from_static("keep-alive");
const PROXY_CONNECTION: HeaderName = HeaderName::from_static("proxy-connection");
const X_FORWARDED_FOR: HeaderName = HeaderName::from_static("x-forwarded-for");
const X_FORWARDED_PROTO: HeaderName = HeaderName::from_static("x-forwarded-proto");

const HOP_BY_HOP: [HeaderName; 9] = [
    CONNECTION,
    KEEP_ALIVE,
    PROXY_AUTHENTICATE,
    PROXY_AUTHORIZATION,
    TE,
    TRAILER,
    TRANSFER_ENCODING,
    UPGRADE,
    PROXY_CONNECTION,
];

/// Fields a `Connection` list may not take away: the ones the message cannot be understood
/// without, and the id that ties the request to its logs.
const KEPT_EVEN_IF_LISTED: [HeaderName; 3] = [HOST, CONTENT_LENGTH, X_REQUEST_ID];

/// Remove the hop-by-hop fields, and the fields the `Connection` field names.
pub fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let listed: Vec<HeaderName> = headers
        .get_all(CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|token| HeaderName::from_bytes(token.trim().as_bytes()).ok())
        .filter(|name| !KEPT_EVEN_IF_LISTED.contains(name))
        .collect();
    for name in HOP_BY_HOP.iter().chain(&listed) {
        headers.remove(name);
    }
}

/// Tell the Python API who connected. `X-Forwarded-For` is the peer's address and nothing else:
/// the front is the edge, so a chain a client wrote is dropped. uvicorn trusts the front
/// (`--forwarded-allow-ips`) and takes the first entry as the client address it logs, so a kept
/// client entry would let a client choose it; before the front uvicorn ignored the header from any
/// peer that was not loopback. `X-Forwarded-Proto` is `http` unless a proxy in front of this one
/// has already said what the client used (`src/core/request_origin.py` reads that header as it
/// always did). `Host` and `X-Forwarded-Host` are left as the client sent them, which is also
/// what `request_origin.py` reads.
///
/// Call it after [`strip_hop_by_hop`]: a client that names `X-Forwarded-For` in `Connection`
/// must not be able to make the proxy drop its own entry.
pub fn add_forwarded(headers: &mut HeaderMap, peer: IpAddr) {
    if let Ok(value) = HeaderValue::from_str(&peer.to_canonical().to_string()) {
        headers.insert(X_FORWARDED_FOR, value);
    }
    if !headers.contains_key(X_FORWARDED_PROTO) {
        headers.insert(X_FORWARDED_PROTO, HeaderValue::from_static("http"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.append(
                HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        map
    }

    #[test]
    fn the_standard_hop_by_hop_fields_go_and_the_rest_stay() {
        let mut map = headers(&[
            ("connection", "keep-alive"),
            ("keep-alive", "timeout=5"),
            ("proxy-authorization", "x"),
            ("te", "trailers"),
            ("trailer", "x-t"),
            ("transfer-encoding", "chunked"),
            ("upgrade", "websocket"),
            ("proxy-connection", "keep-alive"),
            ("accept", "*/*"),
            ("content-length", "5"),
            ("authorization", "Bearer t"),
        ]);
        strip_hop_by_hop(&mut map);
        let left: Vec<&str> = map.keys().map(HeaderName::as_str).collect();
        assert_eq!(left.len(), 3, "{left:?}");
        for kept in ["accept", "content-length", "authorization"] {
            assert!(map.contains_key(kept), "{kept}");
        }
    }

    #[test]
    fn a_field_the_connection_field_names_goes_with_it() {
        let mut map = headers(&[
            ("connection", "close, X-Private, x-other"),
            ("x-private", "1"),
            ("x-other", "2"),
            ("x-kept", "3"),
        ]);
        strip_hop_by_hop(&mut map);
        assert!(map.contains_key("x-kept"));
        assert!(!map.contains_key("x-private") && !map.contains_key("x-other"));
    }

    #[test]
    fn the_connection_field_cannot_take_away_the_host_length_or_request_id() {
        let mut map = headers(&[
            ("connection", "host, content-length, x-request-id"),
            ("host", "example.test"),
            ("content-length", "3"),
            ("x-request-id", "abc"),
        ]);
        strip_hop_by_hop(&mut map);
        for kept in ["host", "content-length", "x-request-id"] {
            assert!(map.contains_key(kept), "{kept}");
        }
    }

    #[test]
    fn the_peer_replaces_any_client_forwarded_for_and_the_proto_defaults_to_http() {
        let mut map = headers(&[
            ("x-forwarded-for", "203.0.113.9"),
            ("x-forwarded-for", "198.51.100.2"),
        ]);
        add_forwarded(&mut map, "192.0.2.7".parse().unwrap());
        let seen: Vec<_> = map.get_all("x-forwarded-for").iter().collect();
        assert_eq!(seen, ["192.0.2.7"]);
        assert_eq!(map["x-forwarded-proto"], "http");
    }

    #[test]
    fn a_proto_a_proxy_in_front_already_gave_is_kept_and_a_mapped_address_is_written_as_v4() {
        let mut map = headers(&[
            ("x-forwarded-proto", "https"),
            ("x-forwarded-host", "photos.test"),
        ]);
        add_forwarded(&mut map, "::ffff:192.0.2.7".parse().unwrap());
        assert_eq!(map["x-forwarded-proto"], "https");
        assert_eq!(map["x-forwarded-host"], "photos.test");
        assert_eq!(map["x-forwarded-for"], "192.0.2.7");
    }
}
