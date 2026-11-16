//! How a photo reaches the engine: its bytes in the request (`image`), or the name of a file the
//! engine reads itself from a read-only root (`source`, the `read_by_key` extension of
//! `contracts/json/frame_engine_v1.json`; only the native engine has it).
//!
//! The `source` document is `{"imohash": "<32 hex>", "key": "<landing or source key>", "size": N}`
//! written as Python's `json.dumps(sort_keys=True)` writes it; the engine's parser accepts the
//! fields in any order, and `contracts/json/frame_engine_source_cases.json` pins this text.

use bytes::Bytes;
use op_local_file::FileRef;

use crate::codec::pack_bytes;
use crate::plan::pyjson;

/// One photo, as the request carries it: exactly one of the two, by the type.
#[derive(Debug, Clone, Copy)]
pub enum PhotoInput<'a> {
    /// The encoded file's bytes, sent in the request.
    Bytes(&'a [u8]),
    /// A file under the landing zone or a source root, which the engine reads itself and checks
    /// against the size and the sampled hash the server holds for it.
    File(&'a FileRef),
}

impl<'a> From<&'a [u8]> for PhotoInput<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        Self::Bytes(bytes)
    }
}

impl<'a, const N: usize> From<&'a [u8; N]> for PhotoInput<'a> {
    fn from(bytes: &'a [u8; N]) -> Self {
        Self::Bytes(bytes)
    }
}

impl<'a> From<&'a Vec<u8>> for PhotoInput<'a> {
    fn from(bytes: &'a Vec<u8>) -> Self {
        Self::Bytes(bytes)
    }
}

impl<'a> From<&'a FileRef> for PhotoInput<'a> {
    fn from(file: &'a FileRef) -> Self {
        Self::File(file)
    }
}

impl PhotoInput<'_> {
    /// The input's name and its BYTES element in raw-content form (a 4-byte length, then the
    /// bytes, copied once): `image` and the photo, or `source` and the document.
    ///
    /// # Errors
    ///
    /// The `serde_json` error of writing the document (never in practice: it holds a string and
    /// two numbers).
    pub(crate) fn packed(&self) -> Result<(&'static str, Bytes), serde_json::Error> {
        match self {
            Self::Bytes(bytes) => Ok(("image", pack_bytes(bytes))),
            Self::File(file) => Ok(("source", pack_bytes(source_document(file)?.as_bytes()))),
        }
    }
}

/// The text of the `source` document for `file`.
///
/// # Errors
///
/// The `serde_json` error of writing it.
pub fn source_document(file: &FileRef) -> Result<String, serde_json::Error> {
    pyjson::dumps(&serde_json::json!({
        "imohash": file.claim().imohash.to_hex(),
        "key": file.key(),
        "size": file.claim().size,
    }))
}
