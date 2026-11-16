//! The BYTES tensor of KServe v2 as `raw_input_contents` and `raw_output_contents` carry it: each
//! element is its length as 4 bytes little-endian, then its bytes
//! (<https://kserve.github.io/website/latest/modelserving/data_plane/binary_tensor_data_extension/>;
//! Triton's `serialize_byte_tensor` writes the same bytes).
//!
//! Written against the protocol, not shared with the engine's own decoder, so a defect in one
//! cannot hide in the other (plan 17 N0, plan 18 Q5).

use bytes::{BufMut, Bytes, BytesMut};

/// One BYTES element as KServe's raw contents carry it. A tensor of several elements is these
/// packs one after the other.
///
/// An element past `u32::MAX` bytes cannot be told apart by its prefix; its prefix saturates and
/// the receiver refuses the tensor, so a message that large (the transport's limit is far below)
/// is never taken for a different one.
#[must_use]
pub fn pack_bytes(item: &[u8]) -> Bytes {
    let mut out = BytesMut::with_capacity(4 + item.len());
    out.put_u32_le(u32::try_from(item.len()).unwrap_or(u32::MAX));
    out.put_slice(item);
    out.freeze()
}

/// The elements of a BYTES tensor's raw contents; `None` when it is cut short (a prefix or an
/// element runs past the end).
#[must_use]
pub fn unpack_bytes(raw: &[u8]) -> Option<Vec<&[u8]>> {
    let mut items = Vec::new();
    let mut at = 0usize;
    while at < raw.len() {
        let size = u32::from_le_bytes(raw.get(at..at.checked_add(4)?)?.try_into().ok()?) as usize;
        at += 4;
        items.push(raw.get(at..at.checked_add(size)?)?);
        at += size;
    }
    Some(items)
}
