//! KServe v2 BYTES tensor contents: each element is a 4-byte little-endian length and the bytes
//! (`raw_input_contents` and `raw_output_contents`; the model server's protocol document,
//! "Binary Tensor Data Extension" / `grpc_service.proto` `ModelInferRequest`).

use bytes::{Bytes, BytesMut};

/// A BYTES tensor that does not parse; the text is the Python front's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BytesError {
    /// The contents end inside a length prefix.
    LengthPrefix,
    /// An element's length runs past the end.
    ElementOverrun,
}

impl BytesError {
    /// The id of the request rule in the contract.
    pub fn rule(self) -> &'static str {
        match self {
            Self::LengthPrefix => "bytes_length_prefix",
            Self::ElementOverrun => "bytes_element_overrun",
        }
    }
}

/// The elements of `raw`, each a slice of it (reference counted, never copied: a photo is up to
/// tens of MB).
///
/// # Errors
///
/// The bytes are not a sequence of length-prefixed elements, or an element runs past the end.
pub fn unpack(raw: &Bytes) -> Result<Vec<Bytes>, BytesError> {
    let mut out = Vec::new();
    let n = raw.len();
    let mut i = 0usize;
    while i < n {
        let prefix = raw.get(i..i + 4).ok_or(BytesError::LengthPrefix)?;
        let size = u32::from_le_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]) as usize;
        i += 4;
        let end = i
            .checked_add(size)
            .filter(|e| *e <= n)
            .ok_or(BytesError::ElementOverrun)?;
        out.push(raw.slice(i..end));
        i = end;
    }
    Ok(out)
}

/// The BYTES contents of `items`.
pub fn pack<T: AsRef<[u8]>>(items: &[T]) -> Bytes {
    let total: usize = items.iter().map(|x| 4 + x.as_ref().len()).sum();
    let mut out = BytesMut::with_capacity(total);
    for item in items {
        let item = item.as_ref();
        out.extend_from_slice(&u32::try_from(item.len()).unwrap_or(u32::MAX).to_le_bytes());
        out.extend_from_slice(item);
    }
    out.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elements_round_trip() {
        let items: Vec<&[u8]> = vec![b"", b"a", &[0; 5]];
        let raw = pack(&items);
        let back = unpack(&raw).expect("unpack");
        assert_eq!(back.iter().map(Bytes::as_ref).collect::<Vec<_>>(), items);
        assert_eq!(raw.len(), 4 * 3 + 6);
        assert_eq!(
            &raw[..5],
            &[0, 0, 0, 0, 1][..],
            "a length is four little-endian bytes"
        );
    }

    #[test]
    fn an_element_is_a_view_of_the_message() {
        let raw = pack(&[vec![7u8; 1000], vec![1, 2]]);
        let back = unpack(&raw).expect("unpack");
        let base = raw.as_ptr() as usize;
        let at = back[0].as_ptr() as usize;
        assert_eq!(at, base + 4, "no copy was made");
    }

    #[test]
    fn a_short_prefix_and_a_long_element_are_the_two_faults() {
        assert_eq!(
            unpack(&Bytes::from_static(&[1, 0])),
            Err(BytesError::LengthPrefix)
        );
        let mut raw = 9u32.to_le_bytes().to_vec();
        raw.extend_from_slice(b"ab");
        assert_eq!(unpack(&Bytes::from(raw)), Err(BytesError::ElementOverrun));
        // a length that wraps usize on a 32-bit host is still an overrun
        assert_eq!(
            unpack(&Bytes::from_static(&[255, 255, 255, 255])),
            Err(BytesError::ElementOverrun)
        );
        assert_eq!(unpack(&Bytes::new()), Ok(vec![]));
        assert_eq!(BytesError::LengthPrefix.rule(), "bytes_length_prefix");
        assert_eq!(BytesError::ElementOverrun.rule(), "bytes_element_overrun");
    }
}
