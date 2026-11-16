//! The content claim: what the server knows about a file when it names it to the engine. The
//! engine reads the file once, then checks the bytes it holds against the claim, so a file that
//! changed after the server looked at it is refused instead of being decoded under the old
//! record. The size alone catches most edits; the `imohash` (three 16 KiB samples) catches a
//! same-size rewrite at the price of reading 48 KiB it already holds.

use std::fmt;
use std::io::Cursor;

use crate::imohash::{ImoHash, hash_reader};

/// The size and the duplicate hash of a file's bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileClaim {
    /// Bytes in the file.
    pub size: u64,
    /// The `imohash` of the file.
    pub imohash: ImoHash,
}

/// How a file's bytes differ from the claim made for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimMismatch {
    /// The file has this many bytes, not the claimed number.
    Size(u64),
    /// The size is right but the sampled hash is not.
    Hash,
}

impl fmt::Display for ClaimMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Size(found) => write!(f, "the file has {found} bytes, not the claimed size"),
            Self::Hash => f.write_str("the file's sampled hash is not the claimed hash"),
        }
    }
}

impl FileClaim {
    /// The claim for `bytes`.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self {
            size: bytes.len() as u64,
            imohash: hash_of(bytes),
        }
    }

    /// Whether `bytes` are the file this claim was made for.
    ///
    /// # Errors
    ///
    /// [`ClaimMismatch::Size`] when the length differs (checked first, with no hashing),
    /// [`ClaimMismatch::Hash`] when the length matches and the hash does not.
    pub fn check(&self, bytes: &[u8]) -> Result<(), ClaimMismatch> {
        let found = bytes.len() as u64;
        if found != self.size {
            return Err(ClaimMismatch::Size(found));
        }
        if hash_of(bytes) == self.imohash {
            Ok(())
        } else {
            Err(ClaimMismatch::Hash)
        }
    }
}

fn hash_of(bytes: &[u8]) -> ImoHash {
    // an in-memory cursor cannot fail to read or seek
    match hash_reader(&mut Cursor::new(bytes)) {
        Ok(hash) => hash,
        Err(never) => unreachable!("a cursor over a slice does not fail: {never}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imohash::SAMPLE_THRESHOLD;

    fn photo(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 7 + i / 251) as u8).collect()
    }

    #[test]
    fn the_bytes_the_claim_was_made_for_pass_small_and_sampled() {
        for len in [0, 1, 1000, SAMPLE_THRESHOLD as usize, 300_000] {
            let bytes = photo(len);
            assert_eq!(FileClaim::of(&bytes).check(&bytes), Ok(()), "{len}");
        }
    }

    #[test]
    fn another_length_is_a_size_mismatch_before_any_hashing() {
        let bytes = photo(200_000);
        let claim = FileClaim::of(&bytes);
        assert_eq!(
            claim.check(&bytes[..199_999]),
            Err(ClaimMismatch::Size(199_999))
        );
    }

    #[test]
    fn a_same_size_rewrite_inside_a_sample_is_a_hash_mismatch() {
        let bytes = photo(200_000);
        let claim = FileClaim::of(&bytes);
        for at in [0, 16_383, 100_000, 199_999] {
            let mut edited = bytes.clone();
            edited[at] ^= 1;
            assert_eq!(claim.check(&edited), Err(ClaimMismatch::Hash), "byte {at}");
        }
    }

    #[test]
    fn a_hash_reads_back_from_its_hex_and_refuses_other_text() {
        let hash = FileClaim::of(&photo(5000)).imohash;
        assert_eq!(ImoHash::from_hex(&hash.to_hex()), Some(hash));
        for bad in ["", "zz", &hash.to_hex().to_uppercase(), &hash.to_hex()[1..]] {
            assert_eq!(ImoHash::from_hex(bad), None, "{bad:?}");
        }
    }
}
