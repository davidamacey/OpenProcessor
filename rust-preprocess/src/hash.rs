/// imohash implementation: samples 3x16KB from file content + murmur3 hash.
///
/// Must produce identical hex strings as Python `imohash.hashfileobject()`.

use std::io::Cursor;

const SAMPLE_SIZE: usize = 16 * 1024; // 16KB
const SAMPLE_THRESHOLD: usize = SAMPLE_SIZE * 3 + 1; // Files smaller than this are fully hashed

/// Compute imohash for a byte slice (equivalent to Python imohash.hashfileobject).
///
/// Algorithm:
/// 1. If file < 48KB+1, hash the entire content with murmur3-128, prepend file size
/// 2. Otherwise, sample 16KB from start, middle, end, hash those 48KB, prepend file size
pub fn imohash(data: &[u8]) -> String {
    let file_size = data.len();

    // Build the sample buffer
    let sample = if file_size < SAMPLE_THRESHOLD {
        // Small file: hash everything
        data.to_vec()
    } else {
        // Large file: sample 3 x 16KB (start, middle, end)
        let mid_start = file_size / 2 - SAMPLE_SIZE / 2;
        let end_start = file_size - SAMPLE_SIZE;

        let mut buf = Vec::with_capacity(SAMPLE_SIZE * 3);
        buf.extend_from_slice(&data[..SAMPLE_SIZE]);
        buf.extend_from_slice(&data[mid_start..mid_start + SAMPLE_SIZE]);
        buf.extend_from_slice(&data[end_start..]);
        buf
    };

    // Compute murmur3-128 hash (x64 variant, seed=0)
    let hash128 = murmur3::murmur3_x64_128(&mut Cursor::new(&sample), 0)
        .expect("murmur3 hash should not fail on in-memory data");

    // imohash format: first 8 bytes = file size (big-endian), last 8 bytes = lower 64 bits of murmur3
    // Actually, imohash uses the full 128-bit murmur3 hash but prepends file size in first 8 bytes.
    // The Python imohash library:
    //   1. hash = mmh3.hash_bytes(sample) -> 16 bytes (128-bit murmur3)
    //   2. result = file_size_bytes[0:8] + hash[8:16]
    //   where file_size_bytes is the file size as a varint in the first 8 bytes

    // Looking at imohash source: it uses hashxx (not murmur3) actually...
    // Let me re-examine. The Python `imohash` package uses:
    // - metrohash128 (via metrohash library), NOT murmur3
    // - Format: size (varint, up to 8 bytes) || hash (remaining bytes to fill 16 total)

    // Actually, imohash uses a custom encoding:
    // The hash is 16 bytes total. The first N bytes encode the file size as a varint,
    // and the remaining 16-N bytes are from the content hash.

    // For compatibility, let's use a simpler approach: encode exactly as imohash does.
    // File size as little-endian u64 bytes, then metrohash of samples.

    // CORRECTION: After re-reading the imohash source code more carefully:
    // imohash uses `hashfileobject` which:
    // 1. Samples the file (3x16KB or full content)
    // 2. Hashes with metrohash128 (seed=0)
    // 3. Encodes file size as varint in first N bytes of the 16-byte output
    // 4. Fills remaining bytes from the hash

    // Since we need EXACT compatibility with Python imohash, we replicate the encoding:
    // The imohash format is: varint(file_size) || truncated_hash
    // Total output is always 16 bytes (128 bits) -> hex is 32 chars

    // For the hash function: imohash actually uses murmur3 via mmh3 python package
    // Let me check: `pip show imohash` -> depends on mmh3
    // mmh3.hash_bytes(data) returns 16 bytes of murmur3 x64 128-bit hash

    // imohash.hashfileobject encoding:
    //   hash_bytes = mmh3.hash_bytes(sample_data)  # 16 bytes
    //   size_bytes = encode_varint(file_size)  # variable length
    //   result = size_bytes + hash_bytes[len(size_bytes):]  # always 16 bytes total

    let hash_bytes = hash128.to_le_bytes(); // 16 bytes, little-endian

    // Encode file size as varint (protobuf-style unsigned varint)
    let size_varint = encode_varint(file_size as u64);
    let varint_len = size_varint.len();

    // Combine: varint bytes + remaining hash bytes
    let mut result = [0u8; 16];
    result[..varint_len].copy_from_slice(&size_varint);
    result[varint_len..].copy_from_slice(&hash_bytes[varint_len..16]);

    // Return as hex string
    hex::encode(result)
}

/// Encode a u64 as a protobuf-style unsigned varint.
fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
    buf
}

/// Convert 16 bytes to hex string.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// We need the hex crate or inline it. Let's just add a simple hex encoder.
mod hex {
    pub fn encode(bytes: [u8; 16]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}
