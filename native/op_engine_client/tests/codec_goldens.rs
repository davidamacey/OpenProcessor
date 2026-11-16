//! The BYTES codec against bytes Triton's own `serialize_byte_tensor` produced
//! (`scripts/codegen/frame_engine_bytes_goldens.py`).

use op_engine_client::codec::{pack_bytes, unpack_bytes};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Goldens {
    small: Vec<Small>,
    long: Vec<Long>,
}

#[derive(Deserialize)]
struct Small {
    id: String,
    elements_hex: Vec<String>,
    packed_hex: String,
}

#[derive(Deserialize)]
struct Long {
    id: String,
    byte: u8,
    times: usize,
    packed_len: usize,
    packed_sha256: String,
}

fn goldens() -> Goldens {
    let text = include_str!("goldens/bytes_codec.json");
    serde_json::from_str(text).expect("the golden file parses")
}

fn unhex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0, "odd hex length");
    (0..text.len() / 2)
        .map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn packing_each_element_gives_the_bytes_triton_serialises() {
    for case in goldens().small {
        let packed: Vec<u8> = case
            .elements_hex
            .iter()
            .flat_map(|h| pack_bytes(&unhex(h)).to_vec())
            .collect();
        assert_eq!(hex(&packed), case.packed_hex, "{}", case.id);
    }
}

#[test]
fn unpacking_triton_bytes_gives_the_elements_back() {
    for case in goldens().small {
        let raw = unhex(&case.packed_hex);
        let items = unpack_bytes(&raw).expect("a whole tensor");
        let elements: Vec<String> = items.iter().map(|i| hex(i)).collect();
        assert_eq!(elements, case.elements_hex, "{}", case.id);
    }
}

#[test]
fn long_elements_pack_to_the_bytes_triton_serialises() {
    for case in goldens().long {
        let packed = pack_bytes(&vec![case.byte; case.times]);
        assert_eq!(packed.len(), case.packed_len, "{}", case.id);
        let digest: String = hex(&Sha256::digest(&packed));
        assert_eq!(digest, case.packed_sha256, "{}", case.id);
        let items = unpack_bytes(&packed).expect("a whole tensor");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].len(), case.times);
    }
}

#[test]
fn a_tensor_cut_short_is_refused() {
    let whole = pack_bytes(b"abcdef");
    for cut in 1..whole.len() {
        assert!(
            unpack_bytes(&whole[..cut]).is_none(),
            "cut at {cut} of {}",
            whole.len()
        );
    }
}
