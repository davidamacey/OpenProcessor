//! The renditions' geometry, interleave and output against the Python engine.

use serde_json::Value;

use super::*;
use crate::wire::contract::contract;

const RENDITIONS: &str = include_str!("../../tests/golden/renditions.json");

fn spec(side: u32, quality: u32) -> RenditionSpec {
    RenditionSpec {
        name: format!("r{side}"),
        side,
        quality,
        format: "jpeg".into(),
    }
}

#[test]
fn the_interleave_equals_numpys_copy_of_the_transposed_view() {
    let g: Value = serde_json::from_str(RENDITIONS).expect("JSON");
    let rows = g["interleave"].as_array().expect("rows");
    assert!(rows.len() >= 5);
    for r in rows {
        let bytes = |v: &Value| -> Vec<u8> {
            v.as_array()
                .expect("bytes")
                .iter()
                .map(|b| u8::try_from(b.as_u64().expect("a byte")).expect("a byte"))
                .collect()
        };
        let (planar, interleaved) = (bytes(&r[2]), bytes(&r[3]));
        let pixels = (r[0].as_u64().expect("w") * r[1].as_u64().expect("h")) as usize;
        assert_eq!(planar.len(), 3 * pixels);
        assert_eq!(interleave_host(&planar), interleaved, "{}x{}", r[0], r[1]);
    }
}

#[test]
fn every_pixel_has_a_thread() {
    let l = interleave_launch(129).expect("a launch");
    assert_eq!((l.grid, l.block), ((2, 1, 1), (128, 1, 1)));
    assert_eq!(interleave_launch(128).expect("a launch").grid, (1, 1, 1));
    assert!(interleave_launch(0).is_err());
}

#[test]
fn a_rendition_is_a_whole_frame_lanczos_resample_or_the_frame_itself() {
    let r = plan(4000, 3000, &[spec(256, 80), spec(8000, 90)]).expect("a plan");
    assert_eq!(r[0].size, (256, 192));
    assert_eq!(r[0].quality, 80);
    assert_eq!(
        r[0].resample,
        Some(Resample {
            roi: [0, 0, 4000, 3000],
            size: [256, 192],
            window: [0, 0, 256, 192],
            dst_offset: 0,
            plane: 256 * 192,
            row_stride: 256,
            filter: Filter::Lanczos,
        })
    );
    assert_eq!((r[1].size, &r[1].resample), ((4000, 3000), &None));
    assert_eq!(sides(&[spec(256, 80), spec(1024, 85)]), [256, 1024]);
}

#[test]
fn the_output_is_the_contracts_rendition_tensor() {
    assert_eq!(DERIVATIVE_JPEGS, contract().derivatives.output);
    let (name, t) = output(vec![Bytes::from_static(b"a"), Bytes::from_static(b"b")]);
    assert_eq!(name, DERIVATIVE_JPEGS);
    assert_eq!(
        t,
        Tensor::Bytes(vec![Bytes::from_static(b"a"), Bytes::from_static(b"b")])
    );
}
