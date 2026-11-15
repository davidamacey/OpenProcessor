use bytes::Bytes;

use super::*;

const RECTS: [[i64; 4]; 3] = [[158, 84, 470, 502], [0, 0, 48, 16], [938, 656, 1000, 750]];

fn crop_spec(batch: u32) -> CropEmbedSpec {
    CropEmbedSpec {
        model: "pe_crop_encoder".into(),
        size: 224,
        batch,
        crop_recipe: "letterbox".into(),
        fill: 128,
    }
}

fn answer(rows: usize, dim: usize, first: f32) -> Vec<OutputTensor> {
    let data: Vec<u8> = (0..rows * dim)
        .flat_map(|i| (first + i as f32).to_le_bytes())
        .collect();
    vec![OutputTensor {
        name: OUTPUT.into(),
        datatype: "FP32".into(),
        shape: vec![rows as i64, dim as i64],
        data: Bytes::from(data),
    }]
}

#[test]
fn the_frame_square_is_the_centre_of_the_shorter_edge_fit() {
    let spec = EmbedSpec {
        model: "pe".into(),
        size: 336,
        frame_recipe: "whole_frame".into(),
    };
    let (input, r) = frame_square(&spec, 1000, 750).expect("square");
    assert_eq!(input.shape, [1, 3, 336, 336]);
    // whole_square_geometry(1000, 750, 336): 448 x 336, the window from column 56
    assert_eq!((r.size, r.window), ([448, 336], [56, 0, 336, 336]));
    assert_eq!((r.dst_offset, r.plane, r.row_stride), (0, 336 * 336, 336));
    assert_eq!(r.extent(), Some(input.bytes()));
}

#[test]
fn crops_go_in_chunks_of_the_plan_batch_each_square_letterboxed_in_its_slot() {
    let plan = crop_plan(&crop_spec(2), None, &RECTS, Some(&[2, 7, 16])).expect("plan");
    assert_eq!(plan.picked, [true, true, true]);
    let sizes: Vec<[i64; 4]> = plan.chunks.iter().map(|c| c.input.shape).collect();
    assert_eq!(sizes, [[2, 3, 224, 224], [1, 3, 224, 224]]);
    let square = 3 * 224 * 224;
    let [first, second] = &plan.chunks[0].resamples[..] else {
        panic!("two squares in the first call");
    };
    // letterbox_geometry(312, 418, 224) = 167 x 224 at left 28; (48, 16) = 224 x 75 at top 74
    assert_eq!(first.roi, [158, 84, 312, 418]);
    assert_eq!((first.size, first.dst_offset), ([167, 224], 28));
    assert_eq!(second.roi, [0, 0, 48, 16]);
    assert_eq!(
        (second.size, second.dst_offset),
        ([224, 75], square + 74 * 224)
    );
    let third = &plan.chunks[1].resamples[0];
    assert_eq!(
        (third.roi, third.size, third.dst_offset),
        ([938, 656, 62, 94], [148, 224], 38)
    );
    for c in &plan.chunks {
        assert_eq!(c.fill, 128);
        for r in &c.resamples {
            assert!(r.extent().expect("extent") <= c.input.bytes());
        }
    }
    let one = crop_plan(&crop_spec(32), None, &RECTS, None).expect("plan");
    assert_eq!(one.chunks.len(), 1);
    assert!(
        crop_plan(&crop_spec(32), None, &[], None)
            .expect("plan")
            .chunks
            .is_empty()
    );
}

#[test]
fn embed_classes_picks_the_wanted_detections_only() {
    let wanted = BTreeSet::from([2, 16]);
    let plan = crop_plan(&crop_spec(32), Some(&wanted), &RECTS, Some(&[2, 7, 16])).expect("plan");
    assert_eq!(plan.picked, [true, false, true]);
    let rois: Vec<[i64; 4]> = plan.chunks[0].resamples.iter().map(|r| r.roi).collect();
    assert_eq!(rois, [[158, 84, 312, 418], [938, 656, 62, 94]]);
}

#[test]
fn the_vectors_of_every_call_are_joined_and_nothing_picked_is_zero_by_zero() {
    let [vectors, embedded] = crop_outputs(
        &[true, false, true],
        &[answer(1, 4, 0.0), answer(1, 4, 10.0)],
    )
    .expect("outputs");
    assert_eq!(
        vectors,
        (
            "crop_embeddings",
            Tensor::Fp32 {
                shape: vec![2, 4],
                data: vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]
            }
        )
    );
    assert_eq!(
        embedded.1,
        Tensor::Bool {
            shape: vec![3],
            data: vec![true, false, true]
        }
    );
    let [none, _] = crop_outputs(&[false], &[]).expect("outputs");
    assert_eq!(
        none.1,
        Tensor::Fp32 {
            shape: vec![0, 0],
            data: vec![]
        }
    );
    assert!(crop_outputs(&[true, true], &[answer(1, 4, 0.0), answer(1, 8, 0.0)]).is_err());
    assert_eq!(
        whole_embedding(&answer(1, 3, 5.0)).expect("vector"),
        Tensor::Fp32 {
            shape: vec![3],
            data: vec![5.0, 6.0, 7.0]
        }
    );
    assert!(whole_embedding(&answer(0, 3, 0.0)).is_err());
}
