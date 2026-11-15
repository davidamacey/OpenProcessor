use bytes::Bytes;

use super::*;

fn spec() -> DetectorSpec {
    DetectorSpec {
        model: "yolo".into(),
        input_size: 640,
        letterbox_fill: 114,
    }
}

/// `output0` of a fused detector: `rows` then zero rows up to 300.
fn answer(rows: &[[f32; 6]]) -> Vec<OutputTensor> {
    let mut data = Vec::new();
    for r in rows
        .iter()
        .chain(std::iter::repeat_n(&[0.0; 6], 300 - rows.len()))
    {
        data.extend(r.iter().flat_map(|v| v.to_le_bytes()));
    }
    vec![OutputTensor {
        name: OUTPUT.into(),
        datatype: "FP32".into(),
        shape: vec![1, 300, 6],
        data: Bytes::from(data),
    }]
}

const ROWS: [[f32; 6]; 3] = [
    [101.37, 133.71, 300.5, 401.25, 0.91, 2.0],
    [-20.25, 70.0, 30.5, 90.0, 0.5, 7.9],
    [600.0, 500.0, 700.0, 600.0, 0.25, 16.0],
];

#[test]
fn the_canvas_is_the_letterbox_on_the_fill() {
    let c = canvas(&spec(), 1000, 750).expect("canvas");
    assert_eq!(c.input.shape, [1, 3, 640, 640]);
    assert_eq!(c.input.bytes(), 3 * 640 * 640);
    assert_eq!(c.fill, 114);
    assert_eq!(c.resample.size, [640, 480]);
    assert_eq!(c.resample.dst_offset, 80 * 640, "80 rows of margin above");
    assert_eq!((c.resample.plane, c.resample.row_stride), (640 * 640, 640));
    assert!(c.resample.extent().expect("extent") <= c.input.bytes());
}

#[test]
fn the_settle_keeps_rows_with_a_score_and_maps_them_to_the_frame_as_python_does() {
    let lb = letterbox_geometry(1000, 750, 640).expect("geometry");
    let d = settle(&answer(&ROWS), &lb, 1000, 750).expect("settle");
    // Python: crop_rect(undo_letterbox(tuple(box), ...)) of the same float32 rows, numpy 2.5.4
    assert_eq!(
        d.rects,
        [[158, 84, 470, 502], [0, 0, 48, 16], [938, 656, 1000, 750]]
    );
    assert_eq!(d.classes, [2, 7, 16], "astype(np.int32) truncates 7.9");
    assert_eq!(d.scores, [0.91, 0.5, 0.25]);
    assert_eq!(d.boxes[0], [101.37, 133.71, 300.5, 401.25]);
    let none = settle(&answer(&[]), &lb, 1000, 750).expect("settle");
    assert!(none.boxes.is_empty() && none.rects.is_empty());
    let out = outputs(&none);
    assert_eq!(
        out[1],
        (
            "det_boxes",
            Tensor::Fp32 {
                shape: vec![0, 4],
                data: vec![]
            }
        ),
        "an empty answer is (0, 4)"
    );
    assert_eq!(
        out[0].1,
        Tensor::Fp64 {
            shape: vec![3],
            data: vec![0.64, 0.0, 80.0]
        }
    );
}

#[test]
fn a_malformed_answer_is_an_error_not_a_detection() {
    let lb = letterbox_geometry(100, 100, 640).expect("geometry");
    let mut bad = answer(&ROWS);
    bad[0].shape = vec![1, 360, 5];
    assert!(settle(&bad, &lb, 100, 100).is_err());
    let mut flat = answer(&ROWS);
    flat[0].shape = vec![1800];
    assert!(settle(&flat, &lb, 100, 100).is_err());
}

#[test]
fn blur_sums_are_the_frame_then_each_detection_with_empty_ones_zero() {
    let lb = letterbox_geometry(1000, 750, 640).expect("geometry");
    let mut d = settle(&answer(&ROWS), &lb, 1000, 750).expect("settle");
    // a fourth detection entirely left of the frame: its clamped rectangle is empty
    d.boxes.push([-50.0, 100.0, -10.0, 200.0]);
    let (summed, clamped) = blur_rects(&d, 1000, 750);
    assert_eq!(
        summed,
        [
            [0, 0, 1000, 750],
            [158, 84, 470, 502],
            [0, 0, 48, 16],
            [938, 656, 1000, 750]
        ]
    );
    assert_eq!(clamped[3], None);
    let sums = [[1, 2], [3, 4], [5, 6], [7, 8]];
    let t = blur_sums(1000, 750, &clamped, &sums).expect("sums");
    assert_eq!(
        t,
        Tensor::Int64 {
            shape: vec![5, 3],
            data: vec![
                750_000,
                1,
                2,
                312 * 418,
                3,
                4,
                48 * 16,
                5,
                6,
                62 * 94,
                7,
                8,
                0,
                0,
                0
            ]
        }
    );
    assert!(blur_sums(1000, 750, &clamped, &sums[..3]).is_err());
}
