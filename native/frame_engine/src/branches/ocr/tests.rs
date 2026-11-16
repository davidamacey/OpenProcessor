use super::*;

fn spec() -> OcrSpec {
    OcrSpec {
        detector: "det".into(),
        recogniser: "rec".into(),
        batch: 2,
        max_lines: 2,
        det_side: 64,
        det_limit: "max".into(),
        det_max_side: 1920,
        det_antialias: true,
        thresh: 0.2,
        box_thresh: 0.45,
        unclip_ratio: 1.4,
        taps: 1,
    }
}

fn tensor(name: &str, datatype: &str, shape: Vec<i64>, words: Vec<[u8; 4]>) -> OutputTensor {
    OutputTensor {
        name: name.into(),
        datatype: datatype.into(),
        shape,
        data: Bytes::from(words.concat()),
    }
}

/// A 64 x 64 map with three bright bars: two on one line, one below (values 0.9, 0.8, 0.6).
fn map() -> Vec<OutputTensor> {
    let mut pred = vec![0f32; 64 * 64];
    for (x0, x1, y0, y1, v) in [
        (30, 60, 10, 18, 0.9f32),
        (2, 26, 11, 19, 0.8),
        (4, 50, 40, 46, 0.6),
    ] {
        for y in y0..y1 {
            for x in x0..x1 {
                pred[y * 64 + x] = v;
            }
        }
    }
    vec![tensor(
        DET_OUTPUT,
        "FP32",
        vec![1, 1, 64, 64],
        pred.iter().map(|v| v.to_le_bytes()).collect(),
    )]
}

#[test]
fn the_detector_input_is_bgr_at_paddles_size() {
    let (input, r) = det_canvas(&spec(), 640, 480).expect("canvas");
    assert_eq!(
        input.shape,
        [1, 3, 64, 64],
        "max 64: 64 x 48, rounded to 64 x 64"
    );
    assert_eq!((r.size, r.plane, r.dst_offset), ([64, 64], -4096, 8192));
    assert_eq!(r.extent(), Some(input.bytes()));
    assert_eq!(r.filter, Filter::Antialias);
    let plain = OcrSpec {
        det_antialias: false,
        det_limit: "x".into(),
        ..spec()
    };
    assert!(det_canvas(&plain, 640, 480).is_err(), "an unknown limit");
}

#[test]
fn boxes_past_max_lines_drop_the_lowest_scores_and_the_rest_read_in_order() {
    let s = settle(&map(), &spec(), 64, 64).expect("settled");
    assert_eq!(s.overflow, 1);
    assert_eq!(s.boxes.len(), 2);
    assert!(
        s.boxes[0].quad[0][0] < s.boxes[1].quad[0][0],
        "left bar first: {s:?}"
    );
    assert!(s.boxes.iter().all(|b| b.score >= 0.8));
    let none = settle(
        &[tensor(
            DET_OUTPUT,
            "FP32",
            vec![1, 1, 2, 2],
            vec![[0; 4]; 4],
        )],
        &spec(),
        64,
        64,
    )
    .expect("settled");
    assert_eq!(none, Settled::default());
    assert!(
        settle(
            &[tensor(DET_OUTPUT, "FP32", vec![4], vec![[0; 4]; 4])],
            &spec(),
            64,
            64
        )
        .is_err()
    );
}

#[test]
fn lines_are_warped_in_batches_and_decoded_back_into_reading_order() {
    let s = Settled {
        boxes: vec![
            TextBox {
                quad: [[0, 0], [200, 0], [200, 20], [0, 20]],
                score: 0.9,
            },
            TextBox {
                quad: [[0, 30], [40, 30], [40, 50], [0, 50]],
                score: 0.8,
            },
            TextBox {
                quad: [[0, 60], [60, 60], [60, 80], [0, 80]],
                score: 0.7,
            },
        ],
        overflow: 0,
    };
    let batches = line_batches(&s, &spec()).expect("batches");
    assert_eq!(
        batches
            .iter()
            .map(|b| b.indices.clone())
            .collect::<Vec<_>>(),
        [vec![1, 2], vec![0]]
    );
    assert_eq!(batches[0].input.shape, [2, 3, 48, 320]);
    assert_eq!(batches[0].warp.widths, [96, 144]);
    assert_eq!(batches[1].input.shape, [1, 3, 48, 480]);
    assert_eq!(
        (batches[1].warp.maps.len(), batches[1].warp.taps.len()),
        (9, 2)
    );

    let decoder = CtcDecoder::from_text("a\nb\n");
    let answer = |idx: Vec<i32>, steps: i64| {
        let n = idx.len() as i64 / steps;
        vec![
            tensor(
                REC_INDICES,
                "INT32",
                vec![n, steps],
                idx.iter().map(|v| v.to_le_bytes()).collect(),
            ),
            tensor(
                REC_SCORES,
                "FP32",
                vec![n, steps],
                vec![0.5f32.to_le_bytes(); idx.len()],
            ),
        ]
    };
    let calls = vec![
        (vec![1, 2], answer(vec![1, 0, 2, 2], 2)),
        (vec![0], answer(vec![0, 0], 2)),
    ];
    let out = outputs(&s, &calls, &decoder).expect("outputs");
    let names: Vec<&str> = out.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        names,
        [
            "ocr_quads",
            "ocr_texts",
            "ocr_scores",
            "ocr_char_min",
            "ocr_det_scores",
            "ocr_overflow"
        ]
    );
    assert_eq!(
        out[1].1,
        Tensor::Bytes(vec![Bytes::new(), Bytes::from("a"), Bytes::from("b")])
    );
    assert_eq!(
        out[2].1,
        Tensor::Fp32 {
            shape: vec![3],
            data: vec![0.0, 0.5, 0.5]
        }
    );
    let short = [(vec![0, 1], answer(vec![1, 1, 1], 3))];
    assert!(
        outputs(&s, &short, &decoder).is_err(),
        "one row for two lines"
    );
}
