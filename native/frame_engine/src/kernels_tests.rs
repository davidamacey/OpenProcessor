use super::*;

/// The detector letterbox of a 1000 x 750 photo in a 640 canvas (`letterbox_geometry`: 640 x 480
/// at top 80), as `Detect.round1` cuts it.
fn letterbox() -> Resample {
    Resample {
        roi: [0, 0, 1000, 750],
        size: [640, 480],
        window: [0, 0, 640, 480],
        dst_offset: 80 * 640,
        plane: 640 * 640,
        row_stride: 640,
        filter: Filter::Antialias,
    }
}

#[test]
fn the_letterbox_launches_as_resample_into_does() {
    let r = letterbox();
    // filter_weights(750, 480): every source row is read, so lo 0 and hi 750 (Python, numpy 2.5.4)
    let shape = resample_shape(&r, 0, 750).expect("shape");
    assert_eq!(shape.rows, 750);
    assert_eq!(shape.scratch_bytes, 750 * 640 * 3 * 4);
    assert_eq!(shape.h.grid, (5, 750, 1));
    assert_eq!(shape.v.grid, (5, 480, 1));
    assert_eq!((shape.h.block, shape.v.block), ((128, 1, 1), (128, 1, 1)));
    let wx = SetAt {
        addr: 0x1000,
        n: 640,
        kmax: 4,
        lo: 0,
    };
    let h = resample_h_args((0x9000, 1000, 750), &r, wx, 0, 750, 0x7000).expect("args");
    use KernelArg::{I32, I64, U64};
    assert_eq!(
        h,
        [
            U64(0x9000),
            I32(750),
            I32(1000),
            I32(0),
            I32(0),
            I32(750),
            U64(0x1000),
            U64(0x1000 + 4 * 640),
            U64(0x1000 + 8 * 640),
            I32(4),
            I32(640),
            U64(0x7000),
        ]
    );
    let wy = SetAt {
        addr: 0x2000,
        n: 480,
        kmax: 4,
        lo: 0,
    };
    let v = resample_v_args(0x7000, &r, wy, 0x5_0000 + 80 * 640).expect("args");
    assert_eq!(
        v,
        [
            U64(0x7000),
            I32(640),
            U64(0x2000),
            U64(0x2000 + 4 * 480),
            U64(0x2000 + 8 * 480),
            I32(4),
            I32(480),
            I32(0),
            U64(0x5_0000 + 80 * 640),
            I64(640 * 640),
            I32(640),
        ]
    );
}

#[test]
fn a_crop_reads_from_its_rectangle_and_the_scratch_starts_at_the_filters_first_row() {
    // a crop at (100, 40) whose vertical filter reads source rows 12..90 of its rectangle
    let r = Resample {
        roi: [100, 40, 300, 200],
        size: [224, 149],
        window: [0, 0, 224, 149],
        dst_offset: 3 * 224 * 224 + 37 * 224,
        plane: 224 * 224,
        row_stride: 224,
        filter: Filter::Antialias,
    };
    let shape = resample_shape(&r, 12, 90).expect("shape");
    assert_eq!(
        (shape.rows, shape.h.grid, shape.v.grid),
        (78, (2, 78, 1), (2, 149, 1))
    );
    let wx = SetAt {
        addr: 0,
        n: 224,
        kmax: 4,
        lo: 0,
    };
    let h = resample_h_args((0, 640, 480), &r, wx, 12, 78, 0).expect("args");
    assert_eq!(
        (h[3], h[4], h[5]),
        (KernelArg::I32(100), KernelArg::I32(52), KernelArg::I32(78))
    );
    let wy = SetAt {
        addr: 0,
        n: 149,
        kmax: 4,
        lo: 12,
    };
    assert_eq!(
        resample_v_args(0, &r, wy, 0).expect("args")[7],
        KernelArg::I32(12)
    );
}

#[test]
fn the_written_window_must_lie_inside_its_slice() {
    let r = letterbox();
    // the last byte: plane 2, row 479 of the window, column 639, after the 80-row top margin
    assert_eq!(r.extent(), Some(80 * 640 + 2 * 640 * 640 + 479 * 640 + 640));
    assert_eq!(
        r.extent(),
        Some(3 * 640 * 640 - 80 * 640),
        "the bottom margin is not written"
    );
    let empty = Resample {
        window: [0, 0, 0, 480],
        ..r
    };
    assert_eq!(empty.extent(), None);
}

#[test]
fn the_filter_keys_follow_each_axis() {
    let r = Resample {
        roi: [0, 0, 1000, 750],
        size: [448, 336],
        window: [56, 0, 336, 336],
        dst_offset: 0,
        plane: 336 * 336,
        row_stride: 336,
        filter: Filter::Antialias,
    };
    assert_eq!(
        r.x_key(),
        WeightKey {
            n_in: 1000,
            n_out: 448,
            left: 56,
            take: 336,
            filter: Filter::Antialias,
        }
    );
    assert_eq!(
        r.y_key(),
        WeightKey {
            n_in: 750,
            n_out: 336,
            left: 0,
            take: 336,
            filter: Filter::Antialias,
        }
    );
    let w = r.x_key().weights().expect("weights");
    assert_eq!(
        (w.n, w.kmax),
        (336, 5),
        "as filter_weights(1000, 448)[56:392]"
    );
}

fn key(n_out: i64) -> WeightKey {
    WeightKey {
        n_in: 1000,
        n_out,
        left: 0,
        take: n_out,
        filter: Filter::Antialias,
    }
}

fn blob_len(k: WeightKey) -> usize {
    k.weights().expect("weights").blob.len()
}

#[test]
fn the_cache_uploads_a_set_once_and_drops_the_least_recently_used_past_its_bytes() {
    let (ka, kb, kc) = (key(100), key(120), key(50));
    assert!(blob_len(kc) <= blob_len(kb), "c fits where b was");
    let mut uploads = 0;
    let mut cache = WeightCache::<Vec<u8>>::new(blob_len(ka) + blob_len(kb));
    let mut get = |cache: &mut WeightCache<Vec<u8>>, k| {
        cache
            .get(k, |blob: &[u8]| {
                uploads += 1;
                Ok::<_, GpuFault>(blob.to_vec())
            })
            .expect("set")
    };
    let a = get(&mut cache, ka);
    assert_eq!(a.bytes, a.buf.len());
    assert_eq!(a.bytes, 100 * (8 + 4 * a.kmax));
    let again = get(&mut cache, ka);
    assert!(Arc::ptr_eq(&a, &again), "a hit is the same set");
    let b = get(&mut cache, kb);
    assert_eq!(cache.held(), a.bytes + b.bytes);
    // touch a, so b is the oldest when c needs room
    get(&mut cache, ka);
    let c = get(&mut cache, kc);
    assert_eq!(cache.held(), a.bytes + c.bytes);
    assert_eq!(cache.len(), 2);
    get(&mut cache, ka);
    get(&mut cache, kc);
    drop(get);
    assert_eq!(uploads, 3, "a and c stayed, b left");
    assert_eq!(b.buf.len(), b.bytes, "a set in use outlives its eviction");
}

#[test]
fn a_set_larger_than_the_cache_is_returned_and_not_kept() {
    let mut cache = WeightCache::<Vec<u8>>::new(64);
    let set = cache
        .get(key(100), |b: &[u8]| Ok::<_, GpuFault>(b.to_vec()))
        .expect("set");
    assert!(set.bytes > 64);
    assert!(cache.is_empty());
    assert_eq!(cache.held(), 0);
}

#[test]
fn an_upload_that_fails_is_the_callers_error_and_caches_nothing() {
    let mut cache = WeightCache::<Vec<u8>>::new(1 << 20);
    let failed = cache.get(key(10), |_| {
        Err::<Vec<u8>, _>(GpuFault::OutOfMemory("cap".into()))
    });
    assert_eq!(failed.err(), Some(GpuFault::OutOfMemory("cap".into())));
    assert!(cache.is_empty());
    let empty = WeightKey {
        n_in: 10,
        n_out: 5,
        left: 5,
        take: 3,
        filter: Filter::Antialias,
    };
    assert!(matches!(
        cache.get(empty, |b: &[u8]| Ok::<_, GpuFault>(b.to_vec())),
        Err(GpuFault::Failed(_))
    ));
}

/// `engine._orient` step by step, on host arrays: transpose, then flip columns, then flip rows.
fn python_orient(src: &[u8], w: usize, h: usize, o: u32) -> (Vec<u8>, usize, usize) {
    let px = |buf: &[u8], width: usize, x: usize, y: usize| {
        let i = (y * width + x) * 3;
        [buf[i], buf[i + 1], buf[i + 2]]
    };
    let (mut t, mut tw, mut th) = (src.to_vec(), w, h);
    if (5..=8).contains(&o) {
        let mut out = Vec::with_capacity(t.len());
        for y in 0..tw {
            for x in 0..th {
                out.extend(px(&t, tw, y, x));
            }
        }
        (t, tw, th) = (out, th, tw);
    }
    if matches!(o, 2 | 3 | 6 | 7) {
        let mut out = Vec::with_capacity(t.len());
        for y in 0..th {
            for x in (0..tw).rev() {
                out.extend(px(&t, tw, x, y));
            }
        }
        t = out;
    }
    if matches!(o, 3 | 4 | 7 | 8) {
        let mut out = Vec::with_capacity(t.len());
        for y in (0..th).rev() {
            for x in 0..tw {
                out.extend(px(&t, tw, x, y));
            }
        }
        t = out;
    }
    (t, tw, th)
}

#[test]
fn every_exif_orientation_moves_the_pixels_as_the_python_engine_does() {
    let (w, h) = (5u32, 3u32);
    let src: Vec<u8> = (0..(w * h * 3)).map(|i| i as u8).collect();
    for o in 0..=9 {
        let (got, gw, gh) = orient_host(&src, w, h, o);
        let (want, ww, wh) = python_orient(&src, w as usize, h as usize, o);
        assert_eq!((gw as usize, gh as usize), (ww, wh), "orientation {o}");
        assert_eq!(got, want, "orientation {o}");
        assert_eq!(reorients(o), got != src, "orientation {o}");
    }
    assert_eq!(oriented_dims(5, 3, 6), (3, 5));
    assert_eq!(
        orient_launch(3000, 4000).expect("launch").grid,
        (12, 4000, 1)
    );
}

#[test]
fn rectangles_go_to_the_kernel_as_int32_and_the_sums_come_back_as_int64_pairs() {
    let blob = rect_blob(&[[0, 0, 640, 480], [1, 2, 3, 4]]).expect("blob");
    assert_eq!(blob.len(), 32);
    assert_eq!(&blob[8..12], &640i32.to_le_bytes());
    assert!(rect_blob(&[[0, 0, 1 << 40, 1]]).is_err());
    let mut sums = Vec::new();
    for v in [7i64, 49, -3, 1 << 40] {
        sums.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(lap_values(&sums), [[7, 49], [-3, 1 << 40]]);
    let launch = lap_launch(3).expect("launch");
    assert_eq!((launch.grid, launch.block), ((64, 3, 1), (256, 1, 1)));
    assert!(lap_launch(0).is_err());
}

#[test]
fn the_kernel_sources_are_the_files_both_engines_read() {
    assert!(PREP_SOURCE.contains("void resample_h(") && PREP_SOURCE.contains("void lap_sums("));
    assert!(ORIENT_SOURCE.contains("extern \"C\" __global__ void orient("));
    // the kernel and its host twin flip and transpose under the same tags
    for tags in [
        "orientation == 3 || orientation == 4 || orientation == 7 || orientation == 8",
        "orientation == 2 || orientation == 3 || orientation == 6 || orientation == 7",
        "orientation >= 5 && orientation <= 8",
    ] {
        assert!(ORIENT_SOURCE.contains(tags), "{tags}");
    }
}
