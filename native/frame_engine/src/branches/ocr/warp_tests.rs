//! The text warp's launch and arguments against `LineWarp.warp_into`.

use super::*;

fn two_lines() -> LineWarp {
    LineWarp {
        maps: (0..18).map(|v| v as f32 * 0.5).collect(),
        widths: vec![100, 37],
        taps: vec![1, 1, 2, 3],
        batch_width: 101,
        pad: 128,
    }
}

#[test]
fn the_grid_covers_every_pixel_of_a_line_per_row_of_blocks() {
    // (101 * 48 + 127) / 128 = 38 (4,848 pixels; 37 blocks hold 4,736)
    let l = launch(2, 101).expect("a launch");
    assert_eq!(
        (l.grid, l.block, l.shared_bytes),
        ((38, 2, 1), (128, 1, 1), 0)
    );
    assert_eq!(launch(1, 8).expect("a launch").grid, (3, 1, 1));
    assert!(launch(0, 8).is_err());
    assert!(launch(3, 0).is_err());
}

#[test]
fn arguments_are_in_warp_intos_order() {
    let w = two_lines();
    let a = args((0xA0, 640, 480), [0xB0, 0xC0, 0xD0], &w, 0xE0).expect("args");
    assert_eq!(
        a,
        [
            KernelArg::U64(0xA0),
            KernelArg::I32(480),
            KernelArg::I32(640),
            KernelArg::U64(0xB0),
            KernelArg::U64(0xC0),
            KernelArg::U64(0xD0),
            KernelArg::I32(2),
            KernelArg::I32(101),
            KernelArg::I32(48),
            KernelArg::U8(128),
            KernelArg::U64(0xE0),
        ]
    );
    assert_eq!(bytes(&w), 2 * 3 * 48 * 101);
}

#[test]
fn tables_are_little_endian_rows_as_the_kernel_reads_them() {
    let [maps, widths, taps] = blobs(&two_lines());
    assert_eq!(maps.len(), 18 * 4);
    assert_eq!(&maps[4..8], &0.5f32.to_le_bytes());
    assert_eq!(widths, [100i32.to_le_bytes(), 37i32.to_le_bytes()].concat());
    assert_eq!(taps[12..16], 3i32.to_le_bytes());
}

#[test]
fn a_line_wider_than_its_call_or_short_tables_are_refused() {
    assert!(check(&two_lines()).is_ok());
    let mut wide = two_lines();
    wide.widths[1] = 102;
    assert!(check(&wide).is_err());
    let mut short = two_lines();
    short.taps.pop();
    assert!(check(&short).is_err());
}
