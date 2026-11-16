//! The detector passes of one photo (`frame_engine/faces/scrfd.py`: `fit`, `whole_pass`,
//! `tile_passes`, `plan_passes`): the whole frame in the detector's input, the optional coarse
//! square, and for a photo larger than the tile side, tiles of the frame scaled to that side.
//!
//! Python's arithmetic is kept to the bit: `int()` truncates, `round()` rounds half to even, and
//! the tile starts divide two integers as a float.

/// The frame resampled to `scaled` (w, h) and the `window` (left, top, w, h) of it that sits at
/// the detector input's top-left; `inner` marks the window edges (left, top, right, bottom)
/// inside the scaled frame, where a tile cuts faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetPass {
    /// The frame's size after the pass's resample.
    pub scaled: [i64; 2],
    /// The part of the resampled frame the input holds: left, top, width, height.
    pub window: [i64; 4],
    /// Whether the window's left, top, right and bottom edges lie inside the resampled frame.
    pub inner: [bool; 4],
}

/// The reference's letterbox size: the longer side to `side`, the other truncated; `None` for a
/// degenerate size.
pub fn fit(w: i64, h: i64, side: i64) -> Option<[i64; 2]> {
    if w <= 0 || h <= 0 {
        return None;
    }
    let ratio = h as f64 / w as f64;
    let side_f = side as f64;
    Some(if ratio > 1.0 {
        [((side_f / ratio) as i64).max(1), side]
    } else {
        [side, ((side_f * ratio) as i64).max(1)]
    })
}

/// The whole frame fitted into a `side` square at the input's top-left.
pub fn whole_pass(w: i64, h: i64, side: i64) -> Option<DetPass> {
    let [sw, sh] = fit(w, h, side)?;
    Some(DetPass {
        scaled: [sw, sh],
        window: [0, 0, sw, sh],
        inner: [false; 4],
    })
}

/// Window starts along one axis: evenly spread, neighbours sharing at least `overlap` pixels.
fn starts(total: i64, net: i64, overlap: i64) -> Vec<i64> {
    if total <= net {
        return vec![0];
    }
    let n = (total - overlap + (net - overlap) - 1) / (net - overlap);
    (0..n)
        .map(|k| ((k * (total - net)) as f64 / (n - 1) as f64).round_ties_even() as i64)
        .collect()
}

/// Windows of `net` covering the frame at `scale`; `None` when `overlap` is outside `[0, net)`.
pub fn tile_passes(w: i64, h: i64, net: i64, scale: f64, overlap: i64) -> Option<Vec<DetPass>> {
    if !(0..net).contains(&overlap) {
        return None;
    }
    let sw = ((w as f64 * scale).round_ties_even() as i64).max(1);
    let sh = ((h as f64 * scale).round_ties_even() as i64).max(1);
    let mut out = Vec::new();
    for top in starts(sh, net, overlap) {
        for left in starts(sw, net, overlap) {
            let (ww, wh) = (net.min(sw - left), net.min(sh - top));
            out.push(DetPass {
                scaled: [sw, sh],
                window: [left, top, ww, wh],
                inner: [left > 0, top > 0, left + ww < sw, top + wh < sh],
            });
        }
    }
    Some(out)
}

/// The passes of one photo: the whole frame in `net`; with `coarse`, the whole frame in a
/// `min(coarse, net)` square as well; with `tile_side`, when the longer side exceeds it, tiles of
/// the frame scaled to that longer side. `None` for a degenerate size or overlap (the plan
/// parser refuses such an overlap first).
pub fn plan_passes(
    w: i64,
    h: i64,
    net: i64,
    coarse: i64,
    tile_side: i64,
    overlap: i64,
) -> Option<Vec<DetPass>> {
    let mut passes = vec![whole_pass(w, h, net)?];
    if coarse > 0 {
        passes.push(whole_pass(w, h, coarse.min(net))?);
    }
    let long = w.max(h);
    if tile_side > 0 && long > tile_side {
        passes.extend(tile_passes(
            w,
            h,
            net,
            tile_side as f64 / long as f64,
            overlap,
        )?);
    }
    Some(passes)
}
