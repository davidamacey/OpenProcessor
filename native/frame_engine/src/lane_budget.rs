//! How many decode lanes fit on the card (plan 17 section 2.3 and N2 "Profiles").
//!
//! nvJPEG's device buffers of each lane grow to the largest photo the lane decoded and sit outside
//! `OP_FRAME_ENGINE_POOL_MIB`, so the default lane count is the physical cores *or what fits*,
//! whichever is fewer. [`lanes_that_fit`] is `scripts/codegen/frame_engine_lane_budget.py`
//! `lanes_that_fit`; `tests/lane_budget.rs` replays the cases that script writes.

/// Measured on an RTX A6000, set P 200 photos at 128 in flight, full plan, no failed photo: the
/// peak of `op_frame_engine_nvjpeg_peak_bytes` over 24 lanes while the largest photo was 26
/// megapixels. Photos of other sizes (COCO, the 20 MP set) are owed; replace these three numbers
/// with the measurement and nothing else changes.
pub const MEASURED_NVJPEG_PEAK_BYTES: u64 = 5_305_587_712;
/// The lanes of the measurement above.
pub const MEASURED_LANES: u64 = 24;
/// The largest photo of the measurement above, in megapixels.
pub const MEASURED_LARGEST_MEGAPIXELS: u64 = 26;

/// Bytes one lane holds per megapixel of the largest photo it decoded, rounded up.
pub const NVJPEG_BYTES_PER_MEGAPIXEL_PER_LANE: u64 =
    MEASURED_NVJPEG_PEAK_BYTES.div_ceil(MEASURED_LANES * MEASURED_LARGEST_MEGAPIXELS);

/// The photo size lanes are sized for: the largest photo measured, lowered by a smaller
/// `OP_FRAME_ENGINE_MAX_MEGAPIXELS`.
pub const SIZED_FOR_MEGAPIXELS: f64 = 26.0;

/// Left free for what is not counted (driver work areas, the model server's own growth), percent
/// of the memory left after the pool.
pub const HEADROOM_PERCENT: u64 = 10;

/// The fewest lanes the engine runs with.
pub const MIN_LANES: u32 = 1;
/// The most lanes `OP_FRAME_ENGINE_LANES` allows.
pub const MAX_LANES: u32 = 256;

/// The lane count and whether even the minimum fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneFit {
    /// Lanes that fit, at least [`MIN_LANES`] and at most [`MAX_LANES`].
    pub lanes: u32,
    /// False when not even [`MIN_LANES`] fit: the engine starts with the minimum and will answer
    /// `gpu_memory` for what does not fit.
    pub fits: bool,
}

/// The photo size to size lanes for under `OP_FRAME_ENGINE_MAX_MEGAPIXELS`.
pub fn sized_megapixels(max_megapixels: f64) -> f64 {
    max_megapixels.min(SIZED_FOR_MEGAPIXELS)
}

/// Lanes whose grown buffers fit in what the card has free once the pool cap is set aside.
/// `free_bytes` is the card's free memory after the context and the region are taken.
pub fn lanes_that_fit(free_bytes: u64, pool_bytes: u64, largest_megapixels: f64) -> LaneFit {
    let usable = free_bytes.saturating_sub(pool_bytes);
    let usable = u128::from(usable) * u128::from(100 - HEADROOM_PERCENT) / 100;
    let per_lane =
        ((largest_megapixels * NVJPEG_BYTES_PER_MEGAPIXEL_PER_LANE as f64).ceil() as u128).max(1);
    let fit = usable / per_lane;
    let lanes = fit.clamp(u128::from(MIN_LANES), u128::from(MAX_LANES));
    LaneFit {
        lanes: u32::try_from(lanes).unwrap_or(MAX_LANES),
        fits: fit >= u128::from(MIN_LANES),
    }
}

/// The default lane count: the cores, lowered to what fits on the card.
pub fn default_lanes(cores: u32, fit: LaneFit) -> u32 {
    cores.min(fit.lanes)
}
