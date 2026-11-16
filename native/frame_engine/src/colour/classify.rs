//! What an embedded profile is and, for one that moves colours, its conversion table: a port of
//! `src/utils/icc_srgb.py` (`tag_verdict`, `classify`) over Little CMS 2.19, the version Pillow
//! 13.0.0 bundles, through the `lcms2` crate built from that source.
//!
//! The transform is Pillow's `ImageCms.buildTransform(profile, sRGB, 'RGB', 'RGB')`: the built-in
//! sRGB profile (`cmsCreate_sRGBProfile`), perceptual intent, flags 0 (no black point
//! compensation, Little CMS's default 8-bit precalculated path, its one-pixel cache)
//! (<https://pillow.readthedocs.io/en/stable/reference/ImageCms.html>,
//! <https://docs.rs/lcms2/6.2.0/lcms2/struct.Transform.html>). The table is the transform's
//! answer at every 8-bit input, `r * 65536 + g * 256 + b`, three bytes each: what the frame's
//! gather kernel (`kernels/colour.cu`) reads.

use lcms2::{
    CIEXYZ, ColorSpaceSignature, Intent, PixelFormat, Profile, Tag, TagSignature, Transform,
};

use super::curves::tone_curves;

/// Levels a profile may move any 8-bit input and still count as sRGB (`SRGB_TOLERANCE`).
const SRGB_TOLERANCE: i16 = 1;
/// Colorant distances (`COLORANT_NEAR`, `COLORANT_FAR`) and curve moves (`CURVE_LEVELS_SRGB`,
/// `CURVE_LEVELS_CONVERT`) that bound the undecided band.
const COLORANT_NEAR: f64 = 0.002;
const COLORANT_FAR: f64 = 0.02;
const CURVE_LEVELS_SRGB: f64 = 0.5;
const CURVE_LEVELS_CONVERT: f64 = 1.5;
/// Entries of a table: every 8-bit RGB input.
pub const TABLE_ENTRIES: usize = 1 << 24;

/// What the decode routes do with a photo tagged with the profile (`ProfileKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Maps to sRGB within a level: left as decoded.
    Srgb,
    /// An RGB profile that moves colours: converted.
    Convert,
    /// Not RGB, or not a profile Little CMS opens: ignored.
    Other,
}

impl Kind {
    /// The word of `ProfileKind`.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Srgb => "srgb",
            Kind::Convert => "convert",
            Kind::Other => "other",
        }
    }
}

/// The nine matrix entries of a matrix/TRC profile (`_colorants`: Pillow's `red_colorant`,
/// `green_colorant`, `blue_colorant`, read only when `cmsIsMatrixShaper`), or `None`.
fn colorants(p: &Profile) -> Option<[[f64; 3]; 3]> {
    if !p.is_matrix_shaper() {
        return None;
    }
    let read = |sig| match p.read_tag(sig) {
        Tag::CIEXYZ(&CIEXYZ { X, Y, Z }) => Some([X, Y, Z]),
        _ => None,
    };
    Some([
        read(TagSignature::RedColorantTag)?,
        read(TagSignature::GreenColorantTag)?,
        read(TagSignature::BlueColorantTag)?,
    ])
}

/// `_srgb_encode`: IEC 61966-2.1 linear light to the encoded value, `NaN` kept.
fn srgb_encode(linear: f64) -> f64 {
    let v = if linear.is_nan() {
        linear
    } else {
        linear.clamp(0.0, 1.0)
    };
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// `_curve_move`: with the matrix at sRGB's, the largest level the curves move any of the 256
/// inputs from the sRGB curve; `None` when they cannot be read or the move is not finite.
fn curve_move(icc: &[u8]) -> Option<f64> {
    let curves = tone_curves(icc)?;
    let mut moved = f64::NEG_INFINITY;
    for c in &curves {
        for k in 0..256 {
            let x = f64::from(k) / 255.0;
            let m = (255.0 * srgb_encode(c.eval(x)) - 255.0 * x).abs();
            // numpy's max propagates NaN; f64::max would drop it
            moved = if m.is_nan() || moved.is_nan() {
                f64::NAN
            } else {
                moved.max(m)
            };
        }
    }
    moved.is_finite().then_some(moved)
}

/// The largest difference of any matrix entry from the built-in sRGB profile's, for a
/// matrix/TRC RGB profile.
pub fn colorant_distance(p: &Profile) -> Option<f64> {
    let mine = colorants(p)?;
    let srgb = colorants(&Profile::new_srgb())?;
    Some(
        mine.iter()
            .flatten()
            .zip(srgb.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max),
    )
}

fn open_rgb(icc: &[u8]) -> Option<Profile> {
    Profile::new_icc(icc)
        .ok()
        .filter(|p| p.color_space() == ColorSpaceSignature::RgbData)
}

/// `tag_verdict`: the kind the profile's own tags settle, or `None` when only the transform can.
pub fn tag_verdict(icc: &[u8]) -> Option<Kind> {
    let Some(p) = open_rgb(icc) else {
        return Some(Kind::Other);
    };
    let distance = colorant_distance(&p)?;
    if distance > COLORANT_FAR {
        return Some(Kind::Convert);
    }
    if distance > COLORANT_NEAR {
        return None;
    }
    let moved = curve_move(icc)?;
    if moved <= CURVE_LEVELS_SRGB {
        Some(Kind::Srgb)
    } else if moved >= CURVE_LEVELS_CONVERT {
        Some(Kind::Convert)
    } else {
        None
    }
}

/// The transform's answer at every 8-bit input, or `None` for a profile that is not RGB or
/// whose transform Little CMS refuses (`_transform`).
fn table(icc: &[u8]) -> Option<Vec<u8>> {
    let profile = open_rgb(icc)?;
    let srgb = Profile::new_srgb();
    let t: Transform<[u8; 3], [u8; 3]> = Transform::new(
        &profile,
        PixelFormat::RGB_8,
        &srgb,
        PixelFormat::RGB_8,
        Intent::Perceptual,
    )
    .ok()?;
    let mut out = vec![[0u8; 3]; TABLE_ENTRIES];
    let mut row = vec![[0u8; 3]; 1 << 16];
    for (r, dst) in out.chunks_mut(1 << 16).enumerate() {
        for (i, px) in row.iter_mut().enumerate() {
            *px = [r as u8, (i >> 8) as u8, i as u8];
        }
        t.transform_pixels(&row, dst);
    }
    Some(out.into_iter().flatten().collect())
}

/// The largest level the table moves any input.
fn table_move(table: &[u8]) -> i16 {
    table
        .as_chunks::<3>()
        .0
        .iter()
        .enumerate()
        .flat_map(|(i, px)| {
            let input = [(i >> 16) as u8, (i >> 8) as u8, i as u8];
            px.iter()
                .zip(input)
                .map(|(o, v)| (i16::from(*o) - i16::from(v)).abs())
        })
        .max()
        .unwrap_or(0)
}

/// `classify`: the profile's kind and, for `convert`, the table (`3 x 2^24` bytes).
pub fn classify(icc: &[u8]) -> (Kind, Option<Vec<u8>>) {
    let verdict = tag_verdict(icc);
    if let Some(kind @ (Kind::Srgb | Kind::Other)) = verdict {
        return (kind, None);
    }
    let Some(table) = table(icc) else {
        return (Kind::Other, None);
    };
    if verdict == Some(Kind::Convert) || table_move(&table) > SRGB_TOLERANCE {
        (Kind::Convert, Some(table))
    } else {
        (Kind::Srgb, None)
    }
}
