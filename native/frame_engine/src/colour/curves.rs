//! The tone curves of an ICC matrix/TRC profile, read from its tag table: a port of
//! `src/utils/icc_curves.py`, which reads them as the ICC specification lays them out
//! (ICC.1:2022, <https://www.color.org/specification/ICC.1-2022-05.pdf>: the tag table of section
//! 7.3, `curveType` of section 10.6, `parametricCurveType` of section 10.18, function types 0 to
//! 4). Everything is read from untrusted bytes: a tag, size or type that is not exactly as the
//! specification allows gives `None`, and the caller falls back to the transform.
//!
//! The evaluation follows numpy's arithmetic in `float64`: `x ** g` as `pow`, the parametric
//! segments as `np.where` picks them, and a sampled curve as `np.interp` over
//! `np.linspace(0, 1, n)` (index `k` at `k * (1 / (n - 1))`, the last at exactly 1).

const TAG_TABLE: usize = 128;
const CURVE_TAGS: [&[u8; 4]; 3] = [b"rTRC", b"gTRC", b"bTRC"];

/// One channel's curve, from the encoded value in `[0, 1]` to linear light.
#[derive(Debug, Clone, PartialEq)]
pub enum Curve {
    /// `curv` with no entry: the identity.
    Identity,
    /// `curv` with one entry, or `para` function type 0: `x ** g`.
    Gamma(f64),
    /// `para` function types 1 to 4 with their parameters (`g, a, b, c, d, e, f` as given).
    Parametric(u16, Vec<f64>),
    /// `curv` with a table: `n` samples over `[0, 1]`, linearly interpolated.
    Sampled(Vec<f64>),
}

fn be_u32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn be_u16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

/// An `s15Fixed16Number`.
fn s15(b: &[u8], at: usize) -> Option<f64> {
    let raw = i32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?);
    Some(f64::from(raw) / 65536.0)
}

/// `_tag_data`: the bytes of one tag, or `None` when it is absent or its entry leaves the file.
fn tag_data<'a>(icc: &'a [u8], signature: &[u8; 4]) -> Option<&'a [u8]> {
    let count = usize::try_from(be_u32(icc, TAG_TABLE)?).ok()?;
    if count > (icc.len() - TAG_TABLE - 4) / 12 {
        return None;
    }
    let entry = (0..count)
        .map(|i| TAG_TABLE + 4 + 12 * i)
        .find(|at| &icc[*at..at + 4] == signature)?;
    let offset = usize::try_from(be_u32(icc, entry + 4)?).ok()?;
    let size = usize::try_from(be_u32(icc, entry + 8)?).ok()?;
    if offset + size > icc.len() {
        return None;
    }
    icc.get(offset..offset + size)
}

/// `_parametric`: parameter counts by function type (ICC.1:2022 table 65).
fn parametric(data: &[u8]) -> Option<Curve> {
    let kind = be_u16(data, 8)?;
    let n = match kind {
        0 => 1,
        1 => 3,
        2 => 4,
        3 => 5,
        4 => 7,
        _ => return None,
    };
    if data.len() < 12 + 4 * n {
        return None;
    }
    let p: Vec<f64> = (0..n)
        .map(|k| s15(data, 12 + 4 * k))
        .collect::<Option<_>>()?;
    Some(if kind == 0 {
        Curve::Gamma(p[0])
    } else {
        Curve::Parametric(kind, p)
    })
}

/// `_curve`: a `curv` or `para` tag.
fn curve(data: &[u8]) -> Option<Curve> {
    if data.len() < 12 {
        return None;
    }
    if &data[..4] == b"para" {
        return parametric(data);
    }
    if &data[..4] != b"curv" {
        return None;
    }
    let n = usize::try_from(be_u32(data, 8)?).ok()?;
    if data.len() < 12 + 2 * n {
        return None;
    }
    Some(match n {
        0 => Curve::Identity,
        1 => Curve::Gamma(f64::from(be_u16(data, 12)?) / 256.0),
        _ => Curve::Sampled(
            (0..n)
                .map(|k| be_u16(data, 12 + 2 * k).map(|v| f64::from(v) / 65535.0))
                .collect::<Option<_>>()?,
        ),
    })
}

/// `tone_curves`: the red, green and blue curves, or `None` when the profile does not hold the
/// three in a form this module reads.
pub fn tone_curves(icc: &[u8]) -> Option<[Curve; 3]> {
    if icc.len() < TAG_TABLE + 4 {
        return None;
    }
    let [r, g, b] = CURVE_TAGS.map(|sig| tag_data(icc, sig).and_then(curve));
    Some([r?, g?, b?])
}

impl Curve {
    /// The curve at `x` (`NaN` where Python's answers `NaN`: a parametric curve with `a = 0`).
    pub fn eval(&self, x: f64) -> f64 {
        match self {
            Curve::Identity => x,
            Curve::Gamma(g) => x.powf(*g),
            Curve::Parametric(kind, p) => parametric_at(*kind, p, x),
            Curve::Sampled(table) => interp(x, table),
        }
    }
}

/// The segments of `_parametric`'s inner `curve`, for one `x`.
fn parametric_at(kind: u16, p: &[f64], x: f64) -> f64 {
    let (g, a, b) = (p[0], p[1], p[2]);
    if a == 0.0 {
        return f64::NAN;
    }
    let base = (a * x + b).max(0.0).powf(g);
    match kind {
        1 => {
            if x >= -b / a {
                base
            } else {
                0.0
            }
        }
        2 => {
            if x >= -b / a {
                base + p[3]
            } else {
                p[3]
            }
        }
        3 => {
            if x >= p[4] {
                base
            } else {
                p[3] * x
            }
        }
        _ => {
            if x >= p[4] {
                base + p[5]
            } else {
                p[3] * x + p[6]
            }
        }
    }
}

/// `np.interp(x, np.linspace(0, 1, n), table)` for `x` in `[0, 1]` (`arr_interp`: the sample at
/// or left of `x`, then `slope * (x - xp[j]) + fp[j]`, exactly `fp[j]` on a sample).
fn interp(x: f64, table: &[f64]) -> f64 {
    let n = table.len();
    let step = 1.0 / (n - 1) as f64;
    let grid = |k: usize| if k == n - 1 { 1.0 } else { k as f64 * step };
    if x >= 1.0 {
        return table[n - 1];
    }
    if x < 0.0 {
        return table[0];
    }
    let j = (0..n - 1).rev().find(|&k| grid(k) <= x).unwrap_or(0);
    if grid(j) == x {
        return table[j];
    }
    let slope = (table[j + 1] - table[j]) / (grid(j + 1) - grid(j));
    slope * (x - grid(j)) + table[j]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = kind.to_vec();
        out.extend([0; 4]);
        out.extend(body);
        out
    }

    #[test]
    fn curves_are_read_as_the_specification_lays_them_out() {
        assert_eq!(curve(&tag(b"curv", &[0, 0, 0, 0])), Some(Curve::Identity));
        assert_eq!(
            curve(&tag(b"curv", &[0, 0, 0, 1, 2, 0x33])),
            Some(Curve::Gamma(563.0 / 256.0))
        );
        let table = curve(&tag(b"curv", &[0, 0, 0, 2, 0, 0, 0xFF, 0xFF])).expect("sampled");
        assert_eq!(table.eval(0.25), 0.25);
        // function type 1 needs three parameters: two is too short
        assert_eq!(
            curve(&tag(b"para", &[0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0])),
            None
        );
        assert_eq!(curve(&tag(b"desc", &[0; 8])), None);
        assert_eq!(tone_curves(&[0; 100]), None);
    }

    #[test]
    fn the_srgb_parametric_curve_has_its_linear_toe() {
        let p = vec![2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045];
        let c = Curve::Parametric(3, p);
        assert_eq!(c.eval(0.04), 0.04 / 12.92);
        assert!((c.eval(1.0) - 1.0).abs() < 1e-12);
        assert!(Curve::Parametric(1, vec![2.2, 0.0, 0.0]).eval(0.5).is_nan());
    }
}
