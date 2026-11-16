//! `face_embed` (`FaceEmbed.round2` and `finish`): which faces get a vector, the recogniser
//! batches, and the vectors L2-normalised in float32 as numpy does it (`np.linalg.norm` over a
//! row is the square root of numpy's pairwise sum of the squares, floored at `1e-10`).

use gpu_frame::OutputTensor;

use super::align::{self, ARCFACE_SIZE, FaceWarp};
use super::scrfd::Faces;
use super::warp::FaceWarps;
use super::{INPUT, OUTPUT};
use crate::branches::InputSpec;
use crate::branches::embed::{flagged, stacked};
use crate::geometry::pairwise_sum;
use crate::pipeline::Tensor;

/// One recogniser call: its input and the warp that writes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    /// The recogniser's input: `[faces, 3, 112, 112]` uint8.
    pub input: InputSpec,
    /// The faces' maps and taps.
    pub warps: FaceWarps,
}

/// What `face_embed` sends: which faces are embedded, and the calls of their crops.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedPlan {
    /// One flag per face: whether it gets a vector.
    pub embedded: Vec<bool>,
    /// The recogniser calls, in face order.
    pub chunks: Vec<Chunk>,
}

/// `FaceEmbed.round2`'s cut: each face's warp (`alignment.warp_params`), the usable ones in calls
/// of `batch`. An error where the reference's SVD fails (a non-finite landmark).
///
/// # Errors
///
/// A face's landmarks are not finite (see `align::estimate`).
pub fn plan(faces: &Faces, batch: u32) -> Result<EmbedPlan, String> {
    let warps: Vec<FaceWarp> = faces
        .dets
        .landmarks
        .iter()
        .map(align::warp_params)
        .collect::<Result<_, _>>()?;
    let usable: Vec<&FaceWarp> = warps.iter().filter(|w| w.ok).collect();
    let side = ARCFACE_SIZE as i64;
    let chunks = usable
        .chunks(batch.max(1) as usize)
        .map(|group| Chunk {
            input: InputSpec {
                name: INPUT,
                shape: [group.len() as i64, 3, side, side],
            },
            warps: FaceWarps {
                inv: group.iter().map(|w| w.inv).collect(),
                taps: group.iter().map(|w| w.taps).collect(),
                size: ARCFACE_SIZE as u32,
            },
        })
        .collect();
    Ok(EmbedPlan {
        embedded: warps.iter().map(|w| w.ok).collect(),
        chunks,
    })
}

/// One row divided by its float32 norm (`v / max(norm(v), 1e-10)`).
pub fn normalise(row: &mut [f32]) {
    let squares: Vec<f32> = row.iter().map(|v| v * v).collect();
    let norm = pairwise_sum(&squares).sqrt();
    let floor = 1e-10f32;
    let norm = if norm.is_nan() { norm } else { norm.max(floor) };
    for v in row.iter_mut() {
        *v /= norm;
    }
}

/// `face_embeddings` (`[embedded, dim]`, `(0, 0)` when no call was made) and `face_embedded`,
/// one flag per face.
///
/// # Errors
///
/// An answer has no `[rows, dim]` float32 `output`, or two answers differ in width.
pub fn outputs(
    embedded: &[bool],
    answers: &[Vec<OutputTensor>],
) -> Result<[(&'static str, Tensor); 2], String> {
    let (rows, dim, mut data) = stacked(answers, OUTPUT)?;
    if dim > 0 {
        data.chunks_mut(dim).for_each(normalise);
    }
    Ok(flagged(
        ["face_embeddings", "face_embedded"],
        embedded,
        (rows, dim, data),
    ))
}
