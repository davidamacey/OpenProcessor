//! The branches of phase N0 (`frame_engine/branches/`): what each cuts from the frame, the model
//! call it makes and how its answer becomes the photo's tensors. These modules are the plan-level
//! arithmetic only (geometry, chunking, the settle, the output tensors); the photo flow
//! (`photo.rs`) schedules them and the device runs their kernels.
//!
//! Built: `detect`, `embed_frame` and `embed_crops` with `crops: inline` (N0), `faces` and
//! `face_embed` (N1b); every other branch the protocol knows answers `unsupported_branch`.

pub mod crops;
pub mod derivatives;
pub mod detect;
pub mod embed;
pub mod faces;
pub mod ocr;

use gpu_frame::OutputTensor;

/// The branches this engine builds, each a `branch_status` entry.
pub const BUILT: [&str; 5] = [
    "detect",
    "embed_frame",
    "embed_crops",
    "faces",
    "face_embed",
];

/// One model call's input tensor as the model server is told it: a uint8 CHW batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputSpec {
    /// The model's input tensor name.
    pub name: &'static str,
    /// The input shape as `[batch, channels, height, width]`.
    pub shape: [i64; 4],
}

impl InputSpec {
    /// Triton datatype of every input the engine sends.
    pub const DATATYPE: &'static str = "UINT8";

    /// Its bytes (uint8, so the element count).
    pub fn bytes(&self) -> usize {
        self.shape
            .iter()
            .map(|d| usize::try_from(*d).unwrap_or(0))
            .product()
    }
}

/// The named output of a call: its shape and its 4-byte elements, checked against both.
fn output_words<'a>(
    outputs: &'a [OutputTensor],
    name: &str,
    datatype: &str,
) -> Result<(&'a [i64], &'a [[u8; 4]]), String> {
    let t = outputs
        .iter()
        .find(|o| o.name == name)
        .ok_or_else(|| format!("the answer has no output {name}"))?;
    if t.datatype != datatype {
        return Err(format!("output {name} is {}, not {datatype}", t.datatype));
    }
    let count: usize = t
        .shape
        .iter()
        .map(|d| usize::try_from(*d).unwrap_or(0))
        .product();
    let (values, rest) = t.data.as_chunks::<4>();
    if !rest.is_empty() || values.len() != count {
        return Err(format!(
            "output {name}: {} bytes for shape {:?}",
            t.data.len(),
            t.shape
        ));
    }
    Ok((&t.shape, values))
}

/// The named output of a call as float32 values, with its shape.
///
/// # Errors
///
/// The output is missing, is not FP32, or its byte length does not match its shape.
pub fn fp32_output<'a>(
    outputs: &'a [OutputTensor],
    name: &str,
) -> Result<(&'a [i64], Vec<f32>), String> {
    let (shape, words) = output_words(outputs, name, "FP32")?;
    Ok((
        shape,
        words.iter().map(|b| f32::from_le_bytes(*b)).collect(),
    ))
}

/// The named output of a call as int32 values, with its shape (the OCR recogniser's `indices`).
///
/// # Errors
///
/// The output is missing, is not INT32, or its byte length does not match its shape.
pub fn int32_output<'a>(
    outputs: &'a [OutputTensor],
    name: &str,
) -> Result<(&'a [i64], Vec<i32>), String> {
    let (shape, words) = output_words(outputs, name, "INT32")?;
    Ok((
        shape,
        words.iter().map(|b| i32::from_le_bytes(*b)).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    fn tensor(datatype: &str, shape: Vec<i64>, data: Vec<u8>) -> OutputTensor {
        OutputTensor {
            name: "o".into(),
            datatype: datatype.into(),
            shape,
            data: Bytes::from(data),
        }
    }

    #[test]
    fn an_fp32_output_is_read_little_endian_and_checked_against_its_shape() {
        let data: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let outputs = [tensor("FP32", vec![1, 2], data.clone())];
        let (shape, values) = fp32_output(&outputs, "o").expect("read");
        assert_eq!((shape, values), (&[1i64, 2][..], vec![1.5, -2.0]));
        assert!(fp32_output(&outputs, "missing").is_err());
        assert!(fp32_output(&[tensor("FP16", vec![1, 2], data.clone())], "o").is_err());
        assert!(fp32_output(&[tensor("FP32", vec![1, 3], data)], "o").is_err());
        let ints = [tensor(
            "INT32",
            vec![2],
            [7i32, -1].iter().flat_map(|v| v.to_le_bytes()).collect(),
        )];
        assert_eq!(int32_output(&ints, "o").expect("read").1, vec![7, -1]);
        assert!(
            int32_output(&outputs, "o").is_err(),
            "an FP32 output is not INT32"
        );
        assert_eq!(
            InputSpec {
                name: "images",
                shape: [2, 3, 224, 224]
            }
            .bytes(),
            2 * 3 * 224 * 224
        );
    }
}
