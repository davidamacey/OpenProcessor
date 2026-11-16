//! The outputs of a `ModelInferResponse` as typed arrays: little-endian values in
//! `raw_output_contents`, one entry per entry of `outputs`
//! (<https://kserve.github.io/website/latest/modelserving/data_plane/binary_tensor_data_extension/>).

use std::collections::HashMap;

use bytes::Bytes;

use super::AnswerError;
use crate::codec::unpack_bytes;
use crate::proto::ModelInferResponse;

/// A value type of a numeric tensor.
pub(super) trait Element: Copy {
    /// The KServe datatype name.
    const DATATYPE: &'static str;
    /// Bytes per value.
    const WIDTH: usize;
    /// One value from exactly `WIDTH` bytes.
    fn read(bytes: &[u8]) -> Self;
}

macro_rules! number {
    ($t:ty, $name:literal) => {
        impl Element for $t {
            const DATATYPE: &'static str = $name;
            const WIDTH: usize = std::mem::size_of::<$t>();
            fn read(bytes: &[u8]) -> Self {
                let mut raw = [0u8; std::mem::size_of::<$t>()];
                raw.copy_from_slice(bytes);
                <$t>::from_le_bytes(raw)
            }
        }
    };
}

number!(f32, "FP32");
number!(f64, "FP64");
number!(i32, "INT32");
number!(i64, "INT64");

impl Element for bool {
    const DATATYPE: &'static str = "BOOL";
    const WIDTH: usize = 1;
    fn read(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
}

/// Rows of `dim` values: the vectors of one output, `rows` of them.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// The number of rows.
    pub rows: usize,
    /// The width of a row.
    pub dim: usize,
    /// Row after row.
    pub data: Vec<f32>,
}

impl Matrix {
    /// Row `index`, if there is one.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        let start = index.checked_mul(self.dim)?;
        self.data.get(start..start.checked_add(self.dim)?)
    }
}

struct Output<'a> {
    datatype: &'a str,
    shape: &'a [i64],
    raw: &'a Bytes,
}

/// The outputs of one response, by name.
pub(super) struct Outputs<'a> {
    by_name: HashMap<&'a str, Output<'a>>,
}

impl<'a> Outputs<'a> {
    /// Pairs each output's metadata with its raw contents.
    pub(super) fn new(response: &'a ModelInferResponse) -> Result<Self, AnswerError> {
        if response.raw_output_contents.len() != response.outputs.len() {
            return Err(AnswerError::RawContents(format!(
                "{} entries for {} outputs",
                response.raw_output_contents.len(),
                response.outputs.len()
            )));
        }
        let by_name = response
            .outputs
            .iter()
            .zip(&response.raw_output_contents)
            .map(|(meta, raw)| {
                let output = Output {
                    datatype: &meta.datatype,
                    shape: &meta.shape,
                    raw,
                };
                (meta.name.as_str(), output)
            })
            .collect();
        Ok(Self { by_name })
    }

    pub(super) fn has(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    fn find(&self, name: &'static str, datatype: &'static str) -> Result<&Output<'a>, AnswerError> {
        let output = self.by_name.get(name).ok_or(AnswerError::Missing(name))?;
        if output.datatype != datatype {
            return Err(AnswerError::Datatype {
                name,
                got: output.datatype.to_owned(),
                want: datatype,
            });
        }
        Ok(output)
    }

    /// The values and shape of a numeric output.
    fn values<T: Element>(&self, name: &'static str) -> Result<(Vec<T>, &'a [i64]), AnswerError> {
        let output = self.find(name, T::DATATYPE)?;
        let shape_error = |detail: String| AnswerError::Shape { name, detail };
        let mut count: usize = 1;
        for &dim in output.shape {
            let dim = usize::try_from(dim).map_err(|_| {
                shape_error(format!("shape {:?} has a negative size", output.shape))
            })?;
            count = count
                .checked_mul(dim)
                .ok_or_else(|| shape_error("shape overflows".into()))?;
        }
        if count.checked_mul(T::WIDTH) != Some(output.raw.len()) {
            return Err(shape_error(format!(
                "shape {:?} of {} is not {} bytes",
                output.shape,
                T::DATATYPE,
                output.raw.len()
            )));
        }
        let values = output.raw.chunks_exact(T::WIDTH).map(T::read).collect();
        Ok((values, output.shape))
    }

    /// A one-dimensional output of any length.
    pub(super) fn list<T: Element>(&self, name: &'static str) -> Result<Vec<T>, AnswerError> {
        let (values, shape) = self.values::<T>(name)?;
        if shape.len() != 1 {
            return Err(AnswerError::Shape {
                name,
                detail: format!("shape {shape:?} is not one-dimensional"),
            });
        }
        Ok(values)
    }

    /// A one-dimensional output of exactly `N` values.
    pub(super) fn fixed<T: Element, const N: usize>(
        &self,
        name: &'static str,
    ) -> Result<[T; N], AnswerError> {
        let values = self.list::<T>(name)?;
        let len = values.len();
        <[T; N]>::try_from(values).map_err(|_| AnswerError::Shape {
            name,
            detail: format!("{len} values, expected {N}"),
        })
    }

    /// A matrix whose rows are `N` values wide.
    pub(super) fn rows<T: Element, const N: usize>(
        &self,
        name: &'static str,
    ) -> Result<Vec<[T; N]>, AnswerError> {
        let (values, shape) = self.values::<T>(name)?;
        if shape.len() != 2 || usize::try_from(shape[1]).ok() != Some(N) {
            return Err(AnswerError::Shape {
                name,
                detail: format!("shape {shape:?} is not [n, {N}]"),
            });
        }
        Ok(values.as_chunks::<N>().0.to_vec())
    }

    /// A matrix of vectors; `[0, 0]` (nothing picked) is the empty matrix.
    pub(super) fn matrix(&self, name: &'static str) -> Result<Matrix, AnswerError> {
        let (data, shape) = self.values::<f32>(name)?;
        let [rows, dim] = shape else {
            return Err(AnswerError::Shape {
                name,
                detail: format!("shape {shape:?} is not [rows, dim]"),
            });
        };
        let (rows, dim) = (
            usize::try_from(*rows).unwrap_or(0),
            usize::try_from(*dim).unwrap_or(0),
        );
        Ok(Matrix { rows, dim, data })
    }

    /// The elements of a BYTES output, as slices of the response (no copy).
    pub(super) fn byte_items(&self, name: &'static str) -> Result<Vec<Bytes>, AnswerError> {
        let output = self.find(name, "BYTES")?;
        let items = unpack_bytes(output.raw).ok_or_else(|| AnswerError::Shape {
            name,
            detail: "a BYTES element runs past the end".into(),
        })?;
        let expected: i64 = output.shape.iter().product();
        if i64::try_from(items.len()).ok() != Some(expected) {
            return Err(AnswerError::Shape {
                name,
                detail: format!("{} elements for shape {:?}", items.len(), output.shape),
            });
        }
        Ok(items.into_iter().map(|i| output.raw.slice_ref(i)).collect())
    }

    /// The elements of a BYTES output as text.
    pub(super) fn texts(&self, name: &'static str) -> Result<Vec<String>, AnswerError> {
        self.byte_items(name)?
            .into_iter()
            .map(|item| String::from_utf8(item.to_vec()).map_err(|_| AnswerError::Utf8(name)))
            .collect()
    }

    /// The only element of a BYTES output of shape `[1]`, as text.
    pub(super) fn text(&self, name: &'static str) -> Result<String, AnswerError> {
        let mut texts = self.texts(name)?;
        match (texts.pop(), texts.is_empty()) {
            (Some(text), true) => Ok(text),
            _ => Err(AnswerError::Shape {
                name,
                detail: "expected exactly one element".into(),
            }),
        }
    }
}
