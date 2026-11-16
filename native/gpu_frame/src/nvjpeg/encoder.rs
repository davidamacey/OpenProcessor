//! nvJPEG encoding of interleaved RGB already on the card (the Python engine's
//! `frame_engine/nvjpeg_encode.py`): only the bitstream crosses to the host.
//!
//! nvJPEG requires an encoder state and encode parameters per thread (documentation 1.3), so each
//! thread that encodes owns one [`Encoder`] made by [`Encoders::create`] on that thread, with the
//! thread's own CUDA stream; the library handle is the decode lanes' (thread safe, 2.2.13).
//!
//! Every setting is the Python engine's: baseline DCT (`0xC0`), 4:2:0 chroma, optimized Huffman
//! off (one pass), the quality set before every encode, `NVJPEG_INPUT_RGBI` with `width x 3` bytes
//! per row. The bitstream is fetched as `nvjpeg_encode.py` fetches it: the length query, the data,
//! then one wait for the stream; after a failed call the stream is waited for before the error
//! returns, since the caller frees the pixels next. Documentation:
//! <https://docs.nvidia.com/cuda/nvjpeg/index.html> (2.3.5 "JPEG Encoding": the encoder state and
//! parameters, `nvjpegEncodeImage`, `nvjpegEncodeRetrieveBitstream`).

use std::ffi::c_int;
use std::fmt;
use std::ptr;
use std::sync::Arc;

use super::ffi::{CudaStream, EncFns, EncoderParams, EncoderState, Handle, NvjpegImage, Shared};
use super::status::{NvjpegFailure, NvjpegStatus, check};
use crate::cuda::{CudaError, DeviceRange, Fault, Stream};

/// `NVJPEG_ENCODING_BASELINE_DCT`.
const ENCODING_BASELINE_DCT: c_int = 0xC0;
/// `NVJPEG_CSS_420`.
const CSS_420: c_int = 2;
/// `NVJPEG_INPUT_RGBI`.
const INPUT_RGBI: c_int = 5;

/// Why an encode gave no JPEG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// An nvJPEG call failed.
    Nvjpeg(NvjpegFailure),
    /// The stream wait failed.
    Cuda(CudaError),
    /// The pixels are fewer than `width x height x 3` bytes, or a side is zero or too large.
    Shape(String),
}

impl EncodeError {
    /// Whether the photo answers `gpu_memory` (`prep_budget.is_memory_error`).
    pub fn is_out_of_memory(&self) -> bool {
        match self {
            EncodeError::Nvjpeg(f) => f.status == NvjpegStatus::ALLOCATOR_FAILURE,
            EncodeError::Cuda(e) => e.fault() == Fault::OutOfMemory,
            EncodeError::Shape(_) => false,
        }
    }

    /// Whether the CUDA context is lost.
    pub fn is_context_lost(&self) -> bool {
        matches!(self, EncodeError::Cuda(e) if e.fault() == Fault::ContextLost)
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::Nvjpeg(e) => write!(f, "nvJPEG encode: {e}"),
            EncodeError::Cuda(e) => write!(f, "nvJPEG encode: {e}"),
            EncodeError::Shape(m) => write!(f, "nvJPEG encode: {m}"),
        }
    }
}

impl std::error::Error for EncodeError {}

/// What makes encoders: the decode lanes' library and handle, shared by every thread.
#[derive(Clone)]
pub struct Encoders {
    shared: Arc<Shared>,
}

impl fmt::Debug for Encoders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Encoders").finish_non_exhaustive()
    }
}

impl Encoders {
    pub(super) fn new(shared: Arc<Shared>) -> Encoders {
        Encoders { shared }
    }

    /// An encoder for the calling thread, encoding on `stream` (the context must be bound).
    ///
    /// # Errors
    ///
    /// [`EncodeError::Nvjpeg`] when nvJPEG refuses the state, the parameters or a setting.
    pub fn create(&self, stream: &Arc<Stream>) -> Result<Encoder, EncodeError> {
        let (fns, handle) = self.shared.encoder();
        let objects = create(fns, handle, stream.raw().cast()).map_err(EncodeError::Nvjpeg)?;
        Ok(Encoder {
            shared: Arc::clone(&self.shared),
            objects,
            stream: Arc::clone(stream),
        })
    }
}

/// One thread's encoder state and parameters, on that thread's stream: `Send`, not `Sync`.
pub struct Encoder {
    shared: Arc<Shared>,
    objects: Objects,
    stream: Arc<Stream>,
}

impl fmt::Debug for Encoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Encoder").finish_non_exhaustive()
    }
}

// SAFETY: nvJPEG encoder states and parameters may move between threads as long as one thread
// uses them at a time (documentation 1.3), which `&mut self` on `encode` enforces.
unsafe impl Send for Encoder {}

impl Encoder {
    /// The stream the encoder runs on: pixels written in order on it need no other wait.
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// The baseline 4:2:0 JPEG of `rgb`, interleaved RGB of `width x height` pixels on the card
    /// (`width x 3` bytes per row), at `quality` 1 to 100. Returns after the stream was waited for.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Shape`] when `rgb` is smaller than the picture; [`EncodeError::Nvjpeg`] or
    /// [`EncodeError::Cuda`] when a call or the stream wait fails (the stream was waited for).
    pub fn encode(
        &mut self,
        rgb: &dyn DeviceRange,
        (width, height): (u32, u32),
        quality: u32,
    ) -> Result<Vec<u8>, EncodeError> {
        let picture = Picture::of(rgb, width, height)?;
        let quality = c_int::try_from(quality)
            .map_err(|_| EncodeError::Shape(format!("quality {quality}")))?;
        let (fns, handle) = self.shared.encoder();
        let stream = &self.stream;
        encode(
            fns,
            handle,
            self.objects,
            &picture,
            quality,
            stream.raw().cast(),
            &mut || stream.synchronize(),
        )
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // nothing of this encoder may still run when its objects go
        let _ = self.stream.synchronize();
        let (fns, _) = self.shared.encoder();
        destroy(fns, self.objects);
    }
}

/// An encoder state and its parameters.
#[derive(Debug, Clone, Copy)]
struct Objects {
    state: EncoderState,
    params: EncoderParams,
}

/// The source of one encode: device address and size in pixels.
#[derive(Debug, Clone, Copy)]
struct Picture {
    addr: u64,
    width: c_int,
    height: c_int,
}

impl Picture {
    fn of(rgb: &dyn DeviceRange, width: u32, height: u32) -> Result<Picture, EncodeError> {
        let need = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3));
        let side = |v: u32| c_int::try_from(v).ok().filter(|v| *v > 0);
        match (need, side(width), side(height)) {
            (Some(need), Some(w), Some(h)) if need <= rgb.byte_len() => Ok(Picture {
                addr: rgb.device_addr(),
                width: w,
                height: h,
            }),
            _ => Err(EncodeError::Shape(format!(
                "{width} x {height} RGB in {} bytes",
                rgb.byte_len()
            ))),
        }
    }
}

/// Destroys what exists of `objects` (null ones are skipped).
fn destroy(fns: &EncFns, objects: Objects) {
    // SAFETY: each non-null object was created by `create` and is destroyed once, by the owner
    // that waited for its stream.
    unsafe {
        if !objects.params.is_null() {
            (fns.nvjpegEncoderParamsDestroy)(objects.params);
        }
        if !objects.state.is_null() {
            (fns.nvjpegEncoderStateDestroy)(objects.state);
        }
    }
}

/// The state and parameters with `nvjpeg_encode.py`'s fixed settings; on a failure what was made
/// is destroyed.
fn create(fns: &EncFns, handle: Handle, stream: CudaStream) -> Result<Objects, NvjpegFailure> {
    let mut objects = Objects {
        state: ptr::null_mut(),
        params: ptr::null_mut(),
    };
    // SAFETY: the handle is live; the out-parameters are valid; the parameters are set only
    // after they were created.
    let made = unsafe {
        check(
            "nvjpegEncoderStateCreate",
            (fns.nvjpegEncoderStateCreate)(handle, &mut objects.state, stream),
        )
        .and_then(|()| {
            check(
                "nvjpegEncoderParamsCreate",
                (fns.nvjpegEncoderParamsCreate)(handle, &mut objects.params, stream),
            )
        })
        .and_then(|()| {
            check(
                "nvjpegEncoderParamsSetEncoding",
                (fns.nvjpegEncoderParamsSetEncoding)(objects.params, ENCODING_BASELINE_DCT, stream),
            )
        })
        .and_then(|()| {
            check(
                "nvjpegEncoderParamsSetSamplingFactors",
                (fns.nvjpegEncoderParamsSetSamplingFactors)(objects.params, CSS_420, stream),
            )
        })
        .and_then(|()| {
            check(
                "nvjpegEncoderParamsSetOptimizedHuffman",
                (fns.nvjpegEncoderParamsSetOptimizedHuffman)(objects.params, 0, stream),
            )
        })
    };
    match made {
        Ok(()) => Ok(objects),
        Err(e) => {
            destroy(fns, objects);
            Err(e)
        }
    }
}

/// One encode on `stream`: the quality, the encode, the length, the data, then `sync`. After a
/// failed call `sync` runs before the error returns; a lost context it reports wins.
fn encode(
    fns: &EncFns,
    handle: Handle,
    objects: Objects,
    picture: &Picture,
    quality: c_int,
    stream: CudaStream,
    sync: &mut dyn FnMut() -> Result<(), CudaError>,
) -> Result<Vec<u8>, EncodeError> {
    match queue(fns, handle, objects, picture, quality, stream) {
        Ok(jpeg) => {
            sync().map_err(EncodeError::Cuda)?;
            Ok(jpeg)
        }
        Err(failure) => Err(match sync() {
            Err(e) if e.fault() == Fault::ContextLost => EncodeError::Cuda(e),
            _ => EncodeError::Nvjpeg(failure),
        }),
    }
}

/// The calls of one encode, in `nvjpeg_encode.py`'s order, without the final wait.
fn queue(
    fns: &EncFns,
    handle: Handle,
    objects: Objects,
    picture: &Picture,
    quality: c_int,
    stream: CudaStream,
) -> Result<Vec<u8>, NvjpegFailure> {
    let image = NvjpegImage {
        channel: [
            picture.addr as *mut u8,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        ],
        pitch: [picture.width as usize * 3, 0, 0, 0],
    };
    let mut length = 0usize;
    // SAFETY: the state and parameters are this encoder's; `image` names `width x height x 3`
    // device bytes (checked by `Picture::of`) the caller keeps alive until the stream wait that
    // follows; `length` is a valid out-parameter.
    unsafe {
        check(
            "nvjpegEncoderParamsSetQuality",
            (fns.nvjpegEncoderParamsSetQuality)(objects.params, quality, stream),
        )?;
        check(
            "nvjpegEncodeImage",
            (fns.nvjpegEncodeImage)(
                handle,
                objects.state,
                objects.params,
                &image,
                INPUT_RGBI,
                picture.width,
                picture.height,
                stream,
            ),
        )?;
        check(
            "nvjpegEncodeRetrieveBitstream",
            (fns.nvjpegEncodeRetrieveBitstream)(
                handle,
                objects.state,
                ptr::null_mut(),
                &mut length,
                stream,
            ),
        )?;
    }
    let mut data = vec![0u8; length];
    // SAFETY: `data` holds the `length` bytes nvJPEG just reported; it answers the bytes written.
    check("nvjpegEncodeRetrieveBitstream", unsafe {
        (fns.nvjpegEncodeRetrieveBitstream)(
            handle,
            objects.state,
            data.as_mut_ptr(),
            &mut length,
            stream,
        )
    })?;
    data.truncate(length);
    Ok(data)
}

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod tests;
