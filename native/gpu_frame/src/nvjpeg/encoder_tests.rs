//! The encoder's calls, recorded by a fake entry-point table: settings, order, the bitstream
//! fetch, the stream wait and the clean-up, against `nvjpeg_encode.py`.

use std::cell::RefCell;
use std::ffi::c_int;

use super::*;
use crate::cuda::CUresult;

thread_local! {
    static LOG: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static FAIL: RefCell<Option<(&'static str, c_int)>> = const { RefCell::new(None) };
}

fn note(call: &'static str, detail: String) -> c_int {
    LOG.with(|l| {
        l.borrow_mut()
            .push(format!("{call} {detail}").trim_end().to_owned())
    });
    FAIL.with(|f| match *f.borrow() {
        Some((name, status)) if name == call => status,
        _ => 0,
    })
}

fn taken() -> Vec<String> {
    LOG.with(|l| std::mem::take(&mut *l.borrow_mut()))
}

const STATE: usize = 0x5;
const PARAMS: usize = 0x9;
const JPEG: &[u8] = b"\xff\xd8 fake \xff\xd9";

unsafe extern "C" fn state_create(_: Handle, out: *mut EncoderState, _: CudaStream) -> c_int {
    // SAFETY: the encoder passes a valid out-parameter.
    unsafe { *out = STATE as EncoderState };
    note("StateCreate", String::new())
}
unsafe extern "C" fn state_destroy(s: EncoderState) -> c_int {
    note("StateDestroy", format!("{}", s as usize))
}
unsafe extern "C" fn params_create(_: Handle, out: *mut EncoderParams, _: CudaStream) -> c_int {
    // SAFETY: as above.
    unsafe { *out = PARAMS as EncoderParams };
    note("ParamsCreate", String::new())
}
unsafe extern "C" fn params_destroy(p: EncoderParams) -> c_int {
    note("ParamsDestroy", format!("{}", p as usize))
}
unsafe extern "C" fn set_quality(_: EncoderParams, q: c_int, _: CudaStream) -> c_int {
    note("SetQuality", q.to_string())
}
unsafe extern "C" fn set_encoding(_: EncoderParams, e: c_int, _: CudaStream) -> c_int {
    note("SetEncoding", format!("{e:#x}"))
}
unsafe extern "C" fn set_huffman(_: EncoderParams, o: c_int, _: CudaStream) -> c_int {
    note("SetOptimizedHuffman", o.to_string())
}
unsafe extern "C" fn set_sampling(_: EncoderParams, s: c_int, _: CudaStream) -> c_int {
    note("SetSamplingFactors", s.to_string())
}
unsafe extern "C" fn encode_image(
    _: Handle,
    s: EncoderState,
    p: EncoderParams,
    img: *const NvjpegImage,
    format: c_int,
    w: c_int,
    h: c_int,
    _: CudaStream,
) -> c_int {
    // SAFETY: the encoder passes a valid image description.
    let img = unsafe { &*img };
    note(
        "EncodeImage",
        format!(
            "state={} params={} addr={:#x} pitch={:?} format={format} {w}x{h}",
            s as usize, p as usize, img.channel[0] as usize, img.pitch
        ),
    )
}
unsafe extern "C" fn retrieve(
    _: Handle,
    _: EncoderState,
    data: *mut u8,
    len: *mut usize,
    _: CudaStream,
) -> c_int {
    let what = if data.is_null() { "length" } else { "data" };
    // SAFETY: `len` is valid; `data`, when given, holds `*len` bytes (the length answered first).
    unsafe {
        if data.is_null() {
            *len = JPEG.len() + 4;
        } else {
            std::ptr::copy_nonoverlapping(JPEG.as_ptr(), data, JPEG.len());
            *len = JPEG.len();
        }
    }
    note("Retrieve", what.to_owned())
}

fn fake() -> EncFns {
    EncFns {
        nvjpegEncoderStateCreate: state_create,
        nvjpegEncoderStateDestroy: state_destroy,
        nvjpegEncoderParamsCreate: params_create,
        nvjpegEncoderParamsDestroy: params_destroy,
        nvjpegEncoderParamsSetQuality: set_quality,
        nvjpegEncoderParamsSetEncoding: set_encoding,
        nvjpegEncoderParamsSetOptimizedHuffman: set_huffman,
        nvjpegEncoderParamsSetSamplingFactors: set_sampling,
        nvjpegEncodeImage: encode_image,
        nvjpegEncodeRetrieveBitstream: retrieve,
    }
}

fn fail_at(call: Option<(&'static str, c_int)>) {
    FAIL.with(|f| *f.borrow_mut() = call);
}

const PICTURE: Picture = Picture {
    addr: 0x1000,
    width: 7,
    height: 3,
};

#[test]
fn the_state_and_parameters_get_the_python_engines_settings() {
    fail_at(None);
    let objects = create(&fake(), ptr::null_mut(), ptr::null_mut()).expect("created");
    assert_eq!(
        (objects.state as usize, objects.params as usize),
        (STATE, PARAMS)
    );
    assert_eq!(
        taken(),
        [
            "StateCreate",
            "ParamsCreate",
            "SetEncoding 0xc0",
            "SetSamplingFactors 2",
            "SetOptimizedHuffman 0",
        ]
    );
}

#[test]
fn a_refused_setting_destroys_what_was_made() {
    fail_at(Some(("SetSamplingFactors", 2)));
    let e = create(&fake(), ptr::null_mut(), ptr::null_mut()).expect_err("refused");
    assert_eq!(e.call, "nvjpegEncoderParamsSetSamplingFactors");
    let log = taken();
    assert_eq!(&log[log.len() - 2..], ["ParamsDestroy 9", "StateDestroy 5"]);
    fail_at(None);
}

#[test]
fn an_encode_sets_quality_encodes_rgbi_fetches_length_then_data_then_waits() {
    fail_at(None);
    let objects = Objects {
        state: STATE as EncoderState,
        params: PARAMS as EncoderParams,
    };
    let mut waits = Vec::new();
    let jpeg = encode(
        &fake(),
        ptr::null_mut(),
        objects,
        &PICTURE,
        85,
        ptr::null_mut(),
        &mut || {
            waits.push(LOG.with(|l| l.borrow().len()));
            Ok(())
        },
    )
    .expect("a JPEG");
    assert_eq!(jpeg, JPEG, "truncated to the length the data call answered");
    assert_eq!(
        taken(),
        [
            "SetQuality 85",
            "EncodeImage state=5 params=9 addr=0x1000 pitch=[21, 0, 0, 0] format=5 7x3",
            "Retrieve length",
            "Retrieve data",
        ]
    );
    assert_eq!(waits, [4], "one wait, after the data");
}

#[test]
fn a_failed_encode_waits_for_the_stream_and_names_its_call() {
    let objects = Objects {
        state: STATE as EncoderState,
        params: PARAMS as EncoderParams,
    };
    fail_at(Some(("EncodeImage", 5)));
    let mut waited = 0;
    let e = encode(
        &fake(),
        ptr::null_mut(),
        objects,
        &PICTURE,
        90,
        ptr::null_mut(),
        &mut || {
            waited += 1;
            Ok(())
        },
    )
    .expect_err("refused");
    assert_eq!(waited, 1);
    assert!(e.is_out_of_memory(), "allocator failure answers gpu_memory");
    assert_eq!(
        taken(),
        [
            "SetQuality 90",
            "EncodeImage state=5 params=9 addr=0x1000 pitch=[21, 0, 0, 0] format=5 7x3"
        ]
    );

    let lost = encode(
        &fake(),
        ptr::null_mut(),
        objects,
        &PICTURE,
        90,
        ptr::null_mut(),
        &mut || {
            Err(CudaError::Driver {
                call: "cuStreamSynchronize",
                code: CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
            })
        },
    )
    .expect_err("lost");
    assert!(
        lost.is_context_lost(),
        "a lost context wins over the call's status"
    );
    fail_at(None);
    taken();
}

#[test]
fn a_picture_larger_than_its_buffer_is_refused_before_any_call() {
    struct Buf(usize);
    impl DeviceRange for Buf {
        fn device_addr(&self) -> u64 {
            0x2000
        }
        fn byte_len(&self) -> usize {
            self.0
        }
    }
    assert!(Picture::of(&Buf(63), 7, 3).is_ok());
    assert!(matches!(
        Picture::of(&Buf(62), 7, 3),
        Err(EncodeError::Shape(_))
    ));
    assert!(matches!(
        Picture::of(&Buf(63), 0, 3),
        Err(EncodeError::Shape(_))
    ));
}
