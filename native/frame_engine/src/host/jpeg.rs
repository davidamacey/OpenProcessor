//! The host JPEG decode: libjpeg-turbo through its TurboJPEG 3 API, as `src/utils/image_decode.py`
//! decodes a JPEG (plan 17 section 3, "Host JPEG").
//!
//! * A JPEG Pillow opens as `RGB` or `L` (one or three components) goes to simplejpeg 1.9.0:
//!   `tjDecompressHeader3`, then `tjDecompress2` to RGB with `TJFLAG_STOPONWARNING`, accurate DCT
//!   and fancy upsampling (TurboJPEG's defaults). Strict: every libjpeg warning (a truncated file,
//!   corrupt entropy data) fails the decode. Two failures take Pillow's path instead, as the API
//!   does: the header read's "Could not determine subsampling level" (`tjDecompressHeader3`
//!   throws it for `TJSAMP_UNKNOWN`) and the warning "Invalid SOS parameters for sequential JPEG".
//! * Pillow's path (those two, and a CMYK or YCCK file, which Pillow opens as `CMYK`): libjpeg
//!   without the stop on warnings, as Pillow's decoder runs it, except that a file without its
//!   end-of-image marker is refused (Pillow raises "image file is truncated" with
//!   `LOAD_TRUNCATED_IMAGES` off). CMYK samples are always inverted (Pillow's raw mode `CMYK;I`,
//!   "assume adobe conventions", whatever the file's segments say), then converted as
//!   `Image.convert('RGB')` does.
//!
//! TurboJPEG documents `TJPARAM_STOPONWARNING` and `tj3Decompress8` in
//! https://libjpeg-turbo.org/Documentation/Documentation (TurboJPEG C API); the handle is one per
//! call, so nothing is shared between threads.

use std::ffi::CStr;
use std::ptr::NonNull;

use turbojpeg_sys as tj;

use super::modes::cmyk_to_rgb;

/// `tjDecompressHeader3`'s refusal that sends simplejpeg's caller to Pillow.
const HEADER_REJECTED: &str = "Could not determine subsampling level";
/// The libjpeg warning Pillow decodes through (`image_decode.SOS_WARNING`).
const SOS_WARNING: &str = "Invalid SOS parameters for sequential JPEG";

/// One TurboJPEG decompressor, destroyed when dropped.
struct Handle(NonNull<std::ffi::c_void>);

impl Handle {
    fn new() -> Result<Self, String> {
        // SAFETY: `tj3Init` takes the init type by value and returns a new handle or null.
        let raw = unsafe { tj::tj3Init(tj::TJINIT_TJINIT_DECOMPRESS as i32) };
        NonNull::new(raw)
            .map(Handle)
            .ok_or_else(|| "tj3Init failed".to_owned())
    }

    fn error(&self) -> String {
        // SAFETY: the handle is live; `tj3GetErrorStr` returns a NUL-terminated string owned by
        // the handle (or a static one), read before the handle is used again.
        unsafe { CStr::from_ptr(tj::tj3GetErrorStr(self.0.as_ptr())) }
            .to_string_lossy()
            .into_owned()
    }

    fn set(&mut self, param: tj::TJPARAM, value: i32) -> Result<(), String> {
        // SAFETY: the handle is live and owned by this value; parameters are plain integers.
        if unsafe { tj::tj3Set(self.0.as_ptr(), param as i32, value) } != 0 {
            return Err(self.error());
        }
        Ok(())
    }

    fn get(&mut self, param: tj::TJPARAM) -> i32 {
        // SAFETY: the handle is live and owned by this value.
        unsafe { tj::tj3Get(self.0.as_ptr(), param as i32) }
    }

    /// A failed call's error, unless it was only a warning and warnings are allowed (TurboJPEG
    /// returns -1 with `TJERR_WARNING` when the stop on warnings is off and the call completed).
    fn check(&mut self, rc: i32, warnings_allowed: bool) -> Result<(), String> {
        if rc == 0 {
            return Ok(());
        }
        // SAFETY: the handle is live and owned by this value.
        let code = unsafe { tj::tj3GetErrorCode(self.0.as_ptr()) };
        if warnings_allowed && code == tj::TJERR_TJERR_WARNING as i32 {
            return Ok(());
        }
        Err(self.error())
    }

    fn header(&mut self, data: &[u8], warnings_allowed: bool) -> Result<(), String> {
        // SAFETY: `data` is a live slice of exactly the length passed; the library only reads it.
        let rc = unsafe {
            tj::tj3DecompressHeader(self.0.as_ptr(), data.as_ptr(), data.len() as tj::size_t)
        };
        self.check(rc, warnings_allowed)
    }

    /// Decode into `out`, `width * height * channels` bytes of `format`.
    fn decompress(
        &mut self,
        data: &[u8],
        out: &mut [u8],
        width: usize,
        format: tj::TJPF,
        channels: usize,
        warnings_allowed: bool,
    ) -> Result<(), String> {
        let pitch = i32::try_from(width * channels).map_err(|_| "row too wide".to_owned())?;
        // SAFETY: `data` is live for the call; `out` holds `pitch * height` bytes for the header
        // just read (checked by the caller), so the library writes inside it.
        let rc = unsafe {
            tj::tj3Decompress8(
                self.0.as_ptr(),
                data.as_ptr(),
                data.len() as tj::size_t,
                out.as_mut_ptr(),
                pitch,
                format,
            )
        };
        self.check(rc, warnings_allowed)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: the handle came from `tj3Init` and is destroyed once, here.
        unsafe { tj::tj3Destroy(self.0.as_ptr()) }
    }
}

/// What a decode gives: interleaved RGB rows of `width x height`.
#[derive(Debug)]
pub struct Rgb {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Pillow's mode for a JPEG with `components` components: `RGB`/`L` take simplejpeg, the rest
/// (`CMYK`) Pillow's own decoder.
pub fn decode(data: &[u8], components: u32, has_end: bool) -> Result<Rgb, String> {
    if matches!(components, 1 | 3) {
        match strict(data) {
            Ok(rgb) => Ok(rgb),
            Err(e) if e.contains(HEADER_REJECTED) || e.contains(SOS_WARNING) => {
                lenient(data, has_end)
            }
            Err(e) => Err(e),
        }
    } else {
        lenient(data, has_end)
    }
}

fn dims(h: &mut Handle) -> Result<(usize, usize), String> {
    let w = usize::try_from(h.get(tj::TJPARAM_TJPARAM_JPEGWIDTH)).map_err(|e| e.to_string())?;
    let ht = usize::try_from(h.get(tj::TJPARAM_TJPARAM_JPEGHEIGHT)).map_err(|e| e.to_string())?;
    Ok((w, ht))
}

fn rgb(pixels: Vec<u8>, w: usize, h: usize) -> Result<Rgb, String> {
    Ok(Rgb {
        pixels,
        width: u32::try_from(w).map_err(|e| e.to_string())?,
        height: u32::try_from(h).map_err(|e| e.to_string())?,
    })
}

/// simplejpeg's `decode_jpeg(data, colorspace='RGB')` with its default `strict=True`.
fn strict(data: &[u8]) -> Result<Rgb, String> {
    let mut h = Handle::new()?;
    h.set(tj::TJPARAM_TJPARAM_STOPONWARNING, 1)?;
    h.header(data, false)?;
    if h.get(tj::TJPARAM_TJPARAM_SUBSAMP) == tj::TJSAMP_TJSAMP_UNKNOWN {
        return Err(format!("{HEADER_REJECTED} of JPEG image"));
    }
    let (w, ht) = dims(&mut h)?;
    let mut out = vec![0u8; w * ht * 3];
    h.decompress(data, &mut out, w, tj::TJPF_TJPF_RGB, 3, false)?;
    rgb(out, w, ht)
}

/// Pillow's decoder: warnings do not stop it; a missing end-of-image marker does.
fn lenient(data: &[u8], has_end: bool) -> Result<Rgb, String> {
    if !has_end {
        return Err("image file is truncated".into());
    }
    let mut h = Handle::new()?;
    h.header(data, true)?;
    let (w, ht) = dims(&mut h)?;
    let cs = h.get(tj::TJPARAM_TJPARAM_COLORSPACE);
    let cmyk_space = [tj::TJCS_TJCS_CMYK, tj::TJCS_TJCS_YCCK].map(|c| c as i32);
    if cmyk_space.contains(&cs) {
        let mut cmyk = vec![0u8; w * ht * 4];
        h.decompress(data, &mut cmyk, w, tj::TJPF_TJPF_CMYK, 4, true)?;
        cmyk.iter_mut().for_each(|v| *v = 255 - *v);
        return rgb(cmyk_to_rgb(&cmyk), w, ht);
    }
    let mut out = vec![0u8; w * ht * 3];
    h.decompress(data, &mut out, w, tj::TJPF_TJPF_RGB, 3, true)?;
    rgb(out, w, ht)
}
