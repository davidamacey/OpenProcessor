//! The browser-shaped multipart body, built as a stream of frames over photos already in memory.
//!
//! The form is what `ingestUpload` in `frontend/src/lib/api_ingest.ts` sends, with the field order
//! of plan 18 section 4.7 (`source`, `image_paths`, `run_id`, then the `images` parts) or the order
//! that function uses today (`images` parts first, then `image_paths`, `source`; no `run_id`).
//! The encoding follows the HTML standard's multipart/form-data algorithm
//! (<https://html.spec.whatwg.org/multipage/form-control-infrastructure.html#multipart-form-data>):
//! `Content-Disposition: form-data; name="..."` per part, `; filename="..."` and a `Content-Type`
//! for a file, `"`, CR and LF in a filename percent-escaped, CRLF line ends, and the boundary in the
//! `Content-Type` of the request. The body length is known in advance and sent as `Content-Length`,
//! as a browser does for a `FormData` of files.
//!
//! No maintained crate writes a streamed multipart body for hyper 1.x (`multer`, the one the
//! server side and the in-process test use, only parses; `reqwest`'s writer brings a whole second
//! client), and the writer is one concatenation of literals and photo slices, so it is written
//! here. A photo is never copied: its frames are slices of the shared `Bytes`.

use std::convert::Infallible;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use bytes::Bytes;
use hyper::body::Frame;
use serde_json::Value;

/// Which order the form's fields are sent in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FieldOrder {
    /// `source`, `image_paths`, `run_id`, then the `images` parts: the order plan 18 section 4.7
    /// asks of the frontend, so the server can commit each photo as its part ends.
    Browser,
    /// The `images` parts, then `image_paths`, then `source`, and no `run_id`: what the frontend
    /// sends today.
    Legacy,
}

/// One photo of the list.
#[derive(Debug, Clone)]
pub struct Photo {
    /// The stable identifier sent in `image_paths` (the server keeps it as `source_identifier`).
    pub identifier: String,
    /// The `filename` of the `images` part.
    pub filename: String,
    /// The encoded file.
    pub data: Bytes,
}

/// The text fields of the form.
#[derive(Debug, Clone)]
pub struct Fields {
    /// The provenance tag, field `source`.
    pub source: String,
    /// The tag of the call, field `run_id`; not sent in the legacy order.
    pub run_id: String,
}

/// When each photo part's last frame was taken by the connection, in photo order, as the
/// connection asked for the frame after it (hyper polls the body again only when its write buffer
/// can take more: at most 16 queued buffers, so at most 16 chunks lie between this instant and the
/// socket). It is a client-side bound, not the instant the server received the part.
#[derive(Debug, Default)]
pub struct PartStamps(Mutex<Vec<Instant>>);

impl PartStamps {
    fn push(&self, at: Instant) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(at);
    }

    /// The stamps taken so far, one per photo part whose last frame has been taken.
    pub fn taken(&self) -> Vec<Instant> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[derive(Debug)]
enum Segment {
    Literal(Bytes),
    Photo(Bytes),
}

/// The body's frames, in order; an [`Iterator`] so `futures_util::stream::iter` turns it into the
/// stream `StreamBody` wraps. Each call is one frame of at most `chunk` bytes.
#[derive(Debug)]
pub struct Frames {
    segments: Vec<Segment>,
    index: usize,
    offset: usize,
    chunk: usize,
    stamps: Arc<PartStamps>,
}

impl Iterator for Frames {
    type Item = Result<Frame<Bytes>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let segment = self.segments.get(self.index)?;
            let (data, is_photo) = match segment {
                Segment::Literal(b) => (b, false),
                Segment::Photo(b) => (b, true),
            };
            if self.offset >= data.len() {
                if is_photo {
                    self.stamps.push(Instant::now());
                }
                self.index += 1;
                self.offset = 0;
                continue;
            }
            let end = if is_photo {
                data.len().min(self.offset + self.chunk)
            } else {
                data.len()
            };
            let frame = data.slice(self.offset..end);
            self.offset = end;
            return Some(Ok(Frame::data(frame)));
        }
    }
}

/// A request body ready to send.
#[derive(Debug)]
pub struct Built {
    /// The request's `Content-Type`, with the boundary.
    pub content_type: String,
    /// The body's exact length in bytes, sent as `Content-Length`.
    pub content_length: u64,
    /// The frames to stream.
    pub frames: Frames,
    /// Filled as the frames are taken.
    pub stamps: Arc<PartStamps>,
}

/// The form-data boundary for request number `seq`: the 16 alphanumerics after
/// `----WebKitFormBoundary` that Chromium writes, derived from `seq` (SplitMix64) so a run's
/// requests are reproducible.
pub fn boundary(seq: u64) -> String {
    const ALPHABET: &[u8; 62] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut state = seq.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = String::from("----WebKitFormBoundary");
    for _ in 0..16 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.push(char::from(ALPHABET[usize::try_from(z % 62).unwrap_or(0)]));
    }
    out
}

/// The media type a browser reports for a file name, by extension; `application/octet-stream`
/// when it does not know the extension.
pub fn content_type_of(filename: &str) -> &'static str {
    let ext = filename.rsplit_once('.').map_or("", |(_, e)| e);
    match ext.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" | "jpe" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        "heic" => "image/heic",
        "heif" => "image/heif",
        _ => "application/octet-stream",
    }
}

fn escape_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        match c {
            '"' => out.push_str("%22"),
            '\r' => out.push_str("%0D"),
            '\n' => out.push_str("%0A"),
            c => out.push(c),
        }
    }
    out
}

/// Accumulates the literal text between photo parts.
struct Writer<'a> {
    boundary: &'a str,
    segments: Vec<Segment>,
    literal: Vec<u8>,
    first_part: bool,
}

impl<'a> Writer<'a> {
    fn new(boundary: &'a str) -> Self {
        Writer {
            boundary,
            segments: Vec::new(),
            literal: Vec::new(),
            first_part: true,
        }
    }

    fn open_part(&mut self, disposition: &str, content_type: Option<&str>) {
        if !self.first_part {
            self.literal.extend_from_slice(b"\r\n");
        }
        self.first_part = false;
        self.literal.extend_from_slice(b"--");
        self.literal.extend_from_slice(self.boundary.as_bytes());
        self.literal
            .extend_from_slice(format!("\r\nContent-Disposition: {disposition}").as_bytes());
        if let Some(t) = content_type {
            self.literal
                .extend_from_slice(format!("\r\nContent-Type: {t}").as_bytes());
        }
        self.literal.extend_from_slice(b"\r\n\r\n");
    }

    fn text(&mut self, name: &str, value: &str) {
        self.open_part(&format!("form-data; name=\"{name}\""), None);
        self.literal.extend_from_slice(value.as_bytes());
    }

    fn file(&mut self, photo: &Photo) {
        let name = escape_filename(&photo.filename);
        self.open_part(
            &format!("form-data; name=\"images\"; filename=\"{name}\""),
            Some(content_type_of(&photo.filename)),
        );
        self.flush_literal();
        self.segments.push(Segment::Photo(photo.data.clone()));
    }

    fn flush_literal(&mut self) {
        if !self.literal.is_empty() {
            let text = std::mem::take(&mut self.literal);
            self.segments.push(Segment::Literal(Bytes::from(text)));
        }
    }

    fn finish(mut self) -> Vec<Segment> {
        self.literal.extend_from_slice(b"\r\n--");
        self.literal.extend_from_slice(self.boundary.as_bytes());
        self.literal.extend_from_slice(b"--\r\n");
        self.flush_literal();
        self.segments
    }
}

/// Build the body of one upload request for `photos`, in the given field order, streamed in
/// frames of at most `chunk` bytes (at least 1).
///
/// `image_paths` is the JSON list of the photos' identifiers, as the frontend sends it.
pub fn build(
    order: FieldOrder,
    boundary: &str,
    fields: &Fields,
    photos: &[Photo],
    chunk: usize,
) -> Built {
    let identifiers = Value::Array(
        photos
            .iter()
            .map(|p| Value::String(p.identifier.clone()))
            .collect(),
    )
    .to_string();
    let mut w = Writer::new(boundary);
    match order {
        FieldOrder::Browser => {
            w.text("source", &fields.source);
            w.text("image_paths", &identifiers);
            w.text("run_id", &fields.run_id);
            photos.iter().for_each(|p| w.file(p));
        }
        FieldOrder::Legacy => {
            photos.iter().for_each(|p| w.file(p));
            w.text("image_paths", &identifiers);
            w.text("source", &fields.source);
        }
    }
    let segments = w.finish();
    let content_length = segments
        .iter()
        .map(|s| match s {
            Segment::Literal(b) | Segment::Photo(b) => b.len() as u64,
        })
        .sum();
    let stamps = Arc::new(PartStamps::default());
    Built {
        content_type: format!("multipart/form-data; boundary={boundary}"),
        content_length,
        frames: Frames {
            segments,
            index: 0,
            offset: 0,
            chunk: chunk.max(1),
            stamps: Arc::clone(&stamps),
        },
        stamps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn photo(name: &str, data: &[u8]) -> Photo {
        Photo {
            identifier: format!("set/{name}"),
            filename: name.to_string(),
            data: Bytes::copy_from_slice(data),
        }
    }

    fn fields() -> Fields {
        Fields {
            source: "upload".into(),
            run_id: "run-1".into(),
        }
    }

    fn body_of(b: Built) -> Vec<u8> {
        b.frames
            .flat_map(|f| match f {
                Ok(frame) => frame.into_data().map(|d| d.to_vec()).unwrap_or_default(),
                Err(never) => match never {},
            })
            .collect()
    }

    #[test]
    fn the_browser_order_writes_exactly_this_text() {
        let built = build(
            FieldOrder::Browser,
            "BB",
            &fields(),
            &[photo("a.jpg", b"AAA"), photo("b\"x.png", b"BB")],
            4,
        );
        let expected = "--BB\r\nContent-Disposition: form-data; name=\"source\"\r\n\r\nupload\r\n\
            --BB\r\nContent-Disposition: form-data; name=\"image_paths\"\r\n\r\n[\"set/a.jpg\",\"set/b\\\"x.png\"]\r\n\
            --BB\r\nContent-Disposition: form-data; name=\"run_id\"\r\n\r\nrun-1\r\n\
            --BB\r\nContent-Disposition: form-data; name=\"images\"; filename=\"a.jpg\"\r\nContent-Type: image/jpeg\r\n\r\nAAA\r\n\
            --BB\r\nContent-Disposition: form-data; name=\"images\"; filename=\"b%22x.png\"\r\nContent-Type: image/png\r\n\r\nBB\r\n\
            --BB--\r\n";
        assert_eq!(built.content_length, expected.len() as u64);
        assert_eq!(built.content_type, "multipart/form-data; boundary=BB");
        assert_eq!(String::from_utf8(body_of(built)).unwrap(), expected);
    }

    #[test]
    fn the_legacy_order_puts_the_images_first_and_sends_no_run_id() {
        let text = String::from_utf8(body_of(build(
            FieldOrder::Legacy,
            "BB",
            &fields(),
            &[photo("a.jpg", b"AAA")],
            2,
        )))
        .unwrap();
        let at = |needle: &str| text.find(needle).unwrap();
        assert!(at("name=\"images\"") < at("name=\"image_paths\""));
        assert!(at("name=\"image_paths\"") < at("name=\"source\""));
        assert!(!text.contains("run_id"));
    }

    #[test]
    fn photo_frames_are_at_most_a_chunk() {
        let p = Photo {
            identifier: "x".into(),
            filename: "x.jpg".into(),
            data: Bytes::from(vec![7u8; 1000]),
        };
        let built = build(FieldOrder::Browser, "BB", &fields(), &[p], 300);
        let sizes: Vec<usize> = built
            .frames
            .map(|f| f.unwrap().into_data().unwrap())
            .filter(|d| d.iter().all(|b| *b == 7))
            .map(|d| d.len())
            .collect();
        assert_eq!(sizes, [300, 300, 300, 100]);
    }

    #[test]
    fn a_stamp_is_taken_after_each_photos_last_frame_and_not_before() {
        let built = build(
            FieldOrder::Browser,
            "BB",
            &fields(),
            &[photo("a.jpg", b"AAAAA"), photo("b.jpg", b"BBBBB")],
            2,
        );
        let stamps = Arc::clone(&built.stamps);
        let seen: Vec<(char, usize)> = built
            .frames
            .map(|f| {
                let d = f.unwrap().into_data().unwrap();
                let kind = if d.iter().all(|b| *b == b'A') {
                    'A'
                } else if d.iter().all(|b| *b == b'B') {
                    'B'
                } else {
                    'L'
                };
                (kind, stamps.taken().len())
            })
            .collect();
        // photo A: 2 + 2 + 1 bytes with no stamp yet; the frame after its last (a literal)
        // is the first to see one stamp; the same for B and the closing literal
        let a_frames: Vec<_> = seen.iter().filter(|(k, _)| *k == 'A').collect();
        assert_eq!(a_frames.len(), 3);
        assert!(a_frames.iter().all(|(_, n)| *n == 0));
        let after_a = seen.iter().position(|(k, _)| *k == 'A').unwrap() + 3;
        assert_eq!(seen[after_a], ('L', 1));
        let b_frames: Vec<_> = seen.iter().filter(|(k, _)| *k == 'B').collect();
        assert!(b_frames.iter().all(|(_, n)| *n == 1));
        assert_eq!(seen.last(), Some(&('L', 2)));
    }

    #[test]
    fn boundaries_are_browser_like_and_differ_per_request() {
        let (a, b) = (boundary(1), boundary(2));
        assert_eq!(a.len(), "----WebKitFormBoundary".len() + 16);
        assert!(a.starts_with("----WebKitFormBoundary"));
        assert_ne!(a, b);
        assert_eq!(a, boundary(1));
    }

    #[test]
    fn content_types_follow_the_extension() {
        assert_eq!(content_type_of("A.JPG"), "image/jpeg");
        assert_eq!(content_type_of("x.tiff"), "image/tiff");
        assert_eq!(content_type_of("noext"), "application/octet-stream");
    }
}
