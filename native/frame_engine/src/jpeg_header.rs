//! What the engine reads of a JPEG without decoding it: the frame header (size, components,
//! precision, progressive or lossless), the EXIF orientation, the embedded ICC profile's bytes and
//! the end-of-image marker, and from them the route (`gpu`, `cpu_<reason>` or refused). A port of
//! `src/domain/jpeg_markers.py` and `frame_engine/prep_geometry.py` (`jpeg_header`, `jpeg_end`,
//! `route`, `oriented_size`), held to them by goldens that include seeded damage to the files.
//!
//! The route is decided before any frame budget is reserved (plan 17 section 2.2), so a truncated
//! file is routed `cpu_truncated` and answered from its own byte need. Where the plan sketches the
//! end-of-image check as a reverse search from the end of the file, this is the Python rule: walk
//! the marker segments and the entropy-coded scans, so bytes appended after the image (maker
//! trailers, a motion photo's video) are not part of it and `end` is where the decoder's input
//! stops.
//!
//! The ICC profile is returned as its joined bytes; what kind of profile it is (sRGB, convert or
//! other) is decided with the colour step (plan 17 N1a), which needs a colour management library.

const SOF_MARKERS: [u8; 13] = [
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
];
const PROGRESSIVE: [u8; 4] = [0xC2, 0xC6, 0xCA, 0xCE];
const LOSSLESS: [u8; 4] = [0xC3, 0xC7, 0xCB, 0xCF];
const ICC_MARKER: u8 = 0xE2;
const ICC_TAG: &[u8; 12] = b"ICC_PROFILE\0";
const EXIF_TAG: &[u8; 6] = b"Exif\0\0";
const EXIF_MARKER: u8 = 0xE1;
const SOS: u8 = 0xDA;
const EOI: u8 = 0xD9;

/// Markers with no length field: `TEM` and the restart markers.
fn standalone(marker: u8) -> bool {
    marker == 0x01 || (0xD0..=0xD7).contains(&marker)
}

/// What a start marker was followed by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A frame header was read.
    Jpeg,
    /// A start marker and no readable frame header.
    Corrupt,
    /// Not a JPEG.
    Other,
}

/// What the decoder choice depends on, read from the markers without decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegHeader {
    /// What the start markers held: a readable frame, a corrupt file, or not a JPEG.
    pub kind: Kind,
    /// Frame width in pixels before the EXIF orientation; 0 when no frame header was read.
    pub width: u32,
    /// Frame height in pixels before the EXIF orientation; 0 when no frame header was read.
    pub height: u32,
    /// Colour components of the frame (1 grey, 3 colour, 4 CMYK).
    pub components: u32,
    /// Bits per sample.
    pub precision: u32,
    /// Whether the frame is progressive (SOF2).
    pub progressive: bool,
    /// EXIF orientation 1 to 8; 1 when absent or invalid.
    pub orientation: u32,
    /// Whether the frame is lossless (SOF3 or the other lossless frame types).
    pub lossless: bool,
    /// Bytes of the image proper, end-of-image marker included (set by [`route`]).
    pub end: usize,
    /// The embedded ICC profile (the APP2 chunks before the frame header, joined), if any.
    pub icc: Option<Vec<u8>>,
}

impl JpegHeader {
    fn without_frame(kind: Kind, orientation: u32) -> Self {
        Self {
            kind,
            width: 0,
            height: 0,
            components: 0,
            precision: 0,
            progressive: false,
            orientation,
            lossless: false,
            end: 0,
            icc: None,
        }
    }
}

fn be16(data: &[u8], at: usize) -> Option<u16> {
    data.get(at..at + 2)
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
}

/// `header_segments`: `(marker, payload)` of each length-carrying segment after the start marker,
/// up to the first scan or the first byte that is not a marker. A length that reaches past the
/// end gives the bytes that are there.
pub(crate) fn header_segments(data: &[u8]) -> Vec<(u8, &[u8])> {
    let mut out = Vec::new();
    let mut i = 2usize;
    while i + 4 <= data.len() {
        if data[i] != 0xFF {
            break;
        }
        let marker = data[i + 1];
        if marker == 0xFF {
            i += 1;
            continue;
        }
        if standalone(marker) {
            i += 2;
            continue;
        }
        if marker == SOS {
            break;
        }
        let length = usize::from(be16(data, i + 2).unwrap_or(0));
        let from = (i + 4).min(data.len());
        let to = (i + 2 + length).min(data.len()).max(from);
        out.push((marker, &data[from..to]));
        i += 2 + length;
    }
    out
}

/// `join_icc_chunks`: the profile split over APP2 payloads (`ICC_PROFILE\0`, the sequence
/// number, the chunk count, then its bytes), joined in sequence order; `None` when the count
/// does not match (Pillow's rule).
fn join_icc_chunks(mut chunks: Vec<&[u8]>) -> Option<Vec<u8>> {
    if chunks.is_empty() || chunks.iter().any(|c| c.len() < 14) {
        return None;
    }
    chunks.sort_unstable();
    if usize::from(chunks[0][13]) != chunks.len() {
        return None;
    }
    Some(
        chunks
            .iter()
            .flat_map(|c| c[14..].iter().copied())
            .collect(),
    )
}

/// `exif_orientation`: the tag 0x0112 of IFD0 of an APP1 `Exif` payload; 1 when absent or invalid.
fn exif_orientation(payload: &[u8]) -> u32 {
    let tiff = payload.get(6..).unwrap_or(&[]);
    if tiff.len() < 8 || !matches!(&tiff[..2], b"II" | b"MM") {
        return 1;
    }
    let little = &tiff[..2] == b"II";
    let u16_at = |at: usize| -> Option<u16> {
        let b = tiff.get(at..at + 2)?;
        Some(if little {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    };
    let ifd = {
        let b = [tiff[4], tiff[5], tiff[6], tiff[7]];
        (if little {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        }) as usize
    };
    if ifd + 2 > tiff.len() {
        return 1;
    }
    let count = usize::from(u16_at(ifd).unwrap_or(0));
    for k in 0..count {
        let start = ifd + 2 + 12 * k;
        let Some(entry) = tiff.get(start..start + 12) else {
            break;
        };
        let tag = if little {
            u16::from_le_bytes([entry[0], entry[1]])
        } else {
            u16::from_be_bytes([entry[0], entry[1]])
        };
        if tag == 0x0112 {
            let value = if little {
                u16::from_le_bytes([entry[8], entry[9]])
            } else {
                u16::from_be_bytes([entry[8], entry[9]])
            };
            return if (1..=8).contains(&value) {
                u32::from(value)
            } else {
                1
            };
        }
    }
    1
}

/// `jpeg_header`: parses `data`'s markers up to the first frame header.
pub fn jpeg_header(data: &[u8]) -> JpegHeader {
    if !data.starts_with(&[0xFF, 0xD8]) {
        return JpegHeader::without_frame(Kind::Other, 1);
    }
    let mut orientation = 1;
    let mut icc: Vec<&[u8]> = Vec::new();
    for (marker, payload) in header_segments(data) {
        if marker == EXIF_MARKER && payload.starts_with(EXIF_TAG) {
            orientation = exif_orientation(payload);
        } else if marker == ICC_MARKER && payload.starts_with(ICC_TAG) {
            icc.push(payload);
        } else if SOF_MARKERS.contains(&marker) && payload.len() >= 6 {
            return JpegHeader {
                kind: Kind::Jpeg,
                width: u32::from(u16::from_be_bytes([payload[3], payload[4]])),
                height: u32::from(u16::from_be_bytes([payload[1], payload[2]])),
                components: u32::from(payload[5]),
                precision: u32::from(payload[0]),
                progressive: PROGRESSIVE.contains(&marker),
                orientation,
                lossless: LOSSLESS.contains(&marker),
                end: 0,
                icc: join_icc_chunks(icc),
            };
        }
    }
    JpegHeader::without_frame(Kind::Corrupt, orientation)
}

/// `jpeg_end`: the offset just past the end-of-image marker, or `None` when the file has none.
///
/// Walks the marker segments (so an end marker inside a header segment, such as the EXIF
/// thumbnail's, is skipped) and the entropy-coded scans. Inside a scan `0xFF` is stuffed
/// (`FF00`), starts a restart marker (`FFD0` to `FFD7`) or is a fill byte before another `0xFF`;
/// an `0xFF` followed by any other byte is a marker that ends the scan.
pub fn jpeg_end(data: &[u8]) -> Option<usize> {
    let n = data.len();
    let mut i = 2usize;
    let mut in_scan = false;
    loop {
        let marker;
        if in_scan {
            // the scan's first byte is at `i`, so the marker that ends it starts at `i` or later
            let mut from = i;
            let found = loop {
                let at = from + memchr::memchr(0xFF, data.get(from..n.saturating_sub(1))?)?;
                let next = data[at + 1];
                if next != 0x00 && next != 0xFF && !(0xD0..=0xD7).contains(&next) {
                    break at + 1;
                }
                from = at + 1;
            };
            marker = data[found];
            i = found + 1;
        } else {
            if i >= n || data[i] != 0xFF {
                return None;
            }
            while i < n && data[i] == 0xFF {
                i += 1;
            }
            if i >= n {
                return None;
            }
            marker = data[i];
            i += 1;
        }
        if marker == EOI {
            return Some(i);
        }
        if standalone(marker) {
            continue;
        }
        let length = usize::from(be16(data, i)?);
        i += length;
        in_scan = marker == SOS;
    }
}

/// Where a photo is decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decoder {
    /// nvJPEG takes it.
    Gpu,
    /// The host decode, for this reason (`not_jpeg`, `precision`, `components`, `progressive`,
    /// `truncated`, `small`).
    Cpu(&'static str),
    /// Nothing is decoded: `corrupt` or `huge`.
    Refused(&'static str),
}

/// A route and the header it was read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    /// Who decodes the photo, or why nobody does.
    pub decoder: Decoder,
    /// The header the choice was made from.
    pub header: JpegHeader,
}

fn structural_reason(h: &JpegHeader) -> Option<&'static str> {
    if h.kind != Kind::Jpeg {
        Some("not_jpeg")
    } else if h.lossless || h.precision != 8 {
        Some("precision")
    } else if h.components != 1 && h.components != 3 {
        Some("components")
    } else if h.progressive {
        Some("progressive")
    } else {
        None
    }
}

/// `route`: the decoder, from the header alone.
///
/// nvJPEG takes a baseline, 8-bit, 1- or 3-component JPEG that has its end-of-image marker and
/// is at least `min_megapixels`; every other readable image goes to the host decode with the first
/// reason that applies. Fail closed: the GPU decoders return an image for a truncated file, so a
/// JPEG without its end-of-image marker is never given to them. Refused without a decode: a JPEG
/// start marker with no readable frame header (`corrupt`) and a baseline JPEG above
/// `max_megapixels` (`huge`).
pub fn route(data: &[u8], min_megapixels: f64, max_megapixels: f64) -> Route {
    let header = jpeg_header(data);
    if header.kind == Kind::Corrupt {
        return Route {
            decoder: Decoder::Refused("corrupt"),
            header,
        };
    }
    if let Some(reason) = structural_reason(&header) {
        return Route {
            decoder: Decoder::Cpu(reason),
            header,
        };
    }
    let Some(end) = jpeg_end(data) else {
        return Route {
            decoder: Decoder::Cpu("truncated"),
            header,
        };
    };
    let header = JpegHeader { end, ..header };
    let mp = f64::from(header.width) * f64::from(header.height) / 1e6;
    let decoder = if mp < min_megapixels {
        Decoder::Cpu("small")
    } else if mp > max_megapixels {
        Decoder::Refused("huge")
    } else {
        Decoder::Gpu
    };
    Route { decoder, header }
}

/// `oriented_size`: `(width, height)` after the EXIF orientation is applied.
pub fn oriented_size(h: &JpegHeader) -> (u32, u32) {
    if (5..=8).contains(&h.orientation) {
        (h.height, h.width)
    } else {
        (h.width, h.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(marker: u8, w: u16, h: u16, comps: u8, precision: u8) -> Vec<u8> {
        let mut body = vec![precision];
        body.extend_from_slice(&h.to_be_bytes());
        body.extend_from_slice(&w.to_be_bytes());
        body.push(comps);
        for c in 0..comps {
            body.extend_from_slice(&[c + 1, 0x11, 0]);
        }
        segment(marker, &body)
    }

    fn segment(marker: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0xFF, marker];
        out.extend_from_slice(
            &u16::try_from(payload.len() + 2)
                .expect("short")
                .to_be_bytes(),
        );
        out.extend_from_slice(payload);
        out
    }

    fn photo(w: u16, h: u16) -> Vec<u8> {
        let mut data = vec![0xFF, 0xD8];
        data.extend(frame(0xC0, w, h, 3, 8));
        data.extend(segment(SOS, &[3, 1, 0, 2, 0, 3, 0, 0, 63, 0]));
        data.extend_from_slice(&[0x12, 0x34, 0xFF, 0x00, 0x56]);
        data.extend_from_slice(&[0xFF, EOI]);
        data
    }

    #[test]
    fn a_baseline_photo_is_read_and_sent_to_the_gpu() {
        let data = photo(640, 480);
        let h = jpeg_header(&data);
        assert_eq!(
            (h.kind, h.width, h.height, h.components, h.precision),
            (Kind::Jpeg, 640, 480, 3, 8)
        );
        assert_eq!(
            (h.progressive, h.lossless, h.orientation, h.icc.clone()),
            (false, false, 1, None)
        );
        assert_eq!(jpeg_end(&data), Some(data.len()));
        let r = route(&data, 0.0, 100.0);
        assert_eq!(r.decoder, Decoder::Gpu);
        assert_eq!(r.header.end, data.len());
    }

    #[test]
    fn a_file_with_no_end_marker_goes_to_the_host_before_anything_is_reserved() {
        let mut data = photo(640, 480);
        data.truncate(data.len() - 2);
        assert_eq!(jpeg_end(&data), None);
        assert_eq!(route(&data, 0.0, 100.0).decoder, Decoder::Cpu("truncated"));
    }

    #[test]
    fn bytes_after_the_end_marker_are_not_part_of_the_image() {
        let mut data = photo(64, 64);
        let end = data.len();
        data.extend_from_slice(b"MPF\0\xff\xd8\xff\xd9 more");
        assert_eq!(jpeg_end(&data), Some(end));
        assert_eq!(route(&data, 0.0, 100.0).header.end, end);
    }

    #[test]
    fn the_order_of_the_reasons_is_the_pythons() {
        let cases: [(Vec<u8>, Decoder); 6] = [
            (b"\x89PNG\r\n".to_vec(), Decoder::Cpu("not_jpeg")),
            (
                [vec![0xFF, 0xD8], frame(0xC3, 64, 64, 3, 8)].concat(),
                Decoder::Cpu("precision"),
            ),
            (
                [vec![0xFF, 0xD8], frame(0xC0, 64, 64, 3, 12)].concat(),
                Decoder::Cpu("precision"),
            ),
            (
                [vec![0xFF, 0xD8], frame(0xC0, 64, 64, 4, 8)].concat(),
                Decoder::Cpu("components"),
            ),
            (
                [vec![0xFF, 0xD8], frame(0xC2, 64, 64, 3, 8)].concat(),
                Decoder::Cpu("progressive"),
            ),
            (vec![0xFF, 0xD8, 0x00, 0x01], Decoder::Refused("corrupt")),
        ];
        for (data, want) in cases {
            assert_eq!(route(&data, 0.0, 100.0).decoder, want);
        }
        assert_eq!(
            route(&photo(100, 100), 1.0, 100.0).decoder,
            Decoder::Cpu("small")
        );
        assert_eq!(
            route(&photo(10001, 10000), 0.0, 100.0).decoder,
            Decoder::Refused("huge")
        );
        assert_eq!(
            route(&photo(10000, 10000), 0.0, 100.0).decoder,
            Decoder::Gpu
        );
        assert_eq!(route(&photo(1000, 1000), 1.0, 100.0).decoder, Decoder::Gpu);
    }

    #[test]
    fn a_turned_photo_has_its_sides_swapped() {
        let mut h = jpeg_header(&photo(640, 480));
        assert_eq!(oriented_size(&h), (640, 480));
        for (orientation, want) in [
            (1, (640, 480)),
            (4, (640, 480)),
            (5, (480, 640)),
            (8, (480, 640)),
        ] {
            h.orientation = orientation;
            assert_eq!(oriented_size(&h), want, "orientation {orientation}");
        }
    }

    #[test]
    fn icc_chunks_are_joined_in_sequence_order_and_a_count_mismatch_is_no_profile() {
        let chunk =
            |seq: u8, total: u8, body: &[u8]| [ICC_TAG.as_slice(), &[seq, total], body].concat();
        let (a, b) = (chunk(1, 2, b"AA"), chunk(2, 2, b"BB"));
        assert_eq!(join_icc_chunks(vec![&b, &a]), Some(b"AABB".to_vec()));
        assert_eq!(join_icc_chunks(vec![&a]), None);
        assert_eq!(join_icc_chunks(vec![]), None);
        assert_eq!(join_icc_chunks(vec![&ICC_TAG[..]]), None);
    }
}
