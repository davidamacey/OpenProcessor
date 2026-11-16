//! The EXIF orientation as the API's host decode reads it: `Image.getexif().get(0x0112, 1)` of
//! Pillow 13.0.0 (`src/utils/image_decode.py`), which is wider than the GPU route's header read.
//!
//! * The EXIF block is the container's (`info['exif']`: a JPEG's `Exif` APP1 segments joined, a
//!   PNG's `eXIf` chunk, a WebP's `EXIF` chunk) or, for a TIFF, the file's own first directory.
//! * `Exif.load` strips a leading `Exif\0\0`; what follows must start with a TIFF header
//!   (`TiffImagePlugin.PREFIXES`), else Pillow raises and the photo does not decode. Past the
//!   header Pillow reads leniently: a directory it cannot read gives no tags.
//! * Without an orientation tag, the XMP packet's first `tiff:Orientation="N"` or
//!   `tiff:Orientation>N` (one digit) stands in (`Image.getexif`'s XMP fallback).
//!
//! The pixels move only for 2 to 8 (`orient_array`, `ImageOps.exif_transpose`).

/// `TiffImagePlugin.PREFIXES`: classic and BigTIFF headers in both byte orders.
const PREFIXES: [&[u8; 4]; 6] = [
    b"MM\x00\x2a",
    b"II\x2a\x00",
    b"MM\x2a\x00",
    b"II\x00\x2a",
    b"MM\x00\x2b",
    b"II\x2b\x00",
];
const ORIENTATION: u16 = 0x0112;

/// Why Pillow refuses an EXIF block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BadExif;

/// The orientation tag of an EXIF block: `Ok(None)` when it has none, `Err` when Pillow raises.
pub fn exif_orientation(block: &[u8]) -> Result<Option<u32>, BadExif> {
    let tiff = block.strip_prefix(b"Exif\0\0").unwrap_or(block);
    if tiff.is_empty() {
        return Ok(None);
    }
    let head = tiff.get(..4).ok_or(BadExif)?;
    if !PREFIXES.iter().any(|p| p.as_slice() == head) {
        return Err(BadExif);
    }
    if head[2..] == [0x00, 0x2b] || head[2..] == [0x2b, 0x00] {
        return Ok(None);
    }
    Ok(ifd0_tag(tiff, head[0] == b'I'))
}

fn read(tiff: &[u8], at: usize, n: usize, little: bool) -> Option<u32> {
    let b = tiff.get(at..at.checked_add(n)?)?;
    let fold = |acc: u32, x: &u8| acc << 8 | u32::from(*x);
    Some(if little {
        b.iter().rev().fold(0, fold)
    } else {
        b.iter().fold(0, fold)
    })
}

/// The orientation entry of the first directory (a SHORT or LONG value held in the entry).
fn ifd0_tag(tiff: &[u8], little: bool) -> Option<u32> {
    let ifd = usize::try_from(read(tiff, 4, 4, little)?).ok()?;
    let count = usize::try_from(read(tiff, ifd, 2, little)?).ok()?;
    (0..count).find_map(|k| {
        let entry = ifd + 2 + 12 * k;
        if read(tiff, entry, 2, little)? != u32::from(ORIENTATION) {
            return None;
        }
        match read(tiff, entry + 2, 2, little)? {
            3 => read(tiff, entry + 8, 2, little),
            4 => read(tiff, entry + 8, 4, little),
            _ => None,
        }
    })
}

/// The XMP fallback: the digit after the first `tiff:Orientation="` or `tiff:Orientation>`.
pub fn xmp_orientation(xmp: &[u8]) -> Option<u32> {
    const KEY: &[u8] = b"tiff:Orientation";
    let mut from = 0;
    while let Some(k) = memchr::memmem::find(&xmp[from..], KEY) {
        let at = from + k + KEY.len();
        let digit = match xmp.get(at) {
            Some(b'=') if xmp.get(at + 1) == Some(&b'"') => xmp.get(at + 2),
            Some(b'>') => xmp.get(at + 1),
            _ => None,
        };
        if let Some(d) = digit.filter(|d| d.is_ascii_digit()) {
            return Some(u32::from(d - b'0'));
        }
        from = at;
    }
    None
}

/// The orientation the API applies: the EXIF tag, else the XMP one, else 1.
pub fn orientation(exif: Option<&[u8]>, xmp: Option<&[u8]>) -> Result<u32, BadExif> {
    let tagged = match exif {
        Some(block) => exif_orientation(block)?,
        None => None,
    };
    Ok(tagged
        .or_else(|| xmp.and_then(xmp_orientation))
        .unwrap_or(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(little: bool, tag: u16, kind: u16, value: u32) -> Vec<u8> {
        let mut b: Vec<u8> = if little {
            b"Exif\0\0II\x2a\x00\x08\x00\x00\x00\x01\x00".to_vec()
        } else {
            b"Exif\0\0MM\x00\x2a\x00\x00\x00\x08\x00\x01".to_vec()
        };
        let put16 = |b: &mut Vec<u8>, v: u16| {
            b.extend(if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            })
        };
        put16(&mut b, tag);
        put16(&mut b, kind);
        b.extend(if little {
            1u32.to_le_bytes()
        } else {
            1u32.to_be_bytes()
        });
        if kind == 3 {
            put16(&mut b, u16::try_from(value).unwrap());
            b.extend([0, 0]);
        } else {
            b.extend(if little {
                value.to_le_bytes()
            } else {
                value.to_be_bytes()
            });
        }
        b
    }

    #[test]
    fn the_tag_is_read_in_either_byte_order_and_type() {
        for little in [true, false] {
            assert_eq!(exif_orientation(&block(little, 0x0112, 3, 6)), Ok(Some(6)));
            assert_eq!(exif_orientation(&block(little, 0x0112, 4, 8)), Ok(Some(8)));
            assert_eq!(exif_orientation(&block(little, 0x0110, 3, 6)), Ok(None));
        }
        assert_eq!(exif_orientation(b"Exif\0\0"), Ok(None));
        assert_eq!(exif_orientation(b"Exif\0\0junk header"), Err(BadExif));
        assert_eq!(exif_orientation(b"II\x2a\x00\xff\xff\xff\x7f"), Ok(None));
    }

    #[test]
    fn xmp_stands_in_only_without_an_exif_tag() {
        let xmp = br#"<x tiff:Orientation="x"/><y tiff:Orientation="3"/>"#;
        assert_eq!(xmp_orientation(xmp), Some(3));
        assert_eq!(
            xmp_orientation(b"<tiff:Orientation>7</tiff:Orientation>"),
            Some(7)
        );
        assert_eq!(orientation(None, Some(xmp)), Ok(3));
        assert_eq!(
            orientation(Some(&block(true, 0x0112, 3, 2)), Some(xmp)),
            Ok(2)
        );
        assert_eq!(
            orientation(Some(&block(true, 0x0100, 3, 2)), Some(xmp)),
            Ok(3)
        );
        assert_eq!(orientation(None, None), Ok(1));
    }
}
