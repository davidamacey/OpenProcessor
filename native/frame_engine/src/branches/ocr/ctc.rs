//! CTC collapse of the recogniser's per-step best class and its probability
//! (`frame_engine/ocr/ctc.py`, PaddleOCR's `CTCLabelDecode`): a line is the characters of the
//! steps whose index is not the blank (0), not the previous step's and inside the dictionary;
//! its confidence is the mean of those steps' probabilities (numpy's float32 mean: the pairwise
//! sum, divided in float32), `char_min` the lowest.

use std::path::Path;

/// Where the image puts the recogniser's dictionary (`frame_engine/settings.py`
/// `ocr_dictionary`; `docker/frame-engine-native/Dockerfile` copies
/// `models/paddleocr_rec_trt/ppocrv6_dict.txt` there).
pub const DICTIONARY_PATH: &str = "/app/ocr/ppocrv6_dict.txt";

/// One decoded line: its text, mean and lowest probability (both 0 for an empty line).
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    pub text: String,
    pub mean: f32,
    pub lowest: f32,
}

/// The recogniser's classes: `blank`, the dictionary's lines, then the space the model was
/// trained with.
#[derive(Debug, Clone)]
pub struct CtcDecoder {
    characters: Vec<String>,
}

impl CtcDecoder {
    /// From the dictionary's text (one character per line), read as Python's text mode reads
    /// it (`\r\n` and `\r` are line ends).
    pub fn from_text(text: &str) -> Self {
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        let mut characters = vec!["blank".to_owned()];
        // `[line.rstrip('\n') for line in f]`: a last line without its newline counts, an
        // empty text has no lines
        characters.extend(
            text.split_inclusive('\n')
                .map(|l| l.trim_end_matches('\n').to_owned()),
        );
        characters.push(" ".to_owned());
        Self { characters }
    }

    /// Read at start; a missing or unreadable file refuses the start with its path.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        std::fs::read_to_string(path)
            .map(|t| Self::from_text(&t))
            .map_err(|e| format!("the OCR dictionary {} cannot be read: {e}", path.display()))
    }

    /// Classes, the blank and the space included.
    pub fn len(&self) -> usize {
        self.characters.len()
    }

    /// Whether only the blank and the space are known (an empty dictionary).
    pub fn is_empty(&self) -> bool {
        self.characters.len() <= 2
    }

    /// The line of one recogniser row: `indices` and `scores` of its steps. An index past the
    /// classes is dropped; a negative one picks from the end as a Python list does, and one past
    /// even that is a fault of the answer.
    pub fn decode(&self, indices: &[i32], scores: &[f32]) -> Result<Line, String> {
        let n = self.characters.len() as i64;
        let mut text = String::new();
        let mut kept = Vec::new();
        let mut prev = 0i32;
        for (&i, &s) in indices.iter().zip(scores) {
            if i != 0 && i != prev && i64::from(i) < n {
                let at = if i < 0 {
                    n + i64::from(i)
                } else {
                    i64::from(i)
                };
                let c = usize::try_from(at)
                    .ok()
                    .and_then(|k| self.characters.get(k))
                    .ok_or_else(|| format!("recogniser index {i} is outside the dictionary"))?;
                text.push_str(c);
                kept.push(s);
            }
            prev = i;
        }
        if kept.is_empty() {
            return Ok(Line {
                text,
                mean: 0.0,
                lowest: 0.0,
            });
        }
        Ok(Line {
            text,
            mean: pairwise_sum_f32(&kept) / kept.len() as f32,
            lowest: kept.iter().copied().fold(f32::INFINITY, f32::min),
        })
    }
}

/// numpy's `FLOAT_pairwise_sum` (the float32 twin of `geometry::pairwise_sum`).
fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        a.iter().fold(0.0, |r, x| r + x)
    } else if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (acc, x) in r.iter_mut().zip(&a[i..i + 8]) {
                *acc += x;
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for x in &a[i..] {
            res += x;
        }
        res
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise_sum_f32(&a[..half]) + pairwise_sum_f32(&a[half..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoder() -> CtcDecoder {
        CtcDecoder::from_text("a\nb\nc\n")
    }

    #[test]
    fn blanks_and_repeats_collapse_and_scores_follow_the_kept_steps() {
        let d = decoder();
        assert_eq!(d.len(), 5, "blank, a, b, c, space");
        let line = d
            .decode(
                &[1, 1, 0, 1, 2, 2, 4],
                &[0.5, 0.9, 0.1, 0.25, 0.75, 0.6, 1.0],
            )
            .expect("decoded");
        assert_eq!(line.text, "aab ");
        assert_eq!((line.mean, line.lowest), (0.625, 0.25));
    }

    #[test]
    fn an_empty_line_and_unknown_indices() {
        let d = decoder();
        let empty = d.decode(&[0, 0], &[0.9, 0.9]).expect("decoded");
        assert_eq!(
            (empty.text.as_str(), empty.mean, empty.lowest),
            ("", 0.0, 0.0)
        );
        assert_eq!(d.decode(&[7, 3], &[0.5, 0.5]).expect("decoded").text, "c");
        assert_eq!(d.decode(&[-1], &[0.5]).expect("decoded").text, " ");
        assert!(d.decode(&[-6], &[0.5]).is_err());
        assert!(CtcDecoder::from_file(Path::new("/nonexistent/dict.txt")).is_err());
        assert_eq!(
            CtcDecoder::from_text("x\ny").len(),
            4,
            "a last line without its newline"
        );
    }
}
