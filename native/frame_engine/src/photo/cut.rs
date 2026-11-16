//! The rectangles round 2 cuts from (`Photo.rects` and `Photo.classes` of the Python engine): the
//! detector's answer or, when the request carries `boxes`, the product's stored boxes
//! (`_branches`: `photo.rects = stored_box_rects(...)`), scaled from normalised coordinates of the
//! oriented frame, rounded half to even, cut to the frame and at least one pixel wide and high
//! ([`crate::geometry::stored_box_rects`]), in the request's order. Stored boxes carry no classes
//! (so no `embed_classes`, which the plan refuses beside `boxes`) and no detector answer (so no
//! blur sums, which belong to `detect`).

use super::{Stop, engine};
use crate::branches::detect::{self, Detections};
use crate::geometry::stored_box_rects;

/// The blur sums' rectangles (`detect::blur_rects`): the summed ones, and each detection's
/// clamped rectangle or `None`.
pub type BlurRects = (Vec<[i64; 4]>, Vec<Option<[i64; 4]>>);

/// The rectangles round 2 cuts from, and where they came from.
pub struct Cut {
    /// Frame rectangles `x1, y1, x2, y2` (the last two exclusive), in order.
    pub rects: Vec<[i64; 4]>,
    /// The detector's class of each rectangle; `None` for stored boxes.
    pub classes: Option<Vec<i32>>,
    /// The blur sums' rectangles (`detect::blur_rects`), for the detector's rectangles only.
    pub blur: Option<BlurRects>,
}

impl Cut {
    /// The detector's rectangles, with their classes and blur rectangles.
    pub fn detected(d: &Detections, (w, h): (u32, u32)) -> Self {
        Self {
            rects: d.rects.clone(),
            classes: Some(d.classes.clone()),
            blur: Some(detect::blur_rects(d, w, h)),
        }
    }

    /// The stored boxes' rectangles.
    ///
    /// # Errors
    ///
    /// A coordinate that is not a finite number fails the photo, as Python's `round` raises on it.
    pub fn stored(boxes: &[[f64; 4]], (w, h): (u32, u32)) -> Result<Self, Stop> {
        if boxes.iter().flatten().any(|v| !v.is_finite()) {
            return Err(engine(
                "stored boxes hold a coordinate that is not a finite number",
            ));
        }
        Ok(Self {
            rects: stored_box_rects(boxes, i64::from(w), i64::from(h)),
            classes: None,
            blur: None,
        })
    }
}
