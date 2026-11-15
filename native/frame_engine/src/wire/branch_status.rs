//! The `branch_status` output: one JSON object with an entry per reported branch, written as
//! Python's `json.dumps(..., sort_keys=True)` writes it (`encode_branch_status`).

use std::collections::BTreeMap;

/// A branch's state; `error` carries a reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BranchState {
    Ok,
    NotRun,
    Error(&'static str),
}

pub const UNSUPPORTED_BRANCH: &str = "unsupported_branch";
pub const MODEL_ERROR: &str = "model_error";
pub const DEPENDENCY_FAILED: &str = "dependency_failed";
pub const IMAGE_FAILED: &str = "image_failed";

/// `{"detect": {"status": "ok"}, "ocr": {"reason": "model_error", "status": "error"}}`.
pub fn encode(states: &BTreeMap<String, BranchState>) -> String {
    let entries: Vec<String> = states
        .iter()
        .map(|(name, state)| {
            let body = match state {
                BranchState::Ok => r#"{"status": "ok"}"#.to_owned(),
                BranchState::NotRun => r#"{"status": "not_run"}"#.to_owned(),
                BranchState::Error(reason) => {
                    format!(r#"{{"reason": "{reason}", "status": "error"}}"#)
                }
            };
            format!(r#""{name}": {body}"#)
        })
        .collect();
    format!("{{{}}}", entries.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_text_is_what_python_dumps_with_sorted_keys() {
        let states = BTreeMap::from([
            ("embed_frame".to_owned(), BranchState::Ok),
            ("detect".to_owned(), BranchState::Error(MODEL_ERROR)),
            ("ocr".to_owned(), BranchState::NotRun),
        ]);
        assert_eq!(
            encode(&states),
            r#"{"detect": {"reason": "model_error", "status": "error"}, "embed_frame": {"status": "ok"}, "ocr": {"status": "not_run"}}"#
        );
        assert_eq!(encode(&BTreeMap::new()), "{}");
    }
}
