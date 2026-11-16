//! What the suites share.

use std::path::PathBuf;

use serde_json::Value;

/// A file of the repository, by its path from the root.
pub fn repo_file(path: &str) -> String {
    let full = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path);
    std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("{}: {e}", full.display()))
}

pub fn repo_json(path: &str) -> Value {
    serde_json::from_str(&repo_file(path)).expect("the file is JSON")
}
