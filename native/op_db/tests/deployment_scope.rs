//! Only the crates listed here may open a deployment-scope transaction, which sees every project's
//! rows. It is the Rust twin of `src/db/scope_allowlist.py`: a crate that needs the scope adds
//! itself here with its reason, in a change a reviewer sees.

use std::fs;
use std::path::{Path, PathBuf};

/// Crate and why it needs to see across projects.
const ALLOWED: &[(&str, &str)] = &[
    ("op_db", "defines it"),
    (
        "op_jobs",
        "the runner claims a job before it knows the project, and writes deployment-wide jobs",
    ),
];

struct Source {
    krate: String,
    path: String,
    text: String,
}

fn violations(sources: &[Source]) -> Vec<String> {
    let mut found = Vec::new();
    for source in sources {
        if ALLOWED.iter().any(|(name, _)| *name == source.krate) {
            continue;
        }
        for (number, line) in source.text.lines().enumerate() {
            let code = line.trim_start();
            if !code.starts_with("//") && code.contains("deployment_tx(") {
                found.push(format!(
                    "{}:{}: `{}` may not open a deployment-scope transaction; add it to ALLOWED in op_db/tests/deployment_scope.rs with its reason",
                    source.path,
                    number + 1,
                    source.krate
                ));
            }
        }
    }
    found
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn workspace_sources() -> Vec<Source> {
    let native = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("op_db sits in the native workspace")
        .to_path_buf();
    let mut sources = Vec::new();
    for entry in fs::read_dir(&native)
        .expect("native/ is readable")
        .flatten()
    {
        let krate = entry.file_name().to_string_lossy().into_owned();
        let mut files = Vec::new();
        rust_files(&entry.path().join("src"), &mut files);
        for file in files {
            sources.push(Source {
                krate: krate.clone(),
                path: file
                    .strip_prefix(&native)
                    .expect("under native/")
                    .display()
                    .to_string(),
                text: fs::read_to_string(&file).expect("a source file is readable"),
            });
        }
    }
    sources
}

#[test]
fn a_crate_outside_the_list_that_opens_the_deployment_scope_is_found() {
    let planted = [Source {
        krate: "op_api".into(),
        path: "op_api/src/x.rs".into(),
        text: "fn f() {}\nlet tx = op_db::deployment_tx(&pool, \"why\").await;\n".into(),
    }];
    let found = violations(&planted);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].contains("op_api/src/x.rs:2"), "{found:?}");
}

#[test]
fn the_listed_crates_and_comments_are_not_violations() {
    let ok = [
        Source {
            krate: "op_jobs".into(),
            path: "op_jobs/src/claim.rs".into(),
            text: "let tx = deployment_tx(&pool, \"claim\").await?;".into(),
        },
        Source {
            krate: "op_api".into(),
            path: "op_api/src/y.rs".into(),
            text: "    // never call deployment_tx( here\n".into(),
        },
    ];
    assert!(violations(&ok).is_empty());
}

#[test]
fn no_crate_of_the_workspace_breaks_the_rule() {
    let sources = workspace_sources();
    assert!(
        sources.iter().any(|s| s.krate == "op_db"),
        "found no sources: the scan is broken"
    );
    let found = violations(&sources);
    assert!(found.is_empty(), "{}", found.join("\n"));
}
