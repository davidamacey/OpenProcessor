//! Property test of plan 18 section 7: no generated key resolves outside its root, also through
//! symlinks that lead out of it.
#![cfg(test)]

use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::sync::LazyLock;

use op_storage::keys::StorageKey;
use op_storage::local::LocalRoots;
use proptest::prelude::*;
use tempfile::TempDir;

struct World {
    _dir: TempDir,
    roots: LocalRoots,
}

/// Roots with files, folders and symlinks that stay inside, leave, loop and dangle.
static WORLD: LazyLock<World> = LazyLock::new(|| {
    let dir = tempfile::tempdir().unwrap();
    let base = std::fs::canonicalize(dir.path()).unwrap();
    let make = |p: &str| std::fs::create_dir_all(base.join(p)).unwrap();
    for d in [
        "landing/dir",
        "sources/dir/sub",
        "outside/dir",
        "sources_evil",
    ] {
        make(d);
    }
    for f in [
        "landing/dir/a",
        "sources/dir/a",
        "sources/dir/sub/b",
        "outside/a",
        "outside/dir/a",
        "sources_evil/a",
    ] {
        std::fs::write(base.join(f), b"x").unwrap();
    }
    for root in ["landing", "sources"] {
        let r = base.join(root);
        symlink(base.join("outside"), r.join("out")).unwrap();
        symlink(base.join("outside/a"), r.join("out_file")).unwrap();
        symlink(base.join("sources_evil"), r.join("evil")).unwrap();
        symlink("dir", r.join("in")).unwrap();
        symlink("loop", r.join("loop")).unwrap();
        symlink("missing", r.join("dangling")).unwrap();
        symlink("../outside", r.join("dir/up")).unwrap();
    }
    let roots = LocalRoots::new(&base.join("landing"), &base.join("sources")).unwrap();
    World { _dir: dir, roots }
});

fn segment() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => prop::sample::select(vec![
            "dir", "sub", "a", "b", "out", "out_file", "evil", "in", "up", "loop", "dangling",
            "..", ".", "", "missing",
        ]).prop_map(str::to_owned),
        1 => "[a-z./\\\\\u{0}\n%#]{0,6}",
    ]
}

fn key() -> impl Strategy<Value = String> {
    (
        prop::sample::select(vec!["landing", "source", "projects", "nope"]),
        prop::collection::vec(segment(), 0..6),
    )
        .prop_map(|(root, segs)| {
            std::iter::once(root.to_owned())
                .chain(segs)
                .collect::<Vec<_>>()
                .join("/")
        })
}

fn canonical_root_of(key: &StorageKey) -> PathBuf {
    match key.as_str().split('/').next() {
        Some("landing") => WORLD.roots.landing.clone(),
        _ => WORLD.roots.sources.clone(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn no_key_resolves_outside_its_root(raw in key()) {
        if let Ok(key) = StorageKey::parse(&raw)
            && let Ok(path) = WORLD.roots.resolve(&key)
        {
            prop_assert!(path.starts_with(canonical_root_of(&key)), "{raw:?} -> {path:?}");
        }
    }

    #[test]
    fn the_shared_key_rule_is_at_least_as_strict_as_the_object_stores(
        raw in "[ -~\\x00-\\x1f\\x7f/.\\\\\u{e9}]{0,14}",
    ) {
        if op_local_file::validate_key(&raw).is_ok() {
            let path = object_store::path::Path::parse(&raw);
            prop_assert!(path.is_ok(), "{raw:?} passes the shared rule but not Path::parse");
            let parsed = path.unwrap();
            prop_assert_eq!(parsed.as_ref(), raw.as_str());
        }
    }

    #[test]
    fn a_parsed_key_has_no_dot_segment_and_prints_back_as_given(raw in key()) {
        if let Ok(key) = StorageKey::parse(&raw) {
            prop_assert_eq!(key.as_str(), raw.as_str());
            for seg in key.as_str().split('/') {
                prop_assert!(!seg.is_empty() && seg != "." && seg != "..");
                prop_assert!(!seg.contains(['\\', '\0', '\n']));
            }
        }
    }
}

#[test]
fn the_world_really_offers_the_ways_out() {
    let ok = |k: &str| WORLD.roots.resolve(&StorageKey::parse(k).unwrap());
    assert!(ok("source/dir/a").is_ok());
    assert!(ok("source/in/a").is_ok());
    assert!(ok("source/out/a").is_err());
    assert!(ok("source/out_file").is_err());
    assert!(ok("source/dir/up/a").is_err());
    assert!(ok("landing/evil/a").is_err());
    assert!(ok("source/loop").is_err());
}
