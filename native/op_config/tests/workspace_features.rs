//! The workspace unifies Cargo features: `frame_engine` turns on serde_json's
//! `arbitrary_precision` and `preserve_order`, and they then apply to the server crates in
//! `cargo test --workspace`. These round-trips fail if the shapes the wire models rely on (an
//! internally tagged enum, `#[serde(flatten)]`) break under those features, which are known to
//! change how numbers pass through buffered content.

use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Tagged {
    Box { x: f64, n: u64, i: i64 },
    Count { n: u64 },
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Inner {
    score: f64,
    count: u64,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Outer {
    id: i64,
    #[serde(flatten)]
    inner: Inner,
}

#[test]
fn an_internally_tagged_enum_round_trips_integers_and_floats() {
    let value = Tagged::Box {
        x: 0.25,
        n: 7,
        i: -3,
    };
    let text = serde_json::to_string(&value).unwrap();
    assert_eq!(serde_json::from_str::<Tagged>(&text).unwrap(), value);
    let count = Tagged::Count { n: u64::MAX };
    let text = serde_json::to_string(&count).unwrap();
    assert_eq!(serde_json::from_str::<Tagged>(&text).unwrap(), count);
}

#[test]
fn a_flattened_struct_round_trips_integers_and_floats() {
    let value = Outer {
        id: -9,
        inner: Inner {
            score: 0.5,
            count: 12,
        },
    };
    let text = serde_json::to_string(&value).unwrap();
    assert_eq!(serde_json::from_str::<Outer>(&text).unwrap(), value);
}
