//! The refusal of retired environment variables.
//!
//! A renamed or removed setting left in a deployment's environment must stop the process, not be
//! ignored: a stale value that quietly does nothing looks like a working configuration. The list
//! is `contracts/json/retired_env.json`, rendered from `src/config/retired_env.py` by
//! `scripts/codegen/retired_env_contract.py` (a drift hook keeps the two equal), and embedded here
//! at build time. The refusal reads as the Python guard's does:
//! `retired env var(s) set: A was renamed to B; C was removed: <what to use instead>`.

use std::error::Error;
use std::fmt;

use serde::Deserialize;

use crate::Env;

const CONTRACT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../contracts/json/retired_env.json"
));

/// One retired variable and what happened to it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Retired {
    /// The environment variable name.
    pub name: String,
    /// The words after "`<name>` was": `renamed to <new name>` or `removed: <what to use instead>`.
    pub was: String,
}

#[derive(Deserialize)]
struct Document {
    retired: Vec<Retired>,
}

/// Why the process may not start on this environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetiredError {
    /// At least one retired variable is set. Sorted by name.
    Set(Vec<Retired>),
    /// The list itself cannot be read. A build with a broken contract file must not start.
    Contract(String),
}

impl fmt::Display for RetiredError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Set(found) => {
                f.write_str("retired env var(s) set: ")?;
                for (i, retired) in found.iter().enumerate() {
                    if i > 0 {
                        f.write_str("; ")?;
                    }
                    write!(f, "{} was {}", retired.name, retired.was)?;
                }
                Ok(())
            }
            Self::Contract(reason) => {
                write!(
                    f,
                    "the retired environment variable list is unreadable: {reason}"
                )
            }
        }
    }
}

impl Error for RetiredError {}

/// The list of retired variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetiredEnv {
    entries: Vec<Retired>,
}

impl RetiredEnv {
    /// The list this build embeds.
    ///
    /// # Errors
    ///
    /// [`RetiredError::Contract`] when the embedded file does not parse.
    pub fn embedded() -> Result<Self, RetiredError> {
        Self::parse(CONTRACT)
    }

    /// Read a list from the contract file's text.
    ///
    /// # Errors
    ///
    /// [`RetiredError::Contract`] when the text is not the contract's JSON, or names a variable
    /// twice (the generator never writes either).
    pub fn parse(text: &str) -> Result<Self, RetiredError> {
        let mut entries = serde_json::from_str::<Document>(text)
            .map_err(|e| RetiredError::Contract(e.to_string()))?
            .retired;
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        if let Some(pair) = entries.windows(2).find(|w| w[0].name == w[1].name) {
            return Err(RetiredError::Contract(format!(
                "{} is listed twice",
                pair[0].name
            )));
        }
        Ok(Self { entries })
    }

    /// Every retired variable, sorted by name.
    #[must_use]
    pub fn entries(&self) -> &[Retired] {
        &self.entries
    }

    /// Refuse when any listed variable is set in `env`, even to an empty value.
    ///
    /// # Errors
    ///
    /// [`RetiredError::Set`] with every listed variable that is set.
    pub fn check(&self, env: &Env) -> Result<(), RetiredError> {
        let found: Vec<Retired> = self
            .entries
            .iter()
            .filter(|retired| env.is_set(&retired.name))
            .cloned()
            .collect();
        if found.is_empty() {
            Ok(())
        } else {
            Err(RetiredError::Set(found))
        }
    }
}

/// Refuse to start when `env` holds a variable the embedded list retires. Every role of the
/// server calls this first.
///
/// # Errors
///
/// [`RetiredError::Set`] naming each retired variable found, or [`RetiredError::Contract`] when
/// the embedded list is unreadable.
pub fn reject_retired_env(env: &Env) -> Result<(), RetiredError> {
    RetiredEnv::embedded()?.check(env)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{"retired": [
        {"name": "OP_B_OLD", "was": "renamed to OP_B"},
        {"name": "OP_A_OLD", "was": "removed: it is a row now"}
    ]}"#;

    fn sample() -> RetiredEnv {
        RetiredEnv::parse(SAMPLE).unwrap()
    }

    #[test]
    fn the_embedded_list_is_the_contract_and_is_not_empty() {
        let list = RetiredEnv::embedded().unwrap();
        assert!(!list.entries().is_empty());
        assert!(list.entries().windows(2).all(|w| w[0].name < w[1].name));
    }

    #[test]
    fn a_clean_environment_starts() {
        assert_eq!(sample().check(&Env::from_pairs([("OP_A", "1")])), Ok(()));
    }

    #[test]
    fn a_retired_variable_refuses_and_names_itself_and_its_replacement() {
        let env = Env::from_pairs([("OP_B_OLD", "1")]);
        let err = sample().check(&env).unwrap_err();
        assert_eq!(
            err.to_string(),
            "retired env var(s) set: OP_B_OLD was renamed to OP_B"
        );
    }

    #[test]
    fn every_retired_variable_found_is_named_in_name_order() {
        let env = Env::from_pairs([("OP_B_OLD", "x"), ("OP_A_OLD", "y"), ("OP_OK", "z")]);
        let err = sample().check(&env).unwrap_err();
        assert_eq!(
            err.to_string(),
            "retired env var(s) set: OP_A_OLD was removed: it is a row now; \
             OP_B_OLD was renamed to OP_B"
        );
    }

    #[test]
    fn an_empty_value_still_counts_as_set() {
        let env = Env::from_pairs([("OP_A_OLD", "")]);
        assert!(matches!(sample().check(&env), Err(RetiredError::Set(_))));
    }

    #[test]
    fn a_value_that_is_not_unicode_still_counts_as_set() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        let env = Env::from_pairs([("OP_A_OLD", OsString::from_vec(vec![0xff]))]);
        assert!(matches!(sample().check(&env), Err(RetiredError::Set(_))));
    }

    #[test]
    fn a_broken_or_ambiguous_list_is_refused_not_ignored() {
        assert!(matches!(
            RetiredEnv::parse("{"),
            Err(RetiredError::Contract(_))
        ));
        let twice = r#"{"retired": [{"name": "X", "was": "a"}, {"name": "X", "was": "b"}]}"#;
        let err = RetiredEnv::parse(twice).unwrap_err();
        assert!(err.to_string().contains("X is listed twice"), "{err}");
    }

    #[test]
    fn the_embedded_list_refuses_through_the_one_entry_point() {
        let first = RetiredEnv::embedded().unwrap().entries()[0].clone();
        let env = Env::from_pairs([(first.name.clone(), "1")]);
        let err = reject_retired_env(&env).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("retired env var(s) set: {} was {}", first.name, first.was)
        );
    }
}
