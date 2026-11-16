//! The identifier of a project.

use std::fmt;
use std::str::FromStr;

use uuid::Uuid;

/// A project's id. The scope of a [`crate::ProjectTx`] is one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProjectId(Uuid);

impl ProjectId {
    /// Wrap a UUID.
    #[must_use]
    pub const fn new(id: Uuid) -> Self {
        Self(id)
    }

    /// The UUID.
    #[must_use]
    pub const fn uuid(self) -> Uuid {
        self.0
    }
}

impl fmt::Display for ProjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.hyphenated().fmt(f)
    }
}

impl FromStr for ProjectId {
    type Err = uuid::Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(text).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_project_id_round_trips_and_a_bad_one_is_refused() {
        let text = "0198c4e2-7d1a-7c33-9a0b-3f6f2b1d9e10";
        let id: ProjectId = text.parse().unwrap();
        assert_eq!(id.to_string(), text);
        assert!("not-a-project".parse::<ProjectId>().is_err());
    }
}
