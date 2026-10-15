"""The public-data guard of the docs screenshot capture: refuse pages that would show a
project the operator has not declared public."""

from __future__ import annotations


# Pages that render a list of every project the backend holds (the project list, the combine
# wizard and its job, copy-settings and clone pickers, model sharing's consumer list).
LISTING_STATES = frozenset(
    {
        "projects",
        "combine-wizard",
        "combine-job",
        "projects-delete-dry-run",
        "projects-copy-settings",
        "models-sharing",
        "models-unshare-force",
        "clone-from-project",
    }
)


def parse_allowlist(raw: str | None) -> set[str]:
    slugs = {s.strip() for s in (raw or "").split(",") if s.strip()}
    if not slugs:
        raise ValueError(
            "a public-project allowlist is required: --public-projects slug[,slug...] "
            "(or OP_CROPWRIGHT_PUBLIC_PROJECTS), naming every project on the backend that is public"
        )
    return slugs


def lists_all_projects(name: str) -> bool:
    return name in LISTING_STATES


def require_allowlisted_project(project: str | None, allowed: set[str]) -> None:
    if project is None:
        raise ValueError("--project is required: the capture only ever reads one named public project")
    if project not in allowed:
        raise ValueError(f"project {project!r} is not on the public allowlist {sorted(allowed)}")


def foreign_projects(listing: dict, allowed: set[str]) -> list[str]:
    """Slugs in a ``GET /curation/projects`` payload that the allowlist does not name."""
    return [p["slug"] for p in listing.get("projects", []) if p["slug"] not in allowed]


def listing_pages_to_refuse(names: list[str], listing: dict, allowed: set[str]) -> list[str]:
    """The pages among ``names`` that list all projects, when any listed project is foreign."""
    if not foreign_projects(listing, allowed):
        return []
    return [n for n in names if lists_all_projects(n)]
