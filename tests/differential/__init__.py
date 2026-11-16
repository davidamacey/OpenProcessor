"""The differential runner: the same generated requests to the reference stack and the Rust stack.

Before a Python route is deleted, its Rust replacement must answer every operation of its family the
way the Python one did (docs/design/v070_rebuild/18a_backend_rust_port_slices.md, section 3.2).

* ``families``   which route family owns each operation (``families.toml``)
* ``cases``      deterministic request generation from the committed OpenAPI files (Schemathesis)
* ``compare``    parsed-JSON comparison under an explicit, reviewed normalisation list (``normalise/``)
* ``providers``  per-family parameter values drawn from what both seeded stacks hold
* ``runner``     sending, JSON-lines results, the summary a sweep gates on
"""
