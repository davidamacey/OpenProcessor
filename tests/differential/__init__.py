"""The differential runner: the same generated requests to the reference stack and the Rust stack.

Before a Python route is deleted, its Rust replacement must answer every operation of its family the
way the Python one did (docs/design/v070_rebuild/18a_backend_rust_port_slices.md, section 3.2).

* ``families``   which route family owns each operation (``families.toml``)
* ``cases``      deterministic request generation from the committed OpenAPI files (Schemathesis)
* ``compare``    parsed-JSON comparison under an explicit, reviewed normalisation list (``normalise/``)
* ``providers``  per-family parameter values drawn from what both seeded stacks hold
* ``send``       one request to one stack; the schema-conformance check on the candidate's answer
* ``runner``     the loop, the JSON-lines results, the summary a sweep gates on (``python -m tests.differential.runner --help``)

What the slice that ports a family adds, before it deletes the family's Python handlers:

1. ``families.toml``: check the family's patterns (an operation may move to a family of its own; the test
   keeps the mapping total) and list its ``fixtures``.
2. ``normalise/<family>.toml``: every allowed difference as a rule with a ``reason`` (generated ids,
   timestamps, durations, the order of sets, the slice's intended changes), and the headers to compare
   beyond the defaults. It adds to ``normalise/_defaults.toml``; nothing is ignored that is not listed.
3. A fixtures provider in ``providers/`` for the ids the family's routes take, copied from
   ``providers/projects.py``: ask each stack's own list route, keep what both hold, register it in
   ``providers/__init__.py``. Both stacks must be seeded through their HTTP routes (the seeder of
   ``scripts/curation/seed_http/``) so that the provider finds the same values.
4. The deleted Python tests' rows in ``tests/parity/ledger/<slice>.tsv``, and the family's ids out of
   ``tests/pending_rust.txt``.
5. The run: ``--families <family> --check-schema`` against both stacks, an empty diff.
"""
