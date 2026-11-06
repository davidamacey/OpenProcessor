"""Repositories: the only code that reads or writes a control-plane table.

Each function takes a caller-owned session inside the caller's ``project_tx`` or
``deployment_tx`` and never commits.
"""
