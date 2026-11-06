"""Repositories: the SQL of one concern each, over a caller-owned session.

A repository function never opens a transaction and never commits; the caller wraps it in
``project_tx`` or ``deployment_tx``. Only services import this package (the layering ratchet).
"""
