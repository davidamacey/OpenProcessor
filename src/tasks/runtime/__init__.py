"""The job runtime shared by every worker image.

A leaf layer: it imports only ``src.config``, ``src.domain``, ``src.db`` and ``src.utils``, so the
trainer, export and evaluator images can use it without the API's services.
"""
