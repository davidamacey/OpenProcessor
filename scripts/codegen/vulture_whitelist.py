"""Reviewed false positives for ``check_vulture.py``.

A name a framework or a config file calls by string is invisible to vulture. Add it
here as a bare expression with the reason on the same line, for example:

    from src.services.some_module import dispatched_by_name

    dispatched_by_name  # looked up by name from examples/prompt_packs
"""
