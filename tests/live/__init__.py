"""Live write-path verification suite.

Every test here is marked ``live`` and is deselected by the default
``addopts`` (``-m 'not live'``). They only pass against the isolated
harness in ``docker/test/compose.yml``; see that directory's README.
"""
