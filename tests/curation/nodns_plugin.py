"""pytest plugin for the DNS-failing run of the selection-path tests: makes
``socket.getaddrinfo`` raise (NODNS=gaierror) or return nothing (NODNS=empty)."""

import os
import socket


mode = os.environ.get('NODNS', 'gaierror')


def _fail(*a, **k):
    if mode == 'empty':
        return []
    raise socket.gaierror(-2, 'Name or service not known')


socket.getaddrinfo = _fail
