"""Which modules may open a deployment-wide transaction.

``project_tx`` needs no permission: it narrows access. ``deployment_tx`` sees every project's
rows, so only modules that must are listed. Entries are ``fnmatch`` patterns on the importing
module's name. A module that needs a new entry adds one line here with its reason; a test module under ``tests/`` is always
allowed because it is never imported by production code.
"""

from __future__ import annotations


DEPLOYMENT_TX_MODULES: tuple[str, ...] = (
    'src.tasks.reaper',  # stuck-job recovery runs across all projects
    'src.tasks.periodic',  # beat tasks (cache prune, retention, ingest dispatch)
    'src.tasks.beat_schedule',
    'src.tasks.runtime',  # the job wrapper claims a job by id before it knows the project
    'src.services.jobs',  # job start and status across scopes, reaper rules
    'src.services.projects',  # the project registry, create, patch, delete and its job
    'src.services.upgrade',  # the 0.6.0 control-data step
    'scripts.db',  # migration runner and db verify
    'scripts.admin',  # admin command-line tools
)
