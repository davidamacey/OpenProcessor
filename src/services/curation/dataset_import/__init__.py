"""Dataset import (W10): bring an already-labeled dataset (YOLO or COCO
format) into a project, mapping its classes onto the registry by name
(never by index — see the class-identity invariant in
``docs/design/openprocessor_internal/any_domain_plan.md``).

Submodules:

* ``scan.py`` — directory scanning, format detection.
* ``yolo.py`` / ``coco.py`` — format readers, each emitting ``ScanEntry``.
* ``mapping.py`` — class-name mapping (exported for P4's combine-projects
  wave to reuse).
* ``regions.py`` — attach region-class label boxes to parent items.
* ``issues.py`` — the served issue-code catalog.
"""

from __future__ import annotations
