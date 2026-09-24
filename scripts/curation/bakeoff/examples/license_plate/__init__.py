"""Example bake-off profile: license-plate detection (reference configuration).

Where the harness came from. Not a default -- select it explicitly with
``--profile license_plate``. Ships:

* ``profile.json``  -- single ``license_plate`` class; coarse stage keeps the
  COCO vehicle classes (car=2, motorcycle=3, bus=5, truck=7) for
  vehicle->plate crop mode.
* ``baselines.json`` -- public plate-detector baselines for the UI picker.
* ``converters``    -- CCPD / UFPR-ALPR / OpenALPR public-benchmark formats.
"""
