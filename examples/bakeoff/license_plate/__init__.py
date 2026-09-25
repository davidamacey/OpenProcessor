"""Example bake-off profile: license-plate detection (reference configuration).

Where the harness came from. Opt-in, never a default: select it by path,
``--profile examples/bakeoff/license_plate/profile.json``. Ships:

* ``profile.json``  -- single ``license_plate`` class; coarse stage keeps the
  COCO vehicle classes (car=2, motorcycle=3, bus=5, truck=7) for
  vehicle->plate crop mode.
* ``baselines.json`` -- public plate-detector baselines for the UI picker.
* ``converters``    -- CCPD / UFPR-ALPR / OpenALPR public-benchmark formats.
* ``backends``      -- the ``lpdnet`` and ``open-image-models`` detector
  backends (plate-only public models), registered when the profile loads.
* ``requirements.txt`` -- the extra package the ``open-image-models``
  backend needs.
"""
