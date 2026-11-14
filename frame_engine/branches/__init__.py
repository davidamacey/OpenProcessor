"""The engine's branches: each cuts its model inputs from the one resident frame and calls a model.

Contract (``engine.py`` is the orchestrator; a branch is a :class:`~.base.Branch` subclass):

* **Hooks**, called for every branch of the plan in plan order, phase by phase:
  ``round1`` (calls that need only the frame; every round-1 call goes out together after ONE
  stream wait), ``settle`` (collect what other branches read and leave it on the
  :class:`~.base.Photo`: ``rects``, ``classes``), ``round2`` (calls cut from those answers; submit
  every call before collecting any, then start host work), ``finish`` (collect, write
  ``photo.out``), ``close`` (always; give up queued host work).
* **Calls** go through ``photo.calls`` (:class:`~.base.PhotoCalls`): ``stage(nbytes, stage)`` a
  slice (bounded by one call's batch, never by a per-photo count), launch the kernels into it,
  ``submit(self, stage, staged, model, ModelInput(...), outputs)`` returns at once,
  ``result(call)`` waits by the request deadline and returns ``{output name: array}``.
* **Failure** of one branch is its own: raise ``ModelCallError`` (``result`` does) or
  :class:`~.base.DependencyFailedError`; the orchestrator sets ``branch_status`` (``model_error``,
  ``dependency_failed``), cancels that branch's calls and skips its later hooks. Anything else
  fails the photo. ``status`` is the ``branch_status`` entry the branch owns (``ok`` when its
  hooks all returned).
* **Dependencies and conditions**: ``needs`` names the ``branch_status`` entries whose answers
  the branch reads; when one fails or does not run, the branch ends the same way without running.
  A branch the plan names in ``if_detected`` (and what needs it) runs late: its ``round1`` after
  the detector's ``settle`` and only when the detector found one of the classes, sent before the
  other branches' ``round2``; then its ``settle`` and ``round2``. Otherwise it is ``not_run``. The
  hooks are the same; the orchestrator schedules them.
* **Time**: wrap kernel launches and host work in ``photo.clock.stage(<stage name>)``; the calls
  clock themselves in the stage given to ``stage``/``submit``.

Worked example, a face-vector branch (round 2, needs face boxes from a ``faces`` branch)::

    class FaceEmbed(Branch):
        status = 'face_embed'

        def round2(self, photo):
            if photo.faces is None:
                raise DependencyFailedError('no faces')
            self.calls = []
            for chunk in batches(photo.faces, 64):
                x = ModelInput('input', 'FP32', (len(chunk), 3, 112, 112))
                staged = photo.calls.stage(x.nbytes, 'face_embed')
                with photo.clock.stage('face_embed'):
                    align_into(photo.frame, chunk, staged.device)  # GPU kernels
                self.calls.append(
                    photo.calls.submit(
                        self, 'face_embed', staged, 'arcface_w600k_r50', x, ('output',)
                    )
                )

        def finish(self, photo):
            rows = [photo.calls.result(c)['output'] for c in self.calls]
            photo.out[wire.FACE_VECTORS] = np.concatenate(rows)

Then add it to ``FrameEngine._plan_branches``, its stage names to ``stage_metrics.STAGES`` and its
outputs to the wire module.
"""
