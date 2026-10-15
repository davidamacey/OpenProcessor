"""Checks on a face-recognition response before its fields are paired up."""

from __future__ import annotations


def face_landmark_mismatch(face_result: dict) -> str | None:
    """The reason a face response is unusable, or ``None``.

    An empty landmark list means "not provided"; a non-empty one shorter than the
    faces it describes cannot be paired with them, so the image is refused rather
    than guessed at.
    """
    faces = min(len(face_result.get('face_embeddings', [])), len(face_result.get('face_boxes', [])))
    landmarks = len(face_result.get('face_landmarks', []))
    if 0 < landmarks < faces:
        return f'face_landmarks_length_mismatch: {landmarks} landmark entries for {faces} faces'
    return None
