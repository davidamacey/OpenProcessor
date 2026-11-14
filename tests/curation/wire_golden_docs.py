"""1,000 deterministic, varied stored item documents for the serializer golden test."""

from __future__ import annotations

import random
from typing import Any


def _box(rng: random.Random, n: int) -> dict[str, Any]:
    x1, y1 = rng.uniform(0, 0.5), rng.uniform(0, 0.5)
    box: dict[str, Any] = {
        'box_id': f'b{n}',
        'bbox_norm': [x1, y1, x1 + rng.uniform(0.05, 0.4), y1 + rng.uniform(0.05, 0.4)],
        'state': rng.choice(['proposed', 'accepted', 'rejected', 'false_positive']),
        'score': round(rng.random(), 4),
        'detector': rng.choice(['yolo', 'segmenter', 'human']),
        'source': rng.choice(['detector', 'human', 'import', None]),
    }
    if rng.random() < 0.5:
        box.update(text=f'AB{rng.randint(100, 999)}', text_raw='ab', text_norm='AB', text_fold='AB')
        box['text_confidence'] = round(rng.random(), 3)
    if rng.random() < 0.3:
        box.update(cluster_partition_id='p1', cluster_uid=f'u{rng.randint(1, 50)}')
        box['cluster_score'] = round(rng.random(), 3)
    return box


def golden_docs(count: int = 1000) -> list[tuple[str, dict[str, Any]]]:
    rng = random.Random(1113)
    out: list[tuple[str, dict[str, Any]]] = []
    for i in range(count):
        x1, y1 = rng.uniform(0, 0.5), rng.uniform(0, 0.5)
        doc: dict[str, Any] = {
            'crop_id': f'crop-{i:05d}',
            'image_id': f'img-{i // 3:05d}',
            'image_path': f'/data/set/{i // 3:05d}.jpg',
            'bbox_norm': [x1, y1, x1 + 0.3, y1 + 0.4],
            'detector_class_name': rng.choice(['car', 'bus', 'truck', 'person']),
            'detector_class_id': rng.randint(0, 79),
            'detector_confidence': round(rng.random(), 4),
            'confidence': round(rng.random(), 4),
            'crop_rank_in_image': rng.randint(1, 6),
            'crop_area_norm': round(rng.random(), 4),
            'blur_lap_ratio': round(rng.random() * 2, 4),
            'updated_at': f'2026-11-{rng.randint(1, 12):02d}T10:00:00Z',
            'geometry_revision': rng.randint(0, 3),
            'import_ids': ['i1'] if rng.random() < 0.1 else [],
            'embedding_state_a': rng.choice(['embedded', 'failed', None]),
        }
        roll = rng.random()
        if roll < 0.35:
            doc.update(
                class_id=rng.randint(0, 9),
                class_name=rng.choice(['sedan', 'van', 'bus']),
                class_source='human',
                class_validated=True,
                label_source='human',
                class_labeled_at='2026-11-01T00:00:00Z',
                class_labeler='dave',
            )
        elif roll < 0.7:
            doc.update(
                class_id=rng.randint(0, 9),
                class_name=rng.choice(['sedan', 'van', 'bus']),
                class_source='vlm',
                label_source='vlm',
                vlm_confidence=rng.choice(['high', 'medium', 'low']),
                vlm_prompt_pack='cars@2',
                vlm_endpoint='local',
                vlm_model='m',
                vlm_raw_class=rng.choice(['sedan', 'estate', None]),
            )
        if rng.random() < 0.2:
            doc.update(
                probe_pred_class=rng.choice(['sedan', 'van']),
                probe_disagreement=rng.choice([True, False, None]),
                probe_pred_confidence=round(rng.random(), 3),
                probe_model_version='p1',
            )
        if rng.random() < 0.3:
            doc.update(
                cluster_partition_id='p1',
                cluster_uid=f'u{rng.randint(1, 80)}',
                cluster_state='assigned',
                cluster_score=round(rng.random(), 3),
            )
        if rng.random() < 0.1:
            doc.update(test_holdout=True)
        if rng.random() < 0.05:
            doc.update(class_excluded=True, excluded_reason='not_an_object')
        n_boxes = rng.choice([0, 1, 1, 2, 3])
        if n_boxes:
            doc['region_boxes'] = [_box(rng, n) for n in range(1, n_boxes + 1)]
            doc['region_count'] = n_boxes
            doc['region_revision'] = rng.randint(0, 4)
        out.append((f'crop-{i:05d}', doc))
    return out
