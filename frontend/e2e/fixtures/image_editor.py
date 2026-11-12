"""A stateful stub of the image editor's backend: one image, its items and boxes, the list it
sits in, and every write the editor makes. Shared by the editor's own tests and the sweeps."""

from __future__ import annotations

from typing import Any

from fixtures.contract import box_view, fill, image_detail
from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS, PLACEHOLDER_PHOTO


EDITOR = '/p/default/images/img-1?view=gallery'

CLASSES = [
    {
        'class_id': i,
        'class_name': name,
        'kind': 'item',
        'group': 'g',
        'hotkey_letter': hotkey,
        'sample_count': 10 - i,
        'validated_count': 1,
        'deprecated': False,
    }
    for i, (name, hotkey) in enumerate([('car', 'k'), ('bus', 'j')])
]

LIST = ['img-0', 'img-1', 'img-2', 'img-3']


def locate(request: Any, _match: Any) -> dict:
    """Where the asked image sits in a four-image list, with the cursors around it."""
    target = request.url.split('target_image_id=')[1].split('&')[0]
    at = LIST.index(target)
    return {
        'image_id': target,
        'in_view': True,
        'index': at + 4,
        'total': 20,
        'cursor_before': f'before-{target}',
        'cursor_after': f'after-{target}',
        'previous_image_id': LIST[at - 1] if at else None,
        'reason': None,
        'reason_message': None,
    }


def image_page(request: Any, _match: Any) -> dict:
    """The page of images after the cursor `after-<id>`."""
    cursor = request.url.split('cursor=')[1].split('&')[0]
    rest = LIST[LIST.index(cursor.removeprefix('after-')) + 1 :]
    return {'items': [{'image_id': i} for i in rest], 'next_cursor': None}


class Editor:
    """A stateful stub of one image: its items, its boxes and every write the editor makes."""

    def __init__(self, stub: Any) -> None:
        self.stub = stub
        self.classes = list(CLASSES)
        self.items: dict[str, dict] = {}
        self.boxes: dict[str, dict] = {}
        self.fail_geometry: str | None = None
        self.add_item('c1', 'bus', [0.1, 0.1, 0.3, 0.3])
        self.add_item('c2', 'car', [0.5, 0.5, 0.8, 0.8])
        stub.on('GET', r'(?<!/stats)/classes(\?|$)', lambda _r, _m: {'classes': self.classes})
        stub.on('GET', r'/images/img-\d$', self.detail)
        stub.on('GET', r'/images/img-\d/file(\?|$)', (200, PLACEHOLDER_PHOTO, 'image/svg+xml'))
        stub.on('GET', r'/images/locate(\?|$)', locate)
        stub.on('GET', r'/images(\?|$)', image_page)
        stub.on('POST', r'/images/img-1/items$', self.create)
        stub.on('PATCH', r'/crops/[^/]+/geometry$', self.geometry)
        stub.on('POST', r'/crops/batch_exclude$', self.exclude)
        stub.on('POST', r'/crops/batch_unexclude$', self.unexclude)
        stub.on('POST', r'/classes$', self.create_class)
        stub.on('PUT', r'/crops/[^/]+/label$', self.label)

    def add_item(self, crop_id: str, class_name: str, box: list[float]) -> None:
        self.items[crop_id] = make_item(
            crop_id=crop_id,
            image_id='img-1',
            class_name=class_name,
            bbox_norm=box,
            geometry_revision=0,
            geometry_locked=False,
            lock_label=None,
            class_excluded=False,
        )
        self.boxes[crop_id] = box_view(crop_id, class_name, bbox_norm=box)

    def detail(self, _request: Any, _match: Any) -> Any:
        return image_detail(
            'img-1',
            list(self.items.values()),
            list(self.boxes.values()),
            width=1000,
            height=750,
        )

    def create(self, request: Any, _match: Any) -> Any:
        body = request.post_data_json
        crop_id = f'c{len(self.items) + 1}'
        self.add_item(crop_id, body['class_name'], body['bbox_norm'])
        return (
            201,
            {
                'item': self.items[crop_id],
                'created': True,
                'image_boxes': list(self.boxes.values()),
                'image_counts': fill('ImageCounts'),
                'embedding_state': 'embedded',
                'embedding_message': None,
            },
        )

    def geometry(self, request: Any, match: Any) -> Any:
        if self.fail_geometry:
            # The server's box wins: another editor moved it first.
            crop_id = match.string.split('/crops/')[1].split('/')[0]
            self.boxes[crop_id]['bbox_norm'] = [0.4, 0.4, 0.6, 0.6]
            return (409, {'detail': self.fail_geometry})
        return (200, {'item': {}, 'image_boxes': [], 'embedding_state': 'embedded'})

    def exclude(self, request: Any, _match: Any) -> Any:
        for crop_id in request.post_data_json['crop_ids']:
            self.items[crop_id]['class_excluded'] = True
            self.boxes[crop_id]['tone'] = 'rejected'
        return {'excluded': 1, 'updated_ids': request.post_data_json['crop_ids'], 'errors': 0}

    def unexclude(self, request: Any, _match: Any) -> Any:
        for crop_id in request.post_data_json['crop_ids']:
            self.items[crop_id]['class_excluded'] = False
            self.boxes[crop_id]['tone'] = 'validated'
        return {'unexcluded': 1, 'updated_ids': request.post_data_json['crop_ids'], 'errors': 0}

    def label(self, request: Any, match: Any) -> Any:
        crop_id = match.string.split('/crops/')[1].split('/')[0]
        name = next(
            c['class_name'] for c in CLASSES if c['class_id'] == request.post_data_json['class_id']
        )
        self.items[crop_id]['class_name'] = name
        self.boxes[crop_id]['label'] = name
        return self.items[crop_id]

    def create_class(self, request: Any, _match: Any) -> Any:
        name = request.post_data_json['name']
        if name != name.lower().replace(' ', '_'):
            return (
                422,
                {
                    'detail': {
                        'error': 'invalid_class_name',
                        'message': 'Class names are lowercase with underscores.',
                        'suggested_name': name.lower().replace(' ', '_'),
                    }
                },
            )
        self.classes.append({**CLASSES[0], 'class_id': 9, 'class_name': name, 'hotkey_letter': ''})
        return (201, {'class_id': 9, 'class_name': name})

    def writes(self, method: str, suffix: str) -> list[Any]:
        return [b for m, p, b in self.stub.calls if m == method and p.endswith(suffix)]

    def open(self, page: Any, app_url: str, url: str = EDITOR) -> None:
        page.goto(f'{app_url}{url}')
        page.get_by_test_id('editor-canvas').wait_for(timeout=ACTION_TIMEOUT_MS)
        page.locator('[data-testid="overlay-box"]').first.wait_for(timeout=ACTION_TIMEOUT_MS)
