"""The ``source`` documents of ``contracts/json/frame_engine_source_cases.json`` (plan 17, N2).

The native engine reads a photo by key when the request carries ``source`` instead of ``image``
(``read_by_key`` in ``frame_engine_v1.json``). The Python engine does not have the input, so no
Python function defines the answer: this table does. Each case is the exact text of the input and
the answer both Rust sides must give: the engine's parser (the parsed key, size and hash, or the
fault and the status details) and the client's builder (``built``: the text it writes for a key,
a size and a hash, the way ``json.dumps(..., sort_keys=True)`` writes it).

The checks run in this order, and the first that fails answers: the text is JSON with no repeated
key and is an object (``bad_source_json``); a field other than the three is present, a field is
missing or has the wrong type (``bad_source_field``, the field named); the key breaks the key
rule (``bad_key``); its root is not the landing zone or a source root (``unknown_root``); the size
is not an integer from 0 to 2**53 - 1 or the hash is not 32 lowercase hex characters
(``bad_claim``, the field named). The status details are ``<fault>: <repr of the subject>``
(``bad_source_json`` carries no subject: the text of a malformed document is the JSON decoder's).
"""

from __future__ import annotations

import json
from typing import Any


SHA = '0123456789abcdef' * 4
UPLOAD = '00000000-0000-4000-8000-000000000abc'
HASH = '8e0124d8a93f5a1b7c6d2e3f40516273'
LANDING = f'landing/7/{SHA}.{UPLOAD}.3'
MAX_SIZE = 2**53 - 1

#: Faults in the order the checks run.
FAULTS = ['bad_source_json', 'bad_source_field', 'bad_key', 'unknown_root', 'bad_claim']


def _doc(**fields: Any) -> str:
    return json.dumps(fields, sort_keys=True)


def _ok(case_id: str, key: str, size: int, imohash: str = HASH, *, text: str | None = None) -> dict:
    document = text if text is not None else _doc(key=key, size=size, imohash=imohash)
    return {'id': case_id, 'text': document, 'key': key, 'size': size, 'imohash': imohash}


def _fault(case_id: str, text: str, fault: str, subject: str | None = None) -> dict:
    case: dict[str, Any] = {'id': case_id, 'text': text, 'fault': fault}
    if subject is not None:
        case['subject'] = subject
    return case


def _obj(*members: str) -> str:
    """A JSON object from its members, each already ``"name": value`` text."""
    return '{' + ', '.join(members) + '}'


def _valid() -> list[dict]:
    key = '"key": "source/café/é 1.jpg"'
    return [
        _ok('landing_key', LANDING, 6_812_345),
        _ok('source_key', 'source/trip/IMG_1.jpg', 6_812_345),
        _ok('empty_file', 'source/empty.jpg', 0),
        _ok('largest_size', 'source/big.jpg', MAX_SIZE),
        _ok('non_ascii_key_escaped', 'source/café/é 1.jpg', 1234),
        _ok(
            'non_ascii_key_literal',
            'source/café/é 1.jpg',
            1234,
            text=_obj(key, '"size": 1234', f'"imohash": "{HASH}"'),
        ),
        _ok(
            'fields_in_any_order',
            'source/a.jpg',
            7,
            text=_obj('"size": 7', f'"imohash": "{HASH}"', '"key": "source/a.jpg"'),
        ),
        _ok(
            'whitespace_around',
            'source/a.jpg',
            7,
            text=f' \n{{ "key" : "source/a.jpg" , "size" : 7 , "imohash" : "{HASH}" }}\n',
        ),
        _ok('hash_with_digits_only', 'source/a.jpg', 7, '0' * 32),
    ]


def _document_faults() -> list[dict]:
    key, size = '"key": "source/a.jpg"', '"size": 7'
    imohash = f'"imohash": "{HASH}"'
    field = 'bad_source_field'
    return [
        _fault('empty_text', '', 'bad_source_json'),
        _fault('truncated_json', '{"key": "source/a.jpg"', 'bad_source_json'),
        _fault('not_an_object', '["source/a.jpg", 7]', 'bad_source_json'),
        _fault('a_string', '"source/a.jpg"', 'bad_source_json'),
        _fault('repeated_key', _obj(key, size, imohash, '"size": 8'), 'bad_source_json'),
        _fault('nan_literal', _obj(key, '"size": NaN', imohash), 'bad_source_json'),
        _fault('missing_key', _obj(size, imohash), field, "'key'"),
        _fault('missing_size', _obj(key, imohash), field, "'size'"),
        _fault('missing_imohash', _obj(key, size), field, "'imohash'"),
        _fault('unknown_field', _obj(key, size, imohash, '"mtime": 5'), field, "'mtime'"),
        _fault('unknown_field_before_a_missing_one', _obj(key, '"mtime": 5'), field, "'mtime'"),
        _fault('key_is_a_number', _obj('"key": 5', size, imohash), field, "'key'"),
        _fault('size_is_a_string', _obj(key, '"size": "7"', imohash), field, "'size'"),
        _fault('size_is_a_float', _obj(key, '"size": 7.0', imohash), field, "'size'"),
        _fault('size_is_a_bool', _obj(key, '"size": true', imohash), field, "'size'"),
        _fault('imohash_is_null', _obj(key, size, '"imohash": null'), field, "'imohash'"),
        _fault('missing_before_wrong_type', _obj('"key": 5', size), field, "'imohash'"),
        _fault(
            'wrong_types_in_field_order', _obj('"imohash": 5', '"key": 5', size), field, "'key'"
        ),
        _fault('imohash_is_a_number', _obj(key, size, '"imohash": 5'), field, "'imohash'"),
    ]


def _key_faults() -> list[dict]:
    def bad(case_id: str, key: str) -> dict:
        return _fault(case_id, _doc(key=key, size=7, imohash=HASH), 'bad_key', repr(key))

    def root(case_id: str, key: str) -> dict:
        return _fault(case_id, _doc(key=key, size=7, imohash=HASH), 'unknown_root', repr(key))

    return [
        bad('empty_key', ''),
        bad('root_only', 'source'),
        bad('root_and_slash', 'source/'),
        bad('leading_slash', '/source/a.jpg'),
        bad('trailing_slash', 'source/a/'),
        bad('empty_segment', 'source//a.jpg'),
        bad('dot_dot_segment', 'source/../a.jpg'),
        bad('dot_segment', 'source/./a.jpg'),
        bad('landing_traversal', 'landing/7/../../etc/passwd'),
        bad('backslash', 'source/a\\b.jpg'),
        bad('nul_byte', 'source/a\u0000b.jpg'),
        bad('newline', 'source/a\nb.jpg'),
        bad('delete_character', 'source/a\u007fb.jpg'),
        root('unknown_root', 'upload/a.jpg'),
        root('miscased_root', 'Source/a.jpg'),
        root('object_store_root', 'projects/1/originals/ab/' + SHA),
        root('models_root', 'models/a.onnx'),
        root('cache_root', 'cache/a'),
        # the key rule is checked before the root, the root before the claim
        _fault(
            'bad_key_before_bad_claim',
            _doc(key='source/../a', size=-1, imohash='x'),
            'bad_key',
            repr('source/../a'),
        ),
        _fault(
            'unknown_root_before_bad_claim',
            _doc(key='projects/1/a', size=-1, imohash='x'),
            'unknown_root',
            repr('projects/1/a'),
        ),
    ]


def _claim_faults() -> list[dict]:
    def bad(case_id: str, field: str, **fields: Any) -> dict:
        body = {'key': 'source/a.jpg', 'size': 7, 'imohash': HASH, **fields}
        return _fault(case_id, json.dumps(body, sort_keys=True), 'bad_claim', repr(field))

    return [
        bad('negative_size', 'size', size=-1),
        bad('size_past_the_json_safe_range', 'size', size=MAX_SIZE + 1),
        bad('size_past_sixty_four_bits', 'size', size=2**64),
        bad('hash_too_short', 'imohash', imohash=HASH[:-1]),
        bad('hash_too_long', 'imohash', imohash=HASH + '0'),
        bad('hash_uppercase', 'imohash', imohash=HASH.upper()),
        bad('hash_not_hex', 'imohash', imohash='g' * 32),
        bad('hash_empty', 'imohash', imohash=''),
        _fault(
            'size_checked_before_hash',
            _doc(key='source/a.jpg', size=-1, imohash='x'),
            'bad_claim',
            "'size'",
        ),
    ]


def source_cases() -> list[dict]:
    """Every case, in a fixed order; ids are unique."""
    cases = [*_valid(), *_document_faults(), *_key_faults(), *_claim_faults()]
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate case id')
    return cases


def message(case: dict) -> str | None:
    """The status details of a refused case; ``None`` for ``bad_source_json``."""
    if case['fault'] == 'bad_source_json':
        return None
    return f'{case["fault"]}: {case["subject"]}'
