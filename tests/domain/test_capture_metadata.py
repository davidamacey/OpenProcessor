"""parse_capture: one case per parse rule (research spatial plan 5.1, CM-1).

The input is the dictionary ExifTool prints for ``-json -n -G`` (numbers for numeric tags, signed
decimal GPS in ``Composite:``), so these cases are the shapes the real binary produces; the
real-binary check lives in ``tests/curation/test_capture_exiftool.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.domain.capture_metadata import PARSER_VERSION, parse_capture, unknown_capture


CLEAN: dict[str, Any] = {
    'EXIF:DateTimeOriginal': '2024:03:05 14:22:11',
    'EXIF:OffsetTimeOriginal': '+01:00',
    'EXIF:SubSecTimeOriginal': 50,
    'EXIF:Make': ' Acme ',
    'EXIF:Model': 'Cam1',
    'EXIF:Orientation': 6,
    'EXIF:FocalLength': 35,
    'EXIF:FocalLengthIn35mmFormat': 52,
    'Composite:GPSLatitude': -47.5,
    'Composite:GPSLongitude': 8.25,
    'Composite:GPSAltitude': 410,
    'EXIF:GPSHPositioningError': 4,
    'EXIF:GPSImgDirection': 90.5,
}


def test_a_clean_header_parses() -> None:
    cap = parse_capture(CLEAN)
    assert cap.exif_status == 'ok'
    assert cap.parser == PARSER_VERSION
    assert cap.taken_local == '2024-03-05T14:22:11.50'
    assert cap.taken_utc == '2024-03-05T13:22:11.50'
    assert (cap.offset_minutes, cap.offset_source) == (60, 'exif_offset')
    assert (cap.time_source, cap.time_precision) == ('exif_original', 'subsecond')
    assert (cap.camera_make, cap.camera_model, cap.orientation) == ('Acme', 'Cam1', 6)
    assert (cap.focal_length_mm, cap.focal_length_35mm) == (35.0, 52.0)
    assert cap.has_location
    assert cap.location is not None
    assert (cap.location.latitude, cap.location.longitude) == (-47.5, 8.25)
    assert (cap.location.altitude_m, cap.location.accuracy_m) == (410.0, 4.0)
    assert cap.location.heading_deg == 90.5


def test_an_empty_dict_is_absent_not_an_error() -> None:
    cap = parse_capture({'SourceFile': 'x.jpg'})
    assert cap.exif_status == 'absent'
    assert cap.taken_local is None
    assert cap.location is None


def test_modify_date_is_never_capture_time() -> None:
    cap = parse_capture({'EXIF:ModifyDate': '2025:01:01 00:00:00', 'EXIF:Make': 'Acme'})
    assert (cap.taken_local, cap.taken_utc, cap.time_source) == (None, None, 'none')
    assert cap.exif_status == 'ok'


def test_digitized_time_is_the_fallback_with_its_own_offset() -> None:
    cap = parse_capture(
        {'EXIF:CreateDate': '2024:03:05 10:00:00', 'EXIF:OffsetTimeDigitized': '-05:00'}
    )
    assert cap.time_source == 'exif_digitized'
    assert (cap.taken_local, cap.taken_utc) == ('2024-03-05T10:00:00', '2024-03-05T15:00:00')


def test_the_offset_is_not_taken_from_the_digitized_tag_for_the_original_time() -> None:
    cap = parse_capture(
        {'EXIF:DateTimeOriginal': '2024:03:05 10:00:00', 'EXIF:OffsetTimeDigitized': '+09:00'}
    )
    assert (cap.offset_minutes, cap.taken_utc, cap.offset_source) == (None, None, 'none')


def test_the_offset_comes_from_the_gps_utc_delta_rounded_to_15_minutes() -> None:
    cap = parse_capture(
        {
            'EXIF:DateTimeOriginal': '2024:03:05 14:22:11',
            'EXIF:GPSDateStamp': '2024:03:05',
            'EXIF:GPSTimeStamp': '09:00:30',
        }
    )
    assert (cap.offset_minutes, cap.offset_source) == (315, 'gps_utc_delta')
    assert cap.taken_utc == '2024-03-05T09:07:11'


def test_a_gps_delta_outside_the_zone_range_is_no_offset() -> None:
    cap = parse_capture(
        {
            'EXIF:DateTimeOriginal': '2024:03:05 14:22:11',
            'EXIF:GPSDateStamp': '2024:03:04',
            'EXIF:GPSTimeStamp': '01:00:00',
        }
    )
    assert (cap.offset_minutes, cap.taken_utc, cap.offset_source) == (None, None, 'none')
    assert cap.taken_local == '2024-03-05T14:22:11'


@pytest.mark.parametrize('offset', ['+15:00', '-12:30', 'Z', '1:00', '+01:75', 3600, None])
def test_an_out_of_range_or_garbled_offset_tag_is_no_offset(offset: Any) -> None:
    cap = parse_capture(
        {'EXIF:DateTimeOriginal': '2024:03:05 14:22:11', 'EXIF:OffsetTimeOriginal': offset}
    )
    assert (cap.offset_minutes, cap.taken_utc) == (None, None)
    assert cap.taken_local == '2024-03-05T14:22:11'


def test_gps_time_alone_is_the_capture_instant_in_utc() -> None:
    cap = parse_capture({'EXIF:GPSDateStamp': '2024:03:05', 'EXIF:GPSTimeStamp': '13:22:09'})
    assert (cap.time_source, cap.taken_utc, cap.taken_local) == (
        'gps_utc',
        '2024-03-05T13:22:09',
        None,
    )


@pytest.mark.parametrize(
    'bad', ['0000:00:00 00:00:00', '2024:13:40 99:99:99', 'yesterday', 20240305, '', ' ']
)
def test_a_malformed_date_is_malformed_status_never_an_error(bad: Any) -> None:
    cap = parse_capture({'EXIF:DateTimeOriginal': bad, 'EXIF:Make': 'Acme'})
    assert cap.exif_status == 'malformed'
    assert (cap.taken_local, cap.taken_utc, cap.time_source) == (None, None, 'none')
    assert cap.camera_make == 'Acme'


@pytest.mark.parametrize(
    ('lat', 'lon'), [(0, 0), (91, 10), (-91, 10), (10, 181), (10, -181), ('x', 3), (None, None)]
)
def test_zero_zero_and_out_of_range_coordinates_are_no_location(lat: Any, lon: Any) -> None:
    cap = parse_capture({'Composite:GPSLatitude': lat, 'Composite:GPSLongitude': lon})
    assert cap.location is None
    assert not cap.has_location


def test_a_void_gps_fix_is_no_location() -> None:
    cap = parse_capture(
        {'Composite:GPSLatitude': 10.0, 'Composite:GPSLongitude': 20.0, 'EXIF:GPSStatus': 'V'}
    )
    assert cap.location is None


def test_a_garbled_optional_tag_is_none_for_that_field_only() -> None:
    cap = parse_capture(
        {
            **CLEAN,
            'EXIF:FocalLength': 'n/a',
            'EXIF:Orientation': 99,
            'EXIF:SubSecTimeOriginal': 'abc',
            'EXIF:GPSImgDirection': 400,
            'EXIF:GPSHPositioningError': -1,
        }
    )
    assert (cap.focal_length_mm, cap.orientation) == (None, None)
    assert cap.time_precision == 'second'
    assert cap.taken_local == '2024-03-05T14:22:11'
    assert cap.location is not None
    assert (cap.location.heading_deg, cap.location.accuracy_m) == (None, None)


def test_long_text_is_cut_and_control_characters_dropped() -> None:
    cap = parse_capture({'EXIF:Make': 'A' * 300, 'EXIF:Model': 'Cam\x00\x01'})
    assert cap.camera_make == 'A' * 128
    assert cap.camera_model == 'Cam'


def test_unrequested_tags_are_ignored() -> None:
    cap = parse_capture({'File:FileName': 'x.jpg', 'EXIF:ImageDescription': 'secret'})
    assert cap.exif_status == 'absent'


def test_an_outage_capture_has_no_parser_stamp() -> None:
    cap = unknown_capture()
    assert (cap.exif_status, cap.parser, cap.location) == ('unknown', None, None)


def test_a_file_exiftool_reports_an_error_for_is_malformed() -> None:
    assert parse_capture({'ExifTool:Error': 'File format error'}).exif_status == 'malformed'
    cap = parse_capture({'ExifTool:Error': 'File format error', 'EXIF:Make': 'Acme'})
    assert (cap.exif_status, cap.camera_make) == ('malformed', 'Acme')
