#!/usr/bin/env python3
"""Cases for the Rust capture mapping (plan 18a, slice R1x, module ``capture``).

Writes ``contracts/json/capture_cases.json``: tag sets (the dictionary one ExifTool read returns,
``-j -G -n``) and what ``src.domain.capture_metadata.parse_capture`` and
``src.domain.location_policy.apply_location_policy`` answered. The expectations are never typed
here: the Python functions are the definition and ``native/op_media`` is held to them case by case.

The tag sets come from two places:

* ``HAND_CASES`` below: missing tags, clocks with and without an offset or sub-seconds, GPS
  references and range limits, orientation 1 to 8 and values around them, focal lengths, texts with
  control characters, very long strings and numbers where text is expected;
* ``native/op_media/tests/fixtures/exiftool/recorded.json``: outputs of the real ExifTool over the
  committed test images and over copies of one of them that ExifTool itself tagged with made-up
  values (nothing private). ``--record`` writes that file (it needs ``exiftool`` on the PATH); the
  normal run and ``--check`` only read it, so they need no ExifTool.

Two Python behaviours the cases leave out because Python raises: a capture time within a day of
year 1 or 9999 whose UTC time falls outside the years 1 to 9999 (``OverflowError``), and a string
with a lone surrogate (not valid JSON text for the Rust reader).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ERROR_KEY = 'ExifTool:Error'


def _domain() -> tuple[Any, Any]:
    """The two Python modules under test, imported once the repository root is importable."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.domain import capture_metadata, location_policy

    return capture_metadata, location_policy


OUT = ROOT / 'contracts' / 'json' / 'capture_cases.json'
RECORDED = ROOT / 'native' / 'op_media' / 'tests' / 'fixtures' / 'exiftool' / 'recorded.json'

OT = 'EXIF:DateTimeOriginal'
OFF = 'EXIF:OffsetTimeOriginal'
SUB = 'EXIF:SubSecTimeOriginal'
CD = 'EXIF:CreateDate'
GPSD = 'EXIF:GPSDateStamp'
GPST = 'EXIF:GPSTimeStamp'
LAT = 'Composite:GPSLatitude'
LON = 'Composite:GPSLongitude'
ALT = 'Composite:GPSAltitude'
STATUS = 'EXIF:GPSStatus'
ACC = 'EXIF:GPSHPositioningError'
DIR = 'EXIF:GPSImgDirection'
MAKE = 'EXIF:Make'
MODEL = 'EXIF:Model'
LENS = 'EXIF:LensModel'
FL = 'EXIF:FocalLength'
FL35 = 'EXIF:FocalLengthIn35mmFormat'
ORI = 'EXIF:Orientation'

BASE_TIME = {OT: '2024:03:05 14:22:11'}
FIX = {LAT: 47.5, LON: 8.25}
ARABIC_2024 = '\u0662\u0660\u0662\u0664'  # 2024 in Arabic-Indic digits
LONG = 'a' * 200


def _time_cases() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = [
        (
            'time_original_with_offset_and_subsec',
            {OT: '2024:03:05 14:22:11', OFF: '+01:00', SUB: '50'},
        ),
        ('time_original_no_offset', BASE_TIME),
        ('time_original_t_separator', {OT: '2024:03:05T14:22:11'}),
        ('time_original_padded_with_spaces', {OT: '  2024:03:05 14:22:11  ', OFF: ' -05:30 '}),
        ('time_createdate_only', {CD: '2023:12:31 23:59:59', 'EXIF:OffsetTimeDigitized': '+14:00'}),
        (
            'time_createdate_digitized_subsec',
            {CD: '2023:12:31 23:59:59', 'EXIF:SubSecTimeDigitized': '123'},
        ),
        (
            'time_original_wins_over_createdate',
            {OT: '2024:03:05 14:22:11', CD: '2020:01:01 00:00:00'},
        ),
        ('time_malformed_original_stops_at_createdate', {OT: 'garbage', CD: '2020:01:01 00:00:00'}),
        ('time_original_null', {OT: None, CD: '2020:01:01 00:00:00'}),
        ('time_original_number', {OT: 20240305}),
        ('time_dashes', {OT: '2024-03-05 14:22:11'}),
        ('time_missing_seconds', {OT: '2024:03:05 14:22'}),
        ('time_trailing_text', {OT: '2024:03:05 14:22:11 DST'}),
        ('time_month_13', {OT: '2024:13:05 14:22:11'}),
        ('time_feb_30', {OT: '2024:02:30 14:22:11'}),
        ('time_feb_29_leap', {OT: '2024:02:29 14:22:11'}),
        ('time_feb_29_not_leap', {OT: '2023:02:29 14:22:11'}),
        ('time_feb_29_1900', {OT: '1900:02:29 14:22:11'}),
        ('time_feb_29_2000', {OT: '2000:02:29 14:22:11'}),
        ('time_hour_24', {OT: '2024:03:05 24:00:00'}),
        ('time_second_60', {OT: '2024:03:05 14:22:60'}),
        ('time_year_0000', {OT: '0000:01:01 00:00:00'}),
        ('time_year_0999_unpadded', {OT: '0999:06:15 10:00:00'}),
        ('time_year_0005_unpadded', {OT: '0005:06:15 10:00:00'}),
        ('time_year_9999', {OT: '9999:12:31 23:59:59'}),
        ('time_year_0001', {OT: '0001:01:01 00:00:00'}),
        ('time_year_1970_epoch', {OT: '1970:01:01 00:00:00', OFF: '+00:00'}),
        ('time_utc_crosses_midnight_back', {OT: '2024:03:01 00:30:00', OFF: '+02:00'}),
        ('time_utc_crosses_midnight_forward', {OT: '2024:02:28 23:30:00', OFF: '-02:00'}),
        ('time_utc_crosses_leap_day', {OT: '2024:02:29 23:30:00', OFF: '-03:00'}),
        ('time_utc_crosses_year', {OT: '2023:12:31 23:59:59', OFF: '-12:00'}),
        ('time_arabic_indic_digits', {OT: f'{ARABIC_2024}:03:05 14:22:11'}),
        ('time_fullwidth_digits', {OT: '\uff12\uff10\uff12\uff14:03:05 14:22:11'}),
        ('time_mixed_script_digits', {OT: '2024:03:05 14:22:11', SUB: '\u0661\u0662\u0663'}),
    ]
    for label, offset in [
        ('plus_14', '+14:00'), ('minus_12', '-12:00'), ('plus_14_01', '+14:01'),
        ('minus_12_01', '-12:01'), ('plus_05_45', '+05:45'), ('minutes_60', '+05:60'),
        ('hours_25', '+25:00'), ('no_colon', '+0100'), ('z', 'Z'), ('empty', ''),
        ('no_sign', '01:00'), ('plus_0', '+00:00'), ('minus_0', '-00:00'),
        ('number', 1), ('null', None), ('seconds', '+01:00:00'),
    ]:  # fmt: skip
        cases.append((f'offset_{label}', {**BASE_TIME, OFF: offset}))
    for label, subsec in [
        ('one_digit', '5'), ('zero', '0'), ('six', '123456'), ('seven', '1234567'),
        ('nine', '123456789'), ('ten', '1234567890'), ('empty', ''), ('letters', '12a'),
        ('padded', ' 42 '), ('int', 42), ('int_big', 1234567890), ('float', 4.5),
        ('negative_int', -5), ('bool', True), ('null', None), ('leading_zero_int_text', '007'),
    ]:  # fmt: skip
        cases.append((f'subsec_{label}', {**BASE_TIME, SUB: subsec}))
    return cases


def _gps_clock_cases() -> list[tuple[str, dict[str, Any]]]:
    local = {OT: '2024:03:05 14:22:11'}
    cases: list[tuple[str, dict[str, Any]]] = [
        ('gps_only_clock', {GPSD: '2024:03:05', GPST: '13:22:11'}),
        ('gps_only_clock_fraction', {GPSD: '2024:03:05', GPST: '13:22:11.987'}),
        ('gps_clock_fraction_empty', {GPSD: '2024:03:05', GPST: '13:22:11.'}),
        ('gps_clock_missing_date', {GPST: '13:22:11'}),
        ('gps_clock_missing_time', {GPSD: '2024:03:05'}),
        ('gps_clock_bad_date', {GPSD: '2024:13:05', GPST: '13:22:11'}),
        ('gps_clock_bad_time', {GPSD: '2024:03:05', GPST: '25:22:11'}),
        ('gps_clock_numbers', {GPSD: 20240305, GPST: 132211}),
        ('gps_clock_padded', {GPSD: ' 2024:03:05 ', GPST: ' 13:22:11 '}),
        ('gps_clock_hh_mm_only', {GPSD: '2024:03:05', GPST: '13:22'}),
        ('gps_delta_whole_hour', {**local, GPSD: '2024:03:05', GPST: '13:22:11'}),
        ('gps_delta_wins_over_nothing_but_not_over_offset_tag', {**local, OFF: '+02:00', GPSD: '2024:03:05', GPST: '13:22:11'}),
        ('gps_delta_invalid_offset_tag_falls_back', {**local, OFF: 'bad', GPSD: '2024:03:05', GPST: '13:22:11'}),
        ('gps_delta_createdate_clock', {CD: '2024:03:05 14:22:11', GPSD: '2024:03:05', GPST: '13:22:11'}),
        ('gps_delta_subsec_kept', {**local, SUB: '25', GPSD: '2024:03:05', GPST: '13:22:11'}),
        ('gps_delta_zero', {**local, GPSD: '2024:03:05', GPST: '14:22:11'}),
        ('gps_delta_negative_day_boundary', {OT: '2024:03:05 01:00:00', GPSD: '2024:03:05', GPST: '13:00:00'}),
        ('gps_delta_other_day_out_of_range', {OT: '2024:03:06 14:22:11', GPSD: '2024:03:05', GPST: '13:22:11'}),
    ]  # fmt: skip
    # Quarter-hour rounding at and around the ties (Python rounds half to even): the GPS clock
    # is ``seconds`` behind the camera clock.
    camera_s = 14 * 3600 + 22 * 60 + 11
    for seconds in [0, 1, 449, 450, 451, 899, 900, 1349, 1350, 1351, 2249, 2250, 2251, 3149, 3150, -449, -450, -451, -1350, -2250]:  # fmt: skip
        hours, rest = divmod(camera_s - seconds, 3600)
        minutes, secs = divmod(rest, 60)
        name = f'gps_delta_round_{seconds}s'.replace('-', 'minus_')
        cases.append(
            (name, {**local, GPSD: '2024:03:05', GPST: f'{hours:02d}:{minutes:02d}:{secs:02d}'})
        )
    for label, hh in [
        ('plus_14', '00:22:11'),
        ('plus_14_15', '00:07:11'),
        ('minus_12', '02:22:11'),
        ('minus_12_15', '02:37:11'),
    ]:
        cases.append((f'gps_delta_limit_{label}', {OT: '2024:03:05 14:22:11', GPSD: '2024:03:05', GPST: hh}))  # fmt: skip
    return cases


def _location_cases() -> list[tuple[str, dict[str, Any]]]:
    return [
        ('loc_plain', {**FIX}),
        ('loc_south_west_signed', {LAT: -33.8688, LON: -151.2093}),
        ('loc_all_fields', {**FIX, ALT: 410.5, ACC: 4.25, DIR: 90.5, STATUS: 'A'}),
        ('loc_negative_altitude', {**FIX, ALT: -12.5}),
        ('loc_altitude_text', {**FIX, ALT: '410.5'}),
        ('loc_altitude_junk', {**FIX, ALT: 'high'}),
        ('loc_lat_only', {LAT: 47.5}),
        ('loc_lon_only', {LON: 8.25}),
        ('loc_lat_90', {LAT: 90, LON: 0.5}),
        ('loc_lat_90_01', {LAT: 90.01, LON: 0.5}),
        ('loc_lat_minus_90', {LAT: -90, LON: -180}),
        ('loc_lon_180', {LAT: 1, LON: 180}),
        ('loc_lon_180_01', {LAT: 1, LON: 180.01}),
        ('loc_lon_minus_180_01', {LAT: 1, LON: -180.01}),
        ('loc_zero_zero', {LAT: 0, LON: 0}),
        ('loc_zero_zero_float', {LAT: 0.0, LON: -0.0}),
        ('loc_zero_lat_nonzero_lon', {LAT: 0, LON: 12.5}),
        ('loc_nonzero_lat_zero_lon', {LAT: 12.5, LON: 0}),
        ('loc_tiny', {LAT: 0.0000001, LON: 0.0000001}),
        ('loc_text_numbers', {LAT: '47.5', LON: ' 8.25 '}),
        ('loc_text_underscores', {LAT: '4_7.5', LON: '8.2_5'}),
        ('loc_text_bad_underscore', {LAT: '47_.5', LON: '8.25'}),
        ('loc_text_leading_underscore', {LAT: '_47.5', LON: '8.25'}),
        ('loc_text_exponent', {LAT: '4.75e1', LON: '8.25E0'}),
        ('loc_text_plus_sign', {LAT: '+47.5', LON: '+8.25'}),
        ('loc_text_dot_forms', {LAT: '.5', LON: '8.'}),
        ('loc_text_hex', {LAT: '0x10', LON: '8'}),
        ('loc_text_junk', {LAT: 'abc', LON: '8'}),
        ('loc_text_nan', {LAT: 'nan', LON: 8}),
        ('loc_text_inf', {LAT: 'inf', LON: 8}),
        ('loc_text_infinity', {LAT: 47, LON: '-Infinity'}),
        ('loc_text_arabic_digits', {LAT: '\u0664\u0667.\u0665', LON: '8.25'}),
        ('loc_text_empty', {LAT: '', LON: 8}),
        ('loc_bool', {LAT: True, LON: 8}),
        ('loc_null', {LAT: None, LON: None}),
        ('loc_list', {LAT: [47.5], LON: 8}),
        ('loc_dms_text', {LAT: '47 deg 30\' 0.00" N', LON: '8 deg 15\' 0.00" E'}),
        ('loc_status_void', {**FIX, STATUS: 'V'}),
        ('loc_status_void_lower', {**FIX, STATUS: 'v'}),
        ('loc_status_void_padded', {**FIX, STATUS: ' V '}),
        ('loc_status_void_separator', {**FIX, STATUS: 'V\x1c'}),
        ('loc_status_active', {**FIX, STATUS: 'A'}),
        ('loc_status_word', {**FIX, STATUS: 'Void'}),
        ('loc_status_number', {**FIX, STATUS: 5}),
        ('loc_accuracy_zero', {**FIX, ACC: 0}),
        ('loc_accuracy_negative', {**FIX, ACC: -1}),
        ('loc_accuracy_text', {**FIX, ACC: '12'}),
        ('loc_accuracy_junk', {**FIX, ACC: 'x'}),
        ('loc_heading_zero', {**FIX, DIR: 0}),
        ('loc_heading_359_99', {**FIX, DIR: 359.99}),
        ('loc_heading_360', {**FIX, DIR: 360}),
        ('loc_heading_negative', {**FIX, DIR: -0.5}),
        ('loc_with_clock', {**FIX, **BASE_TIME, OFF: '+01:00'}),
        ('loc_coarse_ties_a', {LAT: 0.125, LON: 0.135}),
        ('loc_coarse_ties_b', {LAT: 12.345, LON: -12.355}),
        ('loc_coarse_ties_c', {LAT: 2.675, LON: 1.005}),
        ('loc_coarse_to_zero', {LAT: 0.004, LON: -0.004}),
        ('loc_coarse_to_zero_edge', {LAT: 0.005, LON: 0.0049}),
        ('loc_coarse_negative_zero_lat', {LAT: -0.001, LON: 0.5}),
        ('loc_coarse_near_pole', {LAT: 89.996, LON: 179.996}),
        ('loc_coarse_long_fraction', {LAT: 47.123456789, LON: 8.987654321}),
    ]


def _orientation_cases() -> list[tuple[str, dict[str, Any]]]:
    values: list[tuple[str, Any]] = [
        *((f'n{n}', n) for n in range(11)), ('minus_1', -1), ('float_6', 6.0), ('float_6_5', 6.5),
        ('float_8_0000001', 8.0000001), ('float_0_9999999', 0.9999999), ('text_6', '6'),
        ('text_padded', ' 3 '), ('text_float', '3.0'), ('text_junk', 'Rotate 90 CW'), ('text_empty', ''),
        ('null', None), ('bool_true', True), ('huge', 10**30), ('nan_text', 'nan'), ('list', [6]),
    ]  # fmt: skip
    return [(f'orientation_value_{label}', {ORI: value}) for label, value in values]


def _camera_cases() -> list[tuple[str, dict[str, Any]]]:
    texts: list[tuple[str, Any]] = [
        ('plain', 'Canon'), ('padded', '  Canon EOS  '), ('empty', ''), ('spaces_only', '   '),
        ('control_chars', 'Ca\x00no\x07n\x1b'), ('tab_and_newline', 'Ca\tno\nn'), ('only_controls', '\x00\x01'),
        ('zero_width', 'Ca\u200bnon'), ('nbsp', 'Canon\u00a0EOS'), ('line_separator', 'A\u2028B'),
        ('bom', '\ufeffCanon'), ('private_use', 'A\ue000B'), ('combining', 'Cafe\u0301'), ('accents', 'Fujifilm X-T4 \u00e9'),
        ('emoji', 'Cam \U0001f4f7 era'), ('cjk', '\u30ab\u30e1\u30e9'), ('rtl', '\u0643\u0627\u0645\u064a\u0631\u0627'),
        ('len_127', 'a' * 127), ('len_128', 'a' * 128), ('len_129', 'a' * 129), ('len_200', LONG),
        ('truncation_leaves_trailing_space', 'a' * 127 + ' b'), ('truncation_inside_leading_space', ' ' + 'a' * 200),
        ('astral_at_limit', 'a' * 127 + '\U0001f4f7' + 'b'), ('int', 5), ('int_negative', -5), ('int_big', 123456789012345678),
        ('float', 1.5), ('float_whole', 5.0), ('float_1e16', 1e16), ('float_1e15', 1e15), ('float_123456789012345680', 1.2345678901234568e17),
        ('float_small', 1e-5), ('float_0_0001', 0.0001), ('float_negative_zero', -0.0), ('float_1e22', 1e22), ('float_1e300', 1e300),
        ('float_long', 0.1234567890123456), ('float_third', 1 / 3), ('float_2_5e_minus_7', 2.5e-7), ('bool_true', True),
        ('null', None), ('list', ['Canon']), ('object', {'a': 1}),
    ]  # fmt: skip
    cases: list[tuple[str, dict[str, Any]]] = []
    for label, value in texts:
        cases.append((f'make_{label}', {MAKE: value}))
        cases.append((f'model_{label}', {MODEL: value}))
        cases.append((f'lens_{label}', {LENS: value}))
    cases.append(
        (
            'camera_all_three',
            {
                MAKE: 'NIKON CORPORATION',
                MODEL: 'NIKON D850',
                LENS: 'AF-S NIKKOR 24-70mm f/2.8E ED VR',
            },
        )
    )
    return cases


def _focal_cases() -> list[tuple[str, dict[str, Any]]]:
    values: list[tuple[str, Any]] = [
        ('normal', 35), ('float', 4.5), ('zero', 0), ('zero_float', 0.0), ('negative', -1), ('tiny', 1e-9),
        ('absurd', 1e308), ('too_big_text', '1e999'), ('text', '5.6'), ('text_padded', ' 50 '),
        ('junk', 'wide'), ('with_unit', '35.0 mm'), ('nan', 'nan'), ('inf', 'inf'), ('bool', True), ('null', None),
    ]  # fmt: skip
    cases: list[tuple[str, dict[str, Any]]] = []
    for label, value in values:
        cases.append((f'focal_{label}', {FL: value}))
        cases.append((f'focal35_{label}', {FL35: value}))
    cases.append(('focal_both', {FL: 4.5, FL35: 24}))
    return cases


def _status_cases() -> list[tuple[str, dict[str, Any]]]:
    return [
        ('status_empty_tags', {}),
        ('status_unrelated_tags_only', {'EXIF:ModifyDate': '2024:03:05 14:22:11', 'File:FileSize': 5, 'SourceFile': 'x.jpg'}),
        ('status_modify_date_is_not_capture_time', {'EXIF:ModifyDate': '2024:03:05 14:22:11', MAKE: 'X'}),
        ('status_error_only', {ERROR_KEY: 'File format error'}),
        ('status_error_with_unrelated', {ERROR_KEY: 'Unknown file type', 'SourceFile': 'x.bin'}),
        ('status_error_with_tags', {ERROR_KEY: 'Truncated', MAKE: 'X', **BASE_TIME}),
        ('status_error_null_value', {ERROR_KEY: None}),
        ('status_one_tag_ok', {MAKE: 'X'}),
        ('status_null_tag_counts_as_present', {MAKE: None}),
        ('status_keys_without_group_are_ignored', {'DateTimeOriginal': '2024:03:05 14:22:11', 'Make': 'X'}),
        ('status_keys_other_group_are_ignored', {'XMP:DateTimeOriginal': '2024:03:05 14:22:11', 'IFD0:Make': 'X'}),
        (
            'status_realistic_phone_photo',
            {
                'SourceFile': 'IMG_0001.jpg', OT: '2024:07:14 18:03:27', OFF: '+02:00', SUB: '412', CD: '2024:07:14 18:03:27',
                'EXIF:OffsetTimeDigitized': '+02:00', 'EXIF:SubSecTimeDigitized': '412', MAKE: 'Apple', MODEL: 'iPhone 15 Pro',
                LENS: 'iPhone 15 Pro back triple camera 6.765mm f/1.78', FL: 6.765, FL35: 24, ORI: 6, GPSD: '2024:07:14',
                GPST: '16:03:20', STATUS: 'A', ACC: 4.7, DIR: 211.4, LAT: 47.6205, LON: -122.3493, ALT: 56.1,
            },
        ),
    ]  # fmt: skip


def hand_cases() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = [
        *_status_cases(),
        *_time_cases(),
        *_gps_clock_cases(),
        *_location_cases(),
        *_orientation_cases(),
        *_camera_cases(),
        *_focal_cases(),
    ]
    names = [name for name, _ in cases]
    if len(names) != len(set(names)):
        raise ValueError('duplicate case name')
    return cases


# ---------------------------------------------------------------------------- recorded outputs

#: ``name -> exiftool write arguments`` for the copies of ``tests/fixtures/sample_image.jpg`` that
#: ExifTool tags. Every value is made up.
SCENARIOS: dict[str, list[str]] = {
    'tagged_full': [
        '-DateTimeOriginal=2024:03:05 14:22:11', '-OffsetTimeOriginal=+01:00', '-SubSecTimeOriginal=50',
        '-CreateDate=2024:03:05 14:22:12', '-OffsetTimeDigitized=+01:00', '-SubSecTimeDigitized=51',
        '-Make=TestCam', '-Model=T-1000', '-LensModel=Test Lens 35mm', '-FocalLength=35',
        '-FocalLengthIn35mmFormat=52', '-Orientation#=6',
    ],
    'tagged_gps_south_west': [
        '-DateTimeOriginal=2024:03:05 14:22:11', '-GPSLatitude=47.5', '-GPSLatitudeRef=S',
        '-GPSLongitude=8.25', '-GPSLongitudeRef=W', '-GPSAltitude=410', '-GPSAltitudeRef#=1',
        '-GPSStatus#=A', '-GPSImgDirection=90.5', '-GPSHPositioningError=4.25',
        '-GPSDateStamp=2024:03:05', '-GPSTimeStamp=13:22:11',
    ],
    'tagged_gps_north_east_void': [
        '-GPSLatitude=12.5', '-GPSLatitudeRef=N', '-GPSLongitude=100.25', '-GPSLongitudeRef=E', '-GPSStatus#=V',
    ],
    'tagged_gps_zero_zero': [
        '-GPSLatitude=0', '-GPSLatitudeRef=N', '-GPSLongitude=0', '-GPSLongitudeRef=E',
    ],
    'tagged_create_date_only': ['-CreateDate=2023:12:31 23:59:59'],
    'tagged_clock_with_gps_stamps_no_offset': [
        '-DateTimeOriginal=2024:03:05 14:22:11', '-GPSDateStamp=2024:03:05', '-GPSTimeStamp=13:22:11',
    ],
    'tagged_camera_only': ['-Make=Another Maker', '-Model=Z 9'],
    'tagged_orientation_3': ['-Orientation#=3'],
}  # fmt: skip
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}


def _tag_args() -> list[str]:
    capture_metadata, _ = _domain()
    wanted = (*capture_metadata.EXIFTOOL_TAGS, ERROR_KEY)
    return ['-j', '-fast', '-G', '-n', *(f'-{tag}' for tag in wanted)]


def _exiftool(*args: str, cwd: Path | None = None) -> str:
    program = shutil.which('exiftool')
    if program is None:
        raise SystemExit('--record needs exiftool on the PATH')
    done = subprocess.run(  # nosec B603  fixed argv, no shell; the program is looked up above
        [program, *args], cwd=cwd, capture_output=True, check=False, timeout=60
    )
    return done.stdout.decode()


def _read(path: str, cwd: Path) -> dict[str, Any]:
    out = _exiftool(*_tag_args(), path, cwd=cwd)
    rows = json.loads(out) if out.strip() else [{}]
    return dict(rows[0])


def record() -> dict[str, Any]:
    """Run the real ExifTool over the committed images and the scenarios (needs ``exiftool``)."""
    version = _exiftool('-ver').strip()
    rows: list[dict[str, Any]] = []
    sources = sorted(
        p.relative_to(ROOT).as_posix()
        for base in ('native/frame_engine/tests/golden/host', 'tests/fixtures')
        for p in (ROOT / base).rglob('*')
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    for source in sources:
        row = _read(source, ROOT)
        row['SourceFile'] = source
        rows.append({'name': source, 'row': row})
    with tempfile.TemporaryDirectory(prefix='capture-record-') as tmp:
        for name, args in SCENARIOS.items():
            path = Path(tmp) / f'{name}.jpg'
            shutil.copyfile(ROOT / 'tests' / 'fixtures' / 'sample_image.jpg', path)
            _exiftool('-overwrite_original', '-q', *args, str(path))
            row = _read(path.name, Path(tmp))
            row['SourceFile'] = f'scenario/{name}.jpg'
            rows.append({'name': f'scenario/{name}', 'row': row})
    return {'exiftool_version': version, 'args': _tag_args(), 'rows': rows}


def _recorded_cases() -> list[tuple[str, dict[str, Any]]]:
    data = json.loads(RECORDED.read_text(encoding='utf-8'))
    return [(f'recorded:{item["name"]}', item['row']) for item in data['rows']]


# ------------------------------------------------------------------------------------ the file


def _encode(obj: Any) -> Any:
    return dataclasses.asdict(obj)


def build() -> dict[str, Any]:
    capture_metadata, location_policy = _domain()
    if capture_metadata.FILE_ERROR_KEY != ERROR_KEY:
        raise SystemExit('the file error key changed')
    parse_rows = []
    policy_rows = []
    for name, tags in [*hand_cases(), *_recorded_cases()]:
        meta = capture_metadata.parse_capture(tags)
        parse_rows.append({'name': name, 'tags': tags, 'expected': _encode(meta)})
        if meta.location is not None or name.startswith(
            ('loc_coarse', 'recorded:scenario/tagged_gps')
        ):
            policy_rows.append(
                {
                    'name': name,
                    'tags': tags,
                    'expected': {
                        policy: _encode(location_policy.apply_location_policy(meta, policy))
                        for policy in location_policy.LOCATION_POLICIES
                    },
                }
            )
    bad_policies = ['', 'STORE', 'store ', 'drop_all', 'none']
    for policy in bad_policies:
        try:
            location_policy.apply_location_policy(capture_metadata.parse_capture({**FIX}), policy)
        except ValueError:
            continue
        raise SystemExit(f'policy {policy!r} was accepted')
    return {
        'source': 'src/domain/capture_metadata.py parse_capture, src/domain/location_policy.py apply_location_policy',
        'parser_version': capture_metadata.PARSER_VERSION,
        'exiftool_tags': list(capture_metadata.EXIFTOOL_TAGS),
        'file_error_key': ERROR_KEY,
        'policies': list(location_policy.LOCATION_POLICIES),
        'refused_policies': bad_policies,
        'parse': parse_rows,
        'policy': policy_rows,
    }


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=True) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='compare with the committed file')
    parser.add_argument('--record', action='store_true', help='re-record the ExifTool outputs')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.record:
        RECORDED.parent.mkdir(parents=True, exist_ok=True)
        RECORDED.write_text(
            json.dumps(record(), indent=1, ensure_ascii=True) + '\n', encoding='utf-8'
        )
        sys.stdout.write(f'wrote {RECORDED}\n')
    text = render()
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding='utf-8') != text:
            sys.stderr.write(f'{args.out} is stale: run scripts/codegen/capture_cases.py\n')
            return 1
        return 0
    args.out.write_text(text, encoding='utf-8')
    sys.stdout.write(f'wrote {args.out} ({len(text)} bytes)\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
