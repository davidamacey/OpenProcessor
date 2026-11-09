"""Capture metadata of one photo: when and with what it was taken, and where.

``parse_capture`` is pure: it takes the tag dictionary ExifTool prints for one file and returns a
:class:`CaptureMetadata`. The ExifTool call itself lives in ``src/clients/exiftool_client.py``.
The tags are requested with ``-json -n -G``: numeric values (GPS as signed decimals), each key
prefixed by its group (``EXIF:DateTimeOriginal``, ``Composite:GPSLatitude``); see
https://exiftool.org/exiftool_pod.html (``-n``, ``-G``, ``-json``).

Rules (EXIF 3.0, CIPA DC-008, tag meanings in the research spatial plan 5.1):

* ``DateTimeOriginal`` (else ``CreateDate``, the digitising time) is the capture time, a wall-clock
  time without a zone. ``ModifyDate`` (IFD0 ``DateTime``) is the modification time and is never
  capture time.
* The UTC offset comes only from ``OffsetTimeOriginal`` (``OffsetTimeDigitized`` for the digitised
  time) or from the GPS UTC time (rounded to 15 minutes, accepted within -12:00..+14:00).
* (0, 0) and out-of-range coordinates are no location, and neither is a fix flagged void.
* A missing or garbled tag gives ``None`` for that field and never an exception: a malformed capture
  date sets ``exif_status = 'malformed'``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal


PARSER_VERSION: Final = 'capture@1'

#: The tags requested from ExifTool, in the keys ``-G`` prints them under.
EXIFTOOL_TAGS: Final[tuple[str, ...]] = (
    'EXIF:DateTimeOriginal',
    'EXIF:OffsetTimeOriginal',
    'EXIF:SubSecTimeOriginal',
    'EXIF:CreateDate',
    'EXIF:OffsetTimeDigitized',
    'EXIF:SubSecTimeDigitized',
    'EXIF:Make',
    'EXIF:Model',
    'EXIF:LensModel',
    'EXIF:FocalLength',
    'EXIF:FocalLengthIn35mmFormat',
    'EXIF:Orientation',
    'EXIF:GPSDateStamp',
    'EXIF:GPSTimeStamp',
    'EXIF:GPSStatus',
    'EXIF:GPSHPositioningError',
    'EXIF:GPSImgDirection',
    'Composite:GPSLatitude',
    'Composite:GPSLongitude',
    'Composite:GPSAltitude',
)

#: Set by the reader when ExifTool reports an error for the file itself (unreadable or corrupt).
FILE_ERROR_KEY: Final = 'ExifTool:Error'

ExifStatus = Literal['ok', 'absent', 'malformed', 'unknown']
TimeSource = Literal['exif_original', 'exif_digitized', 'gps_utc', 'none']
OffsetSource = Literal['exif_offset', 'gps_utc_delta', 'none']
TimePrecision = Literal['subsecond', 'second']

_MIN_OFFSET_MIN: Final = -12 * 60
_MAX_OFFSET_MIN: Final = 14 * 60
_TEXT_MAX: Final = 128
_EXIF_DATETIME: Final = re.compile(r'^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})$')
_OFFSET: Final = re.compile(r'^([+-])(\d{2}):(\d{2})$')
_GPS_DATE: Final = re.compile(r'^(\d{4}):(\d{2}):(\d{2})$')
_GPS_TIME: Final = re.compile(r'^(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?$')
_SUBSEC: Final = re.compile(r'^\d{1,9}$')


@dataclass(frozen=True)
class CaptureLocation:
    """A valid GPS fix, signed decimal degrees (WGS 84)."""

    latitude: float
    longitude: float
    altitude_m: float | None = None
    accuracy_m: float | None = None
    heading_deg: float | None = None
    precision: Literal['exact', 'coarse'] = 'exact'


@dataclass(frozen=True)
class CaptureMetadata:
    """Everything capture-related read from one file; every field may be ``None``.

    ``taken_local`` is ISO 8601 without a zone (``2024-03-05T14:22:11`` or with ``.fff``), read as
    floating wall-clock time; ``taken_utc`` has the same shape and is set only when an offset is
    known.
    """

    exif_status: ExifStatus
    parser: str | None = PARSER_VERSION
    taken_local: str | None = None
    taken_utc: str | None = None
    offset_minutes: int | None = None
    time_source: TimeSource = 'none'
    offset_source: OffsetSource = 'none'
    time_precision: TimePrecision | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    focal_length_mm: float | None = None
    focal_length_35mm: float | None = None
    orientation: int | None = None
    location: CaptureLocation | None = None
    has_location: bool = False
    """A valid GPS fix was in the file; unchanged by the location policy, which only decides
    whether ``location`` is kept."""


def unknown_capture() -> CaptureMetadata:
    """The capture of a file whose read failed (ExifTool outage): nothing known, and no parser
    stamp, so the capture backfill reads it again."""
    return CaptureMetadata(exif_status='unknown', parser=None)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _text(value: Any) -> str | None:
    if not isinstance(value, str | int | float) or isinstance(value, bool):
        return None
    cleaned = ''.join(ch for ch in str(value) if ch.isprintable()).strip()
    return cleaned[:_TEXT_MAX] or None


def _utc_datetime(
    year: int, month: int, day: int, hour: int, minute: int, second: int
) -> datetime | None:
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=UTC)
    except ValueError:
        return None


def _local_datetime(value: Any) -> datetime | None:
    """The wall-clock time as a datetime carrying UTC only as a placeholder (floating time)."""
    if not isinstance(value, str):
        return None
    m = _EXIF_DATETIME.match(value.strip())
    if m is None:
        return None
    return _utc_datetime(*(int(g) for g in m.groups()))


def _offset_minutes(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    m = _OFFSET.match(value.strip())
    if m is None:
        return None
    hours, minutes = int(m.group(2)), int(m.group(3))
    if minutes > 59:
        return None
    total = hours * 60 + minutes
    total = -total if m.group(1) == '-' else total
    return total if _MIN_OFFSET_MIN <= total <= _MAX_OFFSET_MIN else None


def _subsecond(value: Any) -> str | None:
    """The digits after the decimal point, or ``None`` for anything that is not digits."""
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    digits = str(value).strip()
    return digits if _SUBSEC.match(digits) else None


def _gps_utc(tags: dict[str, Any]) -> datetime | None:
    date_m = _GPS_DATE.match(str(tags.get('EXIF:GPSDateStamp', '')).strip())
    time_m = _GPS_TIME.match(str(tags.get('EXIF:GPSTimeStamp', '')).strip())
    if date_m is None or time_m is None:
        return None
    return _utc_datetime(*(int(g) for g in date_m.groups()), *(int(g) for g in time_m.groups()))


def _iso(moment: datetime, subsec: str | None) -> str:
    base = moment.strftime('%Y-%m-%dT%H:%M:%S')
    return f'{base}.{subsec[:6]}' if subsec else base


def _round_to_quarter_hour(delta: timedelta) -> int:
    quarter = 15 * 60
    return round(delta.total_seconds() / quarter) * 15


@dataclass(frozen=True)
class _Time:
    local: str | None = None
    utc: str | None = None
    offset: int | None = None
    source: TimeSource = 'none'
    offset_source: OffsetSource = 'none'
    precision: TimePrecision | None = None
    malformed: bool = False


_CLOCKS: Final[tuple[tuple[TimeSource, str, str, str], ...]] = (
    (
        'exif_original',
        'EXIF:DateTimeOriginal',
        'EXIF:OffsetTimeOriginal',
        'EXIF:SubSecTimeOriginal',
    ),
    (
        'exif_digitized',
        'EXIF:CreateDate',
        'EXIF:OffsetTimeDigitized',
        'EXIF:SubSecTimeDigitized',
    ),
)


def _parse_time(tags: dict[str, Any]) -> _Time:
    gps = _gps_utc(tags)
    for source, date_key, offset_key, subsec_key in _CLOCKS:
        if date_key not in tags:
            continue
        moment = _local_datetime(tags[date_key])
        if moment is None:
            return _Time(malformed=True)
        subsec = _subsecond(tags.get(subsec_key))
        offset = _offset_minutes(tags.get(offset_key))
        offset_source: OffsetSource = 'exif_offset' if offset is not None else 'none'
        if offset is None and gps is not None:
            delta = _round_to_quarter_hour(moment - gps)
            if _MIN_OFFSET_MIN <= delta <= _MAX_OFFSET_MIN:
                offset, offset_source = delta, 'gps_utc_delta'
        utc = _iso(moment - timedelta(minutes=offset), subsec) if offset is not None else None
        return _Time(
            local=_iso(moment, subsec),
            utc=utc,
            offset=offset,
            source=source,
            offset_source=offset_source,
            precision='subsecond' if subsec else 'second',
        )
    if gps is not None:
        # No camera clock: GPS UTC time is the capture instant; the local wall clock is unknown.
        return _Time(utc=_iso(gps, None), source='gps_utc', precision='second')
    return _Time()


def _parse_location(tags: dict[str, Any]) -> CaptureLocation | None:
    if str(tags.get('EXIF:GPSStatus', '')).strip().upper() == 'V':
        return None
    lat = _number(tags.get('Composite:GPSLatitude'))
    lon = _number(tags.get('Composite:GPSLongitude'))
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    if lat == 0 and lon == 0:
        return None
    accuracy = _number(tags.get('EXIF:GPSHPositioningError'))
    heading = _number(tags.get('EXIF:GPSImgDirection'))
    return CaptureLocation(
        latitude=lat,
        longitude=lon,
        altitude_m=_number(tags.get('Composite:GPSAltitude')),
        accuracy_m=accuracy if accuracy is not None and accuracy >= 0 else None,
        heading_deg=heading if heading is not None and 0 <= heading < 360 else None,
    )


def _positive(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _orientation(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number in range(1, 9) else None


def parse_capture(tags: dict[str, Any]) -> CaptureMetadata:
    """The capture metadata in ``tags`` (ExifTool ``-json -n -G`` output of one file)."""
    wanted = {key: tags[key] for key in EXIFTOOL_TAGS if key in tags}
    unreadable = FILE_ERROR_KEY in tags
    if not wanted:
        return CaptureMetadata(exif_status='malformed' if unreadable else 'absent')
    when = _parse_time(wanted)
    location = _parse_location(wanted)
    return CaptureMetadata(
        exif_status='malformed' if when.malformed or unreadable else 'ok',
        taken_local=when.local,
        taken_utc=when.utc,
        offset_minutes=when.offset,
        time_source=when.source,
        offset_source=when.offset_source,
        time_precision=when.precision,
        camera_make=_text(wanted.get('EXIF:Make')),
        camera_model=_text(wanted.get('EXIF:Model')),
        lens_model=_text(wanted.get('EXIF:LensModel')),
        focal_length_mm=_positive(wanted.get('EXIF:FocalLength')),
        focal_length_35mm=_positive(wanted.get('EXIF:FocalLengthIn35mmFormat')),
        orientation=_orientation(wanted.get('EXIF:Orientation')),
        location=location,
        has_location=location is not None,
    )


__all__ = [
    'EXIFTOOL_TAGS',
    'FILE_ERROR_KEY',
    'PARSER_VERSION',
    'CaptureLocation',
    'CaptureMetadata',
    'parse_capture',
    'unknown_capture',
]
