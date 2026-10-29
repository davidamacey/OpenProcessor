"""Label derivation for the vehicle proxy datasets (class identity is the name, never an index).

Stanford Cars class names read "<Make> <Model> <Body> <Year>" ("BMW M3 Coupe 2012"). CompCars
stores ``image/<make_id>/<model_id>/<year>/<file>.jpg`` with a parallel ``label/...txt`` whose
first line is the viewpoint code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scripts.datasets._common import FetchError
from scripts.datasets.proxy_manifest import safe_relpath


# Longest first so "Crew Cab" wins over "Cab" and "Cargo Van" over "Van".
BODY_STYLES = (
    'Extended Cab',
    'Regular Cab',
    'Crew Cab',
    'Club Cab',
    'Quad Cab',
    'Cargo Van',
    'Convertible',
    'Hatchback',
    'Minivan',
    'Roadster',
    'SuperCab',
    'Sedan',
    'Coupe',
    'Wagon',
    'SUV',
    'Van',
    'Cab',
)
MULTI_WORD_MAKES = ('AM General', 'Aston Martin', 'Land Rover', 'Rolls-Royce')
_YEAR = re.compile(r'^(?P<rest>.*\S)\s+(?P<year>(?:19|20)\d{2})$')

COMPCARS_VIEWS = {
    1: 'front',
    2: 'rear',
    3: 'side',
    4: 'front-side',
    5: 'rear-side',
}


@dataclass(frozen=True)
class VehicleClass:
    make: str
    model: str
    body: str | None
    year: int

    @property
    def class_name(self) -> str:
        return f'{self.make} {self.model}'

    @property
    def taxonomy_path(self) -> list[str]:
        return [self.make, self.class_name]


def parse_stanford_class(name: str) -> VehicleClass:
    """``"BMW M3 Coupe 2012"`` -> make ``BMW``, model ``M3``, body ``Coupe``, year 2012."""
    m = _YEAR.match(name.strip())
    if not m:
        raise FetchError(f'Stanford class {name!r} does not end in a four-digit year')
    rest, year = m.group('rest'), int(m.group('year'))
    make = next((mk for mk in MULTI_WORD_MAKES if rest.startswith(mk + ' ')), None)
    if make is None:
        make = rest.split(' ', 1)[0]
    remainder = rest[len(make) :].strip()
    body = next((b for b in BODY_STYLES if remainder.endswith(' ' + b) or remainder == b), None)
    model = remainder[: -len(body)].strip() if body else remainder
    if not model:
        raise FetchError(f'Stanford class {name!r} has no model part')
    return VehicleClass(make=make, model=model, body=body, year=year)


@dataclass(frozen=True)
class CompCarsRef:
    make_id: int
    model_id: int
    year: str
    file_name: str

    @property
    def label_path(self) -> str:
        stem = self.file_name.rsplit('.', 1)[0]
        return f'label/{self.make_id}/{self.model_id}/{self.year}/{stem}.txt'


def parse_compcars_image(rel: str) -> CompCarsRef:
    """``image/9/1369/2012/abc.jpg`` -> ids and year; anything else is refused."""
    parts = safe_relpath(rel).split('/')
    if len(parts) != 5 or parts[0] != 'image' or not (parts[1].isdigit() and parts[2].isdigit()):
        raise FetchError(f'{rel!r} is not image/<make_id>/<model_id>/<year>/<file>')
    return CompCarsRef(int(parts[1]), int(parts[2]), parts[3], parts[4])


def parse_compcars_view(label_text: str) -> str:
    """The viewpoint name from the first line of a CompCars label file."""
    first = label_text.splitlines()[0].strip() if label_text.strip() else ''
    if not first.lstrip('-').isdigit() or int(first) not in COMPCARS_VIEWS:
        raise FetchError(f'unknown CompCars viewpoint code {first!r}')
    return COMPCARS_VIEWS[int(first)]


def compcars_names(
    ref: CompCarsRef, make_names: list[str], model_names: list[str]
) -> tuple[str, str]:
    """Make and model names for the 1-based ids; an out-of-range id is an error."""
    if not 1 <= ref.make_id <= len(make_names) or not 1 <= ref.model_id <= len(model_names):
        raise FetchError(f'CompCars ids {ref.make_id}/{ref.model_id} are outside the name lists')
    return make_names[ref.make_id - 1], model_names[ref.model_id - 1]
