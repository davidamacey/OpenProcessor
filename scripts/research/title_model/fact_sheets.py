"""Run R8 fact sheets: 200 made-up memories (dates, place, people, event, scene words) and the
list of words a title may use, which is what "no invented fact" is checked against."""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass

from scripts.research.title_model.memories import SceneWord, find_words, load_vocabulary, normalize


N_SHEETS = 200
EVENT_RATE = 0.4  # share of the free sheets (the 140 that are no test memory) that hold an event

CITIES = [
    'Lisbon',
    'Porto',
    'Madrid',
    'Barcelona',
    'Seville',
    'Valencia',
    'Paris',
    'Lyon',
    'Nice',
    'Marseille',
    'Brussels',
    'Amsterdam',
    'Rotterdam',
    'Berlin',
    'Munich',
    'Hamburg',
    'Vienna',
    'Salzburg',
    'Prague',
    'Budapest',
    'Warsaw',
    'Krakow',
    'Rome',
    'Florence',
    'Venice',
    'Naples',
    'Milan',
    'Athens',
    'Istanbul',
    'Dublin',
    'Edinburgh',
    'Glasgow',
    'Manchester',
    'Leeds',
    'Bristol',
    'Oslo',
    'Stockholm',
    'Copenhagen',
    'Helsinki',
    'Reykjavik',
    'Zurich',
    'Geneva',
    'Moscow',
    'Cairo',
    'Marrakech',
    'Nairobi',
    'Cape Town',
    'Lagos',
    'Dubai',
    'Tel Aviv',
    'Mumbai',
    'Delhi',
    'Jaipur',
    'Bangkok',
    'Hanoi',
    'Singapore',
    'Jakarta',
    'Manila',
    'Seoul',
    'Tokyo',
    'Osaka',
    'Kyoto',
    'Beijing',
    'Shanghai',
    'Taipei',
    'Sydney',
    'Melbourne',
    'Perth',
    'Auckland',
    'Vancouver',
    'Toronto',
    'Montreal',
    'Chicago',
    'Boston',
    'Seattle',
    'Denver',
    'Austin',
    'Miami',
    'Atlanta',
    'Portland',
    'Honolulu',
    'New York',
    'San Francisco',
    'Los Angeles',
    'San Diego',
    'Mexico City',
    'Havana',
    'Bogota',
    'Lima',
    'Quito',
    'Santiago',
    'Buenos Aires',
    'Montevideo',
    'Rio de Janeiro',
    'Sao Paulo',
    'Cusco',
    'Kathmandu',
    'Colombo',
    'Hobart',
    'Anchorage',
]
LANDMARKS = [
    'Eiffel Tower',
    'Colosseum',
    'Sagrada Familia',
    'Big Ben',
    'Brandenburg Gate',
    'Acropolis',
    'Taj Mahal',
    'Great Wall',
    'Mount Fuji',
    'Machu Picchu',
    'Table Mountain',
    'Golden Gate Bridge',
    'Grand Canyon',
    'Niagara Falls',
    'Yosemite',
    'Yellowstone',
    'Lake Louise',
    'Angkor Wat',
    'Petra',
    'Uluru',
    'Sydney Opera House',
    'Christ the Redeemer',
    'Stonehenge',
    'Loch Ness',
    'Giants Causeway',
    'Alhambra',
    'Neuschwanstein',
    'Mont Saint-Michel',
    'Victoria Falls',
    'Halong Bay',
]
PLACES = tuple(CITIES) + tuple(LANDMARKS)
NAMES = (
    'Ana',
    'Ben',
    'Kai',
    'Mira',
    'Leo',
    'Nora',
    'Omar',
    'Zoe',
    'Ivan',
    'Sara',
    'Theo',
    'Lena',
    'Raj',
    'Maya',
    'Jon',
    'Ines',
    'Hugo',
    'Aya',
    'Finn',
    'Rosa',
)

# Fixed-date holidays: name -> (month, day). Movable ones are computed in holiday_date.
FIXED_HOLIDAYS = {
    "New Year's Day": (1, 1),
    "Valentine's Day": (2, 14),
    "St Patrick's Day": (3, 17),
    'Independence Day': (7, 4),
    'Halloween': (10, 31),
    'Bonfire Night': (11, 5),
    'Christmas Eve': (12, 24),
    'Christmas': (12, 25),
    'Boxing Day': (12, 26),
    "New Year's Eve": (12, 31),
}
MOVABLE_HOLIDAYS = ('Easter', 'Memorial Day', 'Labor Day', 'Thanksgiving')
HOLIDAYS = tuple(FIXED_HOLIDAYS) + MOVABLE_HOLIDAYS
OTHER_EVENTS = (
    'wedding',
    'concert',
    'hike',
    'beach day',
    'birthday',
    'graduation',
    'road trip',
    'family reunion',
    'ski trip',
    'camping trip',
    'picnic',
    'festival',
    'anniversary',
    'barbecue',
)
EVENTS = HOLIDAYS + OTHER_EVENTS

MONTHS = (
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
)
WEEKDAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
SEASONS = ('winter', 'spring', 'summer', 'autumn')
NUMBER_WORDS = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten',
    11: 'eleven',
    12: 'twelve',
    13: 'thirteen',
    14: 'fourteen',
    15: 'fifteen',
    16: 'sixteen',
    17: 'seventeen',
    18: 'eighteen',
    19: 'nineteen',
    20: 'twenty',
    21: 'twenty one',
    22: 'twenty two',
}


@dataclass(frozen=True)
class FactSheet:
    sheet_id: int
    start: dt.date
    end: dt.date
    place: str | None
    people: int
    named: tuple[str, ...]
    event: str | None
    scene_words: tuple[str, ...]

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The ``n``-th (1-based; -1 = last) ``weekday`` (Monday=0) of a month."""
    if n > 0:
        first = dt.date(year, month, 1)
        return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (the anonymous computus)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    n = h + l_ - 7 * m + 114
    return dt.date(year, n // 31, n % 31 + 1)


def holiday_date(name: str, year: int) -> dt.date:
    if name in FIXED_HOLIDAYS:
        month, day = FIXED_HOLIDAYS[name]
        return dt.date(year, month, day)
    return {
        'Easter': lambda: _easter(year),
        'Memorial Day': lambda: _nth_weekday(year, 5, 0, -1),
        'Labor Day': lambda: _nth_weekday(year, 9, 0, 1),
        'Thanksgiving': lambda: _nth_weekday(year, 11, 3, 4),
    }[name]()


def season_of(month: int) -> str:
    """Northern-hemisphere meteorological season."""
    return SEASONS[(month % 12) // 3]


def _dates(rng: random.Random, event: str | None) -> tuple[dt.date, dt.date]:
    year = rng.randint(2015, 2025)
    length = rng.randint(1, 21)
    if event in HOLIDAYS:
        centre = holiday_date(event, year)
        start = centre - dt.timedelta(days=rng.randint(0, min(length, 3) - 1 if length > 1 else 0))
        return start, start + dt.timedelta(days=min(length, 4) - 1)
    start = dt.date(year, 1, 1) + dt.timedelta(days=rng.randint(0, 364))
    return start, start + dt.timedelta(days=length - 1)


def make_sheets(
    memory_scene_words: list[tuple[str, ...]],
    vocabulary: list[SceneWord],
    seed: int,
    n: int = N_SHEETS,
) -> list[FactSheet]:
    """``n`` sheets. The first ``len(memory_scene_words)`` belong to the test memories: no event,
    the memory's kept scene words. The rest draw an event (or none) and 0 to 2 scene words."""
    rng = random.Random(seed + 7)
    words = [v.word for v in vocabulary]
    sheets = []
    for i in range(n):
        own = i < len(memory_scene_words)
        event = None if own or rng.random() >= EVENT_RATE else rng.choice(EVENTS)
        start, end = _dates(rng, event)
        people = rng.randint(0, 12)
        named = (
            tuple(rng.sample(NAMES, rng.randint(1, min(people, 3))))
            if people and rng.random() < 0.5
            else ()
        )
        scene = memory_scene_words[i] if own else tuple(rng.sample(words, rng.choice((0, 1, 1, 2))))
        place = rng.choice(PLACES) if rng.random() < 0.8 else None
        sheets.append(FactSheet(i, start, end, place, people, named, event, scene))
    return sheets


def date_words(sheet: FactSheet) -> set[str]:
    """Words derived from the dates: months, years, days of the month, seasons, weekdays of the range,
    'weekend' and 'week' when they are true of it."""
    days = [sheet.start + dt.timedelta(days=d) for d in range(sheet.days)]
    out = {normalize(MONTHS[d.month - 1]) for d in days}
    out |= (
        {str(d.year) for d in days}
        | {str(d.day) for d in days}
        | {season_of(d.month) for d in days}
    )
    out |= {normalize(WEEKDAYS[d.weekday()]) for d in days}
    if any(d.weekday() >= 5 for d in days):
        out.add('weekend')
    if sheet.days >= 6:
        out.add('week')
    return out


def allowed_numbers(sheet: FactSheet) -> set[str]:
    nums = {sheet.people, sheet.days, len(sheet.named)}
    out = {str(x) for x in nums}
    out |= {NUMBER_WORDS[x] for x in nums if x in NUMBER_WORDS}
    return out


def invented_facts(text: str, sheet: FactSheet, vocabulary: list[SceneWord]) -> list[str]:
    """Facts in ``text`` that the sheet does not hold; empty when the text is clean.

    A fact is: a number the sheet lacks; a month, year, season or weekday outside its dates; a
    known place, first name or event it lacks; a scene word it lacks. The sheet's own scene words,
    its event, and the synonyms of both are allowed.
    """
    norm = f' {normalize(text)} '
    own = {normalize(p) for p in (sheet.place, sheet.event, *sheet.named) if p}
    dates = date_words(sheet)
    numbers = allowed_numbers(sheet)
    bad = [
        tok
        for tok in norm.split()
        if (tok in _NUMBER_TOKENS and tok not in numbers)
        or (tok.isdigit() and tok not in numbers | dates)
        or (tok in _DATE_TOKENS and tok not in dates)
    ]
    bad += [
        phrase
        for phrase in _KNOWN_PHRASES
        if f' {phrase} ' in norm and not any(f' {phrase} ' in f' {o} ' for o in own)
    ]
    unowned = norm
    for o in sorted(own, key=len, reverse=True):
        while f' {o} ' in unowned:
            unowned = unowned.replace(f' {o} ', '  ')
    bad += sorted(find_words(unowned, vocabulary) - _allowed_scene_words(sheet, vocabulary))
    return bad


def _allowed_scene_words(sheet: FactSheet, vocabulary: list[SceneWord]) -> set[str]:
    """The sheet's scene words plus every word that one of the sheet's event words stands for."""
    event_tokens = set(normalize(sheet.event or '').split()) | {normalize(sheet.event or '')}
    from_event = {v.word for v in vocabulary if event_tokens & {normalize(f) for f in v.forms}}
    return set(sheet.scene_words) | from_event


_NUMBER_TOKENS = {
    t for w in NUMBER_WORDS.values() for t in normalize(w).split() if t != 'twenty'
} | {'twenty'}
_VOCAB = load_vocabulary()
_DATE_TOKENS = {normalize(m) for m in MONTHS} | {normalize(d) for d in WEEKDAYS} | set(SEASONS)
# Events a scene word already covers ("beach day" is "beach") are checked as scene words, not here.
_KNOWN_PHRASES = tuple(
    sorted(
        {normalize(x) for x in (*PLACES, *NAMES)}
        | {normalize(e) for e in EVENTS if not find_words(e, _VOCAB)},
        key=len,
        reverse=True,
    )
)
