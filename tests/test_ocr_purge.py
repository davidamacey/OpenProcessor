"""The core text-index purge: dry-run counts, removal, idempotence, counts-only audit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts import purge_ocr_data


if TYPE_CHECKING:
    from pathlib import Path

OCR = 'visual_search_ocr'
SECRET = 'STORED-LINE-9Q'


class FakeIndices:
    def __init__(self, os_: FakeOpenSearch) -> None:
        self.os = os_

    async def exists(self, index: str) -> bool:
        return index in self.os.docs

    async def forcemerge(self, index: str, **kw: Any) -> None:
        self.os.mutations.append(('forcemerge', index, kw))


class FakeOpenSearch:
    def __init__(self, leave_behind: bool = False) -> None:
        self.leave_behind = leave_behind
        self.docs: dict[str, list[dict[str, Any]]] = {
            OCR: [{'image_id': f'img{i}', 'text': f'{SECRET} {i}'} for i in range(4)]
        }
        self.mutations: list[tuple[Any, ...]] = []
        self.indices = FakeIndices(self)

    async def count(self, index: str, **_kw: Any) -> dict[str, int]:
        return {'count': len(self.docs.get(index, []))}

    async def delete_by_query(self, index: str, body: dict[str, Any], **_kw: Any) -> dict[str, int]:
        self.mutations.append(('delete_by_query', index, body))
        n = len(self.docs[index])
        if not self.leave_behind:
            self.docs[index] = []
        return {'deleted': n}

    async def close(self) -> None:
        return None


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.chdir(tmp_path)
    fake = FakeOpenSearch()
    monkeypatch.setattr(purge_ocr_data, 'make_client', lambda: fake)
    return {'fake': fake, 'root': tmp_path}


def _last_json(out: str) -> Any:
    return json.loads(out.strip().splitlines()[-1])


def test_dry_run_counts_and_changes_nothing(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert purge_ocr_data.main(['--dry-run']) == 0
    assert _last_json(capsys.readouterr().out)['lines'] == 4
    assert env['fake'].mutations == []


def test_without_yes_refuses(env: dict[str, Any]) -> None:
    assert purge_ocr_data.main([]) == 2
    assert env['fake'].mutations == []


def test_yes_removes_every_line_and_audits_counts_only(env: dict[str, Any]) -> None:
    assert purge_ocr_data.main(['--yes']) == 0
    assert env['fake'].docs[OCR] == []
    assert [m[0] for m in env['fake'].mutations] == ['delete_by_query', 'forcemerge']
    audit = (env['root'] / 'data' / 'audit' / 'ocr_purge.jsonl').read_text().splitlines()
    assert len(audit) == 1
    assert SECRET not in audit[0]
    assert json.loads(audit[0])['lines'] == 4


def test_second_purge_is_a_clean_no_op(env: dict[str, Any]) -> None:
    assert purge_ocr_data.main(['--yes']) == 0
    assert purge_ocr_data.main(['--yes']) == 0
    assert len((env['root'] / 'data' / 'audit' / 'ocr_purge.jsonl').read_text().splitlines()) == 2


def test_lines_left_behind_fail_the_purge(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(purge_ocr_data, 'make_client', lambda: FakeOpenSearch(leave_behind=True))
    assert purge_ocr_data.main(['--yes']) == 1
