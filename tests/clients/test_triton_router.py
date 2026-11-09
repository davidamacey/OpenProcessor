"""The Triton router: fewest outstanding among ready servers, image affinity, fail closed."""

from __future__ import annotations

from contextlib import ExitStack

import pytest
from prometheus_client import REGISTRY

from src.clients.triton_router import NoServerError, TritonRouter


MODEL = 'yolo'


def two_servers() -> TritonRouter:
    router = TritonRouter({'a': 'a:8001', 'b': 'b:8001'})
    router.update_ready({'a': [MODEL], 'b': [MODEL]})
    return router


def test_one_hundred_concurrent_calls_split_within_ten_percent() -> None:
    router = two_servers()
    with ExitStack() as stack:
        servers = [stack.enter_context(router.route(MODEL)).server for _ in range(100)]
        assert abs(servers.count('a') - servers.count('b')) <= 10
        assert router.outstanding() == {'a': 50, 'b': 50}
    assert router.outstanding() == {'a': 0, 'b': 0}


def test_a_finished_call_frees_its_server_for_the_next_one() -> None:
    router = two_servers()
    with router.route(MODEL) as first:
        with router.route(MODEL) as second:
            assert {first.server, second.server} == {'a', 'b'}
        with router.route(MODEL) as third:
            assert third.server == second.server


def test_a_server_whose_report_lacks_the_model_gets_no_call() -> None:
    router = TritonRouter({'a': 'a:8001', 'b': 'b:8001'})
    router.update_ready({'a': [MODEL], 'b': ['other']})
    with ExitStack() as stack:
        servers = {stack.enter_context(router.route(MODEL)).server for _ in range(20)}
    assert servers == {'a'}


def test_no_ready_server_refuses_instead_of_guessing() -> None:
    router = TritonRouter({'a': 'a:8001'})
    with pytest.raises(NoServerError), router.route(MODEL):
        pass
    router.update_ready({'a': ['other']})
    with pytest.raises(NoServerError), router.route(MODEL):
        pass


def test_a_server_that_stops_reporting_is_dropped_on_the_next_update() -> None:
    router = two_servers()
    router.update_ready({'b': [MODEL]})
    with router.route(MODEL) as route:
        assert route.server == 'b'


def test_follow_up_calls_of_one_image_stay_on_its_server_even_when_it_is_busier() -> None:
    router = two_servers()
    # The image-prep call is still in flight, so its server is busier than the other one.
    with (
        router.affinity('img-1'),
        router.route(MODEL, affinity='img-1') as prep,
        router.route(MODEL) as unrelated,
        router.route(MODEL, affinity='img-1') as follow_up,
    ):
        assert unrelated.server != prep.server
        assert follow_up.server == prep.server


def test_a_pinned_server_that_loses_the_model_refuses_the_follow_up() -> None:
    router = two_servers()
    with router.affinity('img-1'):
        with router.route(MODEL, affinity='img-1') as first:
            pinned = first.server
        router.update_ready({s: [MODEL] for s in ('a', 'b') if s != pinned})
        with pytest.raises(NoServerError), router.route(MODEL, affinity='img-1'):
            pass


def test_the_pin_is_released_when_the_scope_closes() -> None:
    router = two_servers()
    with router.affinity('img-1'), router.route(MODEL, affinity='img-1'):
        pass
    with pytest.raises(ValueError, match='not open'), router.route(MODEL, affinity='img-1'):
        pass
    with router.affinity('img-1'):
        pass


def test_one_server_behaves_as_before_the_router() -> None:
    router = TritonRouter.single('triton-server:8001')
    with router.route('any-model') as route:
        assert (route.server, route.url) == ('default', 'triton-server:8001')
    with router.route('another') as route:
        assert route.url == 'triton-server:8001'


def test_the_gauge_follows_the_outstanding_count() -> None:
    router = TritonRouter({'gauge-server': 'g:8001'})
    router.update_ready({'gauge-server': [MODEL]})

    def gauge() -> float | None:
        return REGISTRY.get_sample_value('op_triton_router_outstanding', {'server': 'gauge-server'})

    with router.route(MODEL):
        assert gauge() == 1
    assert gauge() == 0


@pytest.mark.asyncio
async def test_the_manager_hands_out_the_client_of_the_routed_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.clients.triton_client_manager import TritonClientManager

    urls: list[str] = []

    async def fake_client(url: str, **_: object) -> str:
        urls.append(url)
        return f'client-{url}'

    monkeypatch.setattr(TritonClientManager, 'get_async_client', fake_client)
    monkeypatch.setattr(TritonClientManager, '_router', None)
    with pytest.raises(RuntimeError, match='not configured'):
        async with TritonClientManager.routed_async_client(MODEL):
            pass
    router = two_servers()
    TritonClientManager.configure_router(router)
    async with TritonClientManager.routed_async_client(MODEL) as client:
        assert client == f'client-{urls[0]}'
        assert sum(router.outstanding().values()) == 1
    assert sum(router.outstanding().values()) == 0
