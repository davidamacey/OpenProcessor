"""The native frame engine's image builds from the repository root (plan 17 N0,
``docker/frame-engine-native/Dockerfile``): the root ``.dockerignore`` must let through what that
build reads from the context and keep excluding the rest of what it excluded before. The contract
file the engine embeds is not read from the context (``contracts/`` stays excluded): the compose
file hands it over as a named build context."""

from __future__ import annotations

import pytest

from tests.installer.test_dockerignore_context import is_ignored, load_rules


NEEDED = [
    'native/Cargo.toml',
    'native/Cargo.lock',
    'native/rust-toolchain.toml',
    'native/frame_engine/Cargo.toml',
    'native/frame_engine/build.rs',
    'native/frame_engine/src/main.rs',
    'native/frame_engine/kernels/prep.cu',
    'native/gpu_frame/proto/grpc_service.proto',
    'native/frame_engine_loadgen/Cargo.toml',
    'native/op_server/Cargo.toml',
    'native/op_server/src/main.rs',
    'native/op_api/src/lib.rs',
    'native/op_config/src/retired.rs',
    'docker/frame-engine/requirements.txt',
]
STILL_EXCLUDED = [
    'contracts/json/frame_engine_v1.json',  # the image reads it as a named build context instead
    'contracts/json/retired_env.json',  # so does the server image
    'requirements.lock',
    'docker/frame-engine-native/stand_in/null_detector_640/1/model.plan',
    'native/target/release/op-frame-engine',
]


@pytest.mark.parametrize('path', NEEDED)
def test_the_native_build_reads_it(path: str) -> None:
    assert not is_ignored(load_rules('.'), path), f'{path} would not reach the build'


@pytest.mark.parametrize('path', STILL_EXCLUDED)
def test_the_rest_stays_out_of_the_context(path: str) -> None:
    assert is_ignored(load_rules('.'), path), f'{path} would now be sent to every build'
