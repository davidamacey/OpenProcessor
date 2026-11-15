"""``scripts/bench/frame_engine_stand_ins.py``: the null and capture models of the N0 comparison.

The configs are parsed as Triton parses them (``model_config_pb2``), the null graphs are run on
the ONNX reference evaluator, and the capture model's ``execute`` runs on host arrays against a
stand-in for the server's ``triton_python_backend_utils``."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from google.protobuf import text_format
from onnx.reference import ReferenceEvaluator
from tritonclient.grpc import model_config_pb2

from scripts.bench import frame_engine_stand_ins as si


REPO = Path(__file__).resolve().parents[2]
SPEC = {s.null_name: s for s in si.SPECS}


def parse(text: str) -> Any:
    config = model_config_pb2.ModelConfig()
    text_format.Parse(text, config)
    return config


def real(name: str) -> Any:
    return parse((REPO / 'models' / name / 'config.pbtxt').read_text())


def contract(c: Any) -> dict[str, Any]:
    """What a client sees of a model: tensors, batching."""
    return {
        'max_batch': c.max_batch_size,
        'inputs': [(i.name, i.data_type, list(i.dims)) for i in c.input],
        'outputs': [(o.name, o.data_type, list(o.dims)) for o in c.output],
        'preferred': list(c.dynamic_batching.preferred_batch_size),
        'delay': c.dynamic_batching.max_queue_delay_microseconds,
    }


def test_the_null_detector_config_is_the_plans() -> None:
    c = parse(si.null_config(SPEC['null_detector_640']))
    assert (c.name, c.platform, c.max_batch_size) == ('null_detector_640', 'tensorrt_plan', 64)
    assert contract(c)['inputs'] == [('images', model_config_pb2.TYPE_UINT8, [3, 640, 640])]
    assert contract(c)['outputs'] == [('output0', model_config_pb2.TYPE_FP32, [300, 6])]
    assert list(c.dynamic_batching.preferred_batch_size) == [8, 16, 32, 64]
    assert c.dynamic_batching.max_queue_delay_microseconds == 25000
    group = c.instance_group[0]
    assert (group.count, group.kind, list(group.gpus)) == (
        2,
        model_config_pb2.ModelInstanceGroup.KIND_GPU,
        [0],
    )


@pytest.mark.parametrize(
    ('null', 'model'),
    [('null_encoder_336', 'pe_image_encoder'), ('null_encoder_224', 'pe_crop_encoder')],
)
def test_a_null_encoder_repeats_the_real_encoders_contract_under_its_own_name(
    null: str, model: str
) -> None:
    ours, theirs = parse(si.null_config(SPEC[null])), real(model)
    assert ours.name == null
    assert contract(ours) == contract(theirs)
    assert ours.platform == theirs.platform


def test_the_detector_stand_in_mirrors_the_real_detectors_contract_too() -> None:
    ours, theirs = (
        parse(si.null_config(SPEC['null_detector_640'])),
        real('yolov11_small_trt_end2end'),
    )
    assert contract(ours) == contract(theirs)


def test_more_instances_are_a_flag_not_a_rewrite() -> None:
    assert (
        parse(si.null_config(SPEC['null_detector_640'], instances=5)).instance_group[0].count == 5
    )
    assert parse(si.null_config(SPEC['null_encoder_224'], instances=3)).instance_group[0].count == 3


def test_a_capture_model_is_the_null_models_tensors_on_the_python_backend() -> None:
    for s in si.SPECS:
        c = parse(si.capture_config(s))
        assert c.backend == 'python'
        assert not c.platform
        assert contract(c) == contract(parse(si.null_config(s)))
        assert c.parameters['FORCE_CPU_ONLY_INPUT_TENSORS'].string_value == 'yes'
        assert c.parameters['CAPTURE_DIR'].string_value == si.SERVER_CAPTURE_DIR
        assert c.instance_group[0].count == 1, 'one instance: one arrival order'


def test_the_fixed_boxes_are_centred_and_straddle_the_crop_square() -> None:
    rows = np.array(si.detector_rows())
    assert (rows[:, 2] - rows[:, 0]).tolist() == [16, 48, 160, 320, 600]
    assert ((rows[:, 0] + rows[:, 2]) / 2).tolist() == [320] * 5
    assert rows[:, 4].tolist() == pytest.approx([0.9] * 5)
    assert rows[:, 5].tolist() == [0, 2, 5, 7, 16]
    sides = rows[:, 2] - rows[:, 0]
    assert sides.min() < 224 < sides.max()


def evaluate(spec: Any, batch: np.ndarray) -> np.ndarray:
    outputs: Any = ReferenceEvaluator(si.null_onnx(spec)).run(None, {'images': batch})
    return np.asarray(outputs[0])


def test_the_null_detector_graph_answers_the_fixed_rows_whatever_the_pixels_are() -> None:
    spec = SPEC['null_detector_640']
    rng = np.random.default_rng(0)
    batch = rng.integers(0, 256, (3, 3, 640, 640), dtype=np.uint8)
    out = evaluate(spec, batch)
    assert out.shape == (3, 300, 6)
    assert out[1, :5].tolist() == np.array(si.detector_rows(), np.float32).tolist()
    assert not out[:, 5:].any(), 'every row past the fifth is empty'
    assert np.array_equal(out, evaluate(spec, np.full_like(batch, 255)))


def test_a_null_encoder_graph_answers_one_unit_vector() -> None:
    spec = SPEC['null_encoder_224']
    out = evaluate(
        spec, np.random.default_rng(1).integers(0, 256, (4, 3, 224, 224), dtype=np.uint8)
    )
    assert out.shape == (4, 1024)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert (out == out[0]).all()


def test_the_graph_keeps_its_input_connected_to_the_output() -> None:
    # the input must feed the output, or TensorRT drops the binding the server config names
    graph = si.null_onnx(SPEC['null_encoder_336']).graph
    produced = {o for n in graph.node for o in n.output}
    assert graph.node[0].input[0] == 'images'
    assert graph.output[0].name in produced
    chain = [n.op_type for n in graph.node]
    assert chain == ['Slice', 'Cast', 'Mul', 'Reshape', 'Add']


def test_the_trtexec_command_builds_the_real_models_batch_range() -> None:
    args = si.trtexec_args(SPEC['null_detector_640'])
    assert '--minShapes=images:1x3x640x640' in args
    assert '--optShapes=images:16x3x640x640' in args
    assert '--maxShapes=images:64x3x640x640' in args
    assert '--saveEngine=/stand_in/null_detector_640/1/model.plan' in args
    assert '--onnx=/stand_in/_onnx/null_detector_640.onnx' in args
    assert '--skipInference' in args
    assert '--maxShapes=images:32x3x224x224' in si.trtexec_args(SPEC['null_encoder_224'])


@pytest.fixture
def pb_utils(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """The server's ``triton_python_backend_utils``, as far as the capture model uses it."""
    module = types.ModuleType('triton_python_backend_utils')

    class Tensor:
        def __init__(self, name: str, array: np.ndarray) -> None:
            self.name, self.array = name, array

        def as_numpy(self) -> np.ndarray:
            return self.array

    class InferenceResponse:
        def __init__(self, output_tensors: list[Tensor]) -> None:
            self.output_tensors = output_tensors

    module.Tensor = Tensor  # type: ignore[attr-defined]
    module.InferenceResponse = InferenceResponse  # type: ignore[attr-defined]
    module.get_input_tensor_by_name = lambda request, name: Tensor(name, request[name])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'triton_python_backend_utils', module)
    return module


def load_capture(spec: Any, capture_dir: Path) -> Any:
    """The model.py the generator writes, initialised the way the server does: the config's
    parameters as a JSON string."""
    config = parse(si.capture_config(spec))
    params = {k: {'string_value': v.string_value} for k, v in config.parameters.items()}
    params['CAPTURE_DIR'] = {'string_value': str(capture_dir)}
    namespace: dict[str, Any] = {'__name__': 'capture_model'}
    exec(compile(si.CAPTURE_MODEL_SOURCE.read_text(), 'model.py', 'exec'), namespace)
    model = namespace['TritonPythonModel']()
    model.initialize(
        {'model_name': spec.capture_name, 'model_config': json.dumps({'parameters': params})}
    )
    return model


def test_the_capture_model_answers_like_the_null_model_and_records_each_row(
    pb_utils: types.ModuleType, tmp_path: Path
) -> None:
    spec = SPEC['null_detector_640']
    capture = si.SPECS[0]
    model = load_capture(capture, tmp_path)
    rng = np.random.default_rng(2)
    first = rng.integers(0, 256, (2, 3, 640, 640), dtype=np.uint8)
    second = first[:1].copy()  # the same content again: a second index line, no second file
    responses = model.execute([{'images': first}, {'images': second}])
    answers = [r.output_tensors[0] for r in responses]
    assert [a.name for a in answers] == ['output0', 'output0']
    assert np.array_equal(answers[0].array, evaluate(spec, first))
    assert answers[1].array.shape == (1, 300, 6)
    root = tmp_path / 'default' / 'capture_detector_640'
    lines = [ln.split('\t') for ln in (root / 'index.txt').read_text().splitlines()]
    digests = [
        hashlib.sha256(first[0].tobytes()).hexdigest(),
        hashlib.sha256(first[1].tobytes()).hexdigest(),
    ]
    assert [(order, digest, shape) for order, digest, shape in lines] == [
        ('0', digests[0], '3x640x640'),
        ('1', digests[1], '3x640x640'),
        ('2', digests[0], '3x640x640'),
    ]
    assert sorted(p.name for p in root.glob('*.bin')) == sorted(f'{d}.bin' for d in set(digests))
    assert (root / f'{digests[1]}.bin').read_bytes() == first[1].tobytes()


def test_a_reloaded_capture_model_continues_the_arrival_order(
    pb_utils: types.ModuleType, tmp_path: Path
) -> None:
    batch = np.zeros((1, 3, 224, 224), np.uint8)
    spec = si.SPECS[2]
    load_capture(spec, tmp_path).execute([{'images': batch}])
    load_capture(spec, tmp_path).execute([{'images': batch}])
    lines = (tmp_path / 'default' / 'capture_encoder_224' / 'index.txt').read_text().splitlines()
    assert [ln.split('\t')[0] for ln in lines] == ['0', '1']


def test_each_arm_is_captured_into_its_own_directory(
    pb_utils: types.ModuleType, tmp_path: Path
) -> None:
    model = load_capture(si.SPECS[2], tmp_path)
    one = np.full((1, 3, 224, 224), 1, np.uint8)
    two = np.full((2, 3, 224, 224), 2, np.uint8)
    (tmp_path / 'ARM').write_text('python\n')
    model.execute([{'images': one}])
    (tmp_path / 'ARM').write_text('native')
    model.execute([{'images': two}])
    model.execute([{'images': one}])

    def count(arm: str) -> int:
        index = tmp_path / arm / 'capture_encoder_224' / 'index.txt'
        return len(index.read_text().splitlines())

    assert (count('python'), count('native')) == (1, 3)
    assert not (tmp_path / 'default').exists()


def test_generate_writes_the_six_models_and_the_graphs_only_when_asked(tmp_path: Path) -> None:
    si.generate(tmp_path / 'plain', onnx=False, detector_instances=2, encoder_instances=1)
    names = sorted(p.name for p in (tmp_path / 'plain').iterdir())
    assert names == sorted([*si.null_names(), *si.capture_names()])
    for name in si.capture_names():
        assert (
            tmp_path / 'plain' / name / '1' / 'model.py'
        ).read_text() == si.CAPTURE_MODEL_SOURCE.read_text()
    assert (tmp_path / 'plain' / 'null_detector_640' / '1').is_dir(), 'the plan goes here'
    si.generate(tmp_path / 'withonnx', onnx=True, detector_instances=4, encoder_instances=2)
    assert sorted(p.name for p in (tmp_path / 'withonnx' / '_onnx').iterdir()) == sorted(
        f'{n}.onnx' for n in si.null_names()
    )
    config = parse((tmp_path / 'withonnx' / 'null_detector_640' / 'config.pbtxt').read_text())
    assert config.instance_group[0].count == 4


def test_plan_commands_print_by_default_and_run_only_when_asked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert si.main(['plan-commands', '--runner', 'definitely-not-installed-trtexec']) == 0
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 3
    assert all(line.startswith('definitely-not-installed-trtexec --onnx=') for line in printed)
    assert si.main(['plan-commands', '--runner', 'definitely-not-installed-trtexec', '--run']) == 1


def test_names_are_printed_for_the_hold_script(capsys: pytest.CaptureFixture[str]) -> None:
    assert si.main(['names', '--kind', 'null']) == 0
    assert json.loads(capsys.readouterr().out) == [
        'null_detector_640',
        'null_encoder_336',
        'null_encoder_224',
    ]


def test_the_detection_count_is_the_fixed_boxes(capsys: pytest.CaptureFixture[str]) -> None:
    assert si.main(['detections']) == 0
    assert capsys.readouterr().out.strip() == str(len(si.detector_rows())) == '5'
