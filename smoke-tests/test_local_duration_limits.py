"""Exercise API duration boundaries without loading GPU packages or launching jobs.

Compile the actual endpoint functions with mocked decoding/job dependencies. This
checks request acceptance and rejection; it does not test long inference quality.
Run: python smoke-tests/test_local_duration_limits.py
"""
import ast
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_functions(relative_path, names, namespace, optional_names=()):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (node.name in names or node.name in optional_names):
            node.decorator_list = []
            selected.append(node)
    assert names <= {node.name for node in selected}, "endpoint renamed or missing"
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(ROOT / relative_path), "exec"), namespace)


def sa3_default_limit():
    tree = ast.parse((ROOT / "services/sa3/api.py").read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "MAX_DURATION" for t in node.targets))
    return eval(compile(ast.Expression(assignment.value), "api.py", "eval"), {"os": os})


class LocalDurationLimitsTest(unittest.TestCase):
    def sa3(self, source_seconds, continuation_seconds=0):
        with patch.dict(os.environ, {}, clear=True):
            maximum = sa3_default_limit()
        data = {"prompt": "piano", "duration": source_seconds,
                "continuation_seconds": continuation_seconds}
        worker = Mock()
        namespace = dict(
            MAX_DURATION=maximum, DEFAULT_DURATION=30, DEFAULT_STEPS=8, DEFAULT_CFG=1,
            DEFAULT_CONTINUATION_SECONDS=8, CONTINUE_TAIL_PAD=6, CONTINUE_TAIL_PAD_MAX=60,
            CONTINUE_TAIL_MODE="regen_past", OUTPUT_SAMPLE_RATE=44100,
            VALID_SHIFTS={"default"}, VALID_CONTINUATION_MODES={"inpaint", "latent_prefix"},
            get_json_body=lambda: data, cleanup_old_sessions=lambda: None,
            request=SimpleNamespace(path="/duration-test"),
            jsonify=lambda value: value, create_session=Mock(return_value="test-job"),
            generation_worker=Mock(), threading=SimpleNamespace(Thread=worker),
            decode_audio_data=lambda _: (44100, SimpleNamespace(shape=(2, round(source_seconds*44100)))),
            parse_tail_pad=lambda *args: 6,
            loudness_params=lambda _: dict(latent_target_std=None, latent_rescale=0,
                latent_adapt_min=0, latent_adapt_max=1, limiter_knee=1),
            continuation_splice_params=lambda _: dict(splice_xfade=0, mask_overlap=0),
            continuation_mask_bounds=lambda source, overlap: (0, source),
            common_params=lambda data, duration: dict(prompt="piano", steps=8, seed=1,
                duration=duration, sampler_type="pingpong", mask_overlap=0,
                splice_source=False, splice_xfade=0, splice_gain_match=False),
        )
        load_functions("services/sa3/api.py",
            {"parse_float", "parse_int", "validate_common", "transform", "continue_audio"}, namespace,
            optional_names={"reject_bad_request"})
        return namespace, worker

    def test_sa3_default_and_explicit_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(sa3_default_limit(), 380)
        with patch.dict(os.environ, {"SA3_MAX_DURATION": "275"}):
            self.assertEqual(sa3_default_limit(), 275)

    def test_sa3_generate_boundary(self):
        namespace, worker = self.sa3(380)
        self.assertEqual(namespace["validate_common"]({"prompt": "piano", "duration": 380}), [])
        self.assertTrue(namespace["validate_common"]({"prompt": "piano", "duration": 380.01}))
        worker.assert_not_called()

    def test_sa3_transform_boundary(self):
        namespace, worker = self.sa3(380)
        self.assertTrue(namespace["transform"]()["success"])
        worker.assert_called_once()
        namespace, worker = self.sa3(380.01)
        response, status = namespace["transform"]()
        self.assertEqual(status, 400)
        self.assertIn("380", response["error"])
        worker.assert_not_called()

    def test_sa3_continuation_total_boundary(self):
        namespace, worker = self.sa3(330, 50)
        response = namespace["continue_audio"]()
        self.assertEqual(response["total_duration"], 380)
        worker.assert_called_once()
        namespace, worker = self.sa3(330, 51)
        response, status = namespace["continue_audio"]()
        self.assertEqual(status, 400)
        worker.assert_not_called()

    def test_carey_complete_boundary(self):
        class RequestError(Exception):
            def __init__(self, status, message):
                self.status, self.message = status, message

        async def generation(*args):
            pass

        submit = Mock(side_effect=lambda coroutine: coroutine.close())
        namespace = dict(HTTPException=RequestError, _validate_lora_request=Mock(),
            _cleanup_old_jobs=Mock(), uuid4=lambda: "test-job", Job=lambda **kw: SimpleNamespace(**kw),
            _jobs={}, _run_generation=generation, JSONResponse=lambda value: value,
            asyncio=SimpleNamespace(create_task=submit))
        load_functions("services/carey/carey_wrapper.py", {"complete_submit"}, namespace)
        request = SimpleNamespace(audio_duration=380, bpm=120, audio_format="wav")
        response = asyncio.run(namespace["complete_submit"](request))
        self.assertTrue(response["success"])
        self.assertEqual(namespace["_jobs"]["test-job"].target_duration, 380)
        submit.assert_called_once()
        submit.reset_mock()
        request.audio_duration = 380.01
        with self.assertRaises(RequestError) as error:
            asyncio.run(namespace["complete_submit"](request))
        self.assertEqual(error.exception.status, 400)
        submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
