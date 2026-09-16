"""Offline integration checks for the original GPT-4 evaluation protocol."""
import json
import hashlib
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / "ToolAlpaca_original"


class OriginalProcessMetricTests(unittest.TestCase):
    def run_script(self, root, source, completion, extra=()):
        client = Mock(return_value=completion)
        utils = types.ModuleType("utils")
        utils.openai_chat_completions = client
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "input.json"
            output_path = Path(directory) / "output.json"
            source_path.write_text(json.dumps(source))
            argv = [str(root / "evaluation.py"), "-api", str(source_path),
                    "-out", str(output_path), "-temp", str(root / "prompts/Evaluation.txt"), *extra]
            with patch.dict(sys.modules, {"utils": utils}), patch.object(sys, "argv", argv):
                error = None
                try:
                    runpy.run_path(str(root / "evaluation.py"), run_name="__main__")
                except RuntimeError as exc:
                    error = exc
            result = json.loads(output_path.read_text())
        return result, client, error

    def source(self):
        return [{"Name": "weather", "NLDocumentation": "weather(city)",
                 "Instructions": ["Weather in Paris?", "Weather in Rome?"],
                 "Golden_Answers": [[{"Action": "weather", "Action_Input": {"city": "Paris"}}], []],
                 "Instances": [{"intermediate_steps": [[["weather", {"city": "Paris"}], "sunny"]],
                                "output": "It is sunny."}, {"intermediate_steps": [], "output": ""}]}]

    def completion(self, text="## Results\nProcess Correctness: Yes\nFinal Response Correctness: No"):
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}

    def test_matches_original_prompt_and_process_denominator(self):
        original, old_client, _ = self.run_script(ORIGINAL, self.source(), self.completion())
        with patch.dict(os.environ, {"OPENAI_DEFAULT_MODEL": "gpt-3.5-turbo"}):
            result, client, error = self.run_script(ROOT, self.source(), [self.completion()])
        self.assertIsNone(error)
        self.assertEqual(client.call_args.args[0][0], old_client.call_args.args[0])
        self.assertEqual(client.call_args.kwargs["model"], "gpt-4-0613")
        self.assertEqual(client.call_args.kwargs["temperature"], 0.2)
        for key in ("num", "error_num", "process", "response", "both"):
            self.assertEqual(result["statistics"][key], original["statistics"][key])
        self.assertEqual(result["statistics"]["primary_metric"], "process_accuracy")
        self.assertEqual(result["statistics"]["process_accuracy"], 0.5)
        self.assertEqual(result["statistics"]["overall_accuracy"], 0)
        self.assertEqual(result["statistics"]["evaluator_sha256"],
                         hashlib.sha256((ROOT / "evaluation.py").read_bytes()).hexdigest())
        self.assertEqual(result["statistics"]["template_sha256"],
                         hashlib.sha256((ROOT / "prompts/Evaluation.txt").read_bytes()).hexdigest())
        self.assertEqual(result["statistics"]["input_sha256"],
                         hashlib.sha256(json.dumps(self.source()).encode()).hexdigest())

    def test_invalid_judge_is_incomplete_and_not_counted_as_uncertain(self):
        result, client, error = self.run_script(ROOT, self.source(), [self.completion("no verdict")])
        self.assertIsNotNone(error)
        self.assertEqual(client.call_count, 3)
        self.assertFalse(result["statistics"]["complete"])
        self.assertIsNone(result["statistics"]["process_accuracy"])
        self.assertEqual(result["statistics"]["process"]["Uncertain"], 0)
        self.assertEqual(result["statistics"]["num"], 1)  # Only the failed rollout is scored.
        self.assertEqual(result["statistics"]["judge_errors"][0]["id"], 0)

    def test_explicit_uncertain_is_valid(self):
        result, client, error = self.run_script(
            ROOT, self.source(), [self.completion(
                "## Results\nProcess Correctness: Uncertain\nFinal Response Correctness: Uncertain")])
        self.assertIsNone(error)
        self.assertTrue(result["statistics"]["complete"])
        self.assertEqual(client.call_count, 1)
        self.assertEqual(result["statistics"]["process"]["Uncertain"], 1)

    def test_partial_trajectory_without_output_is_judged_with_empty_response(self):
        source = self.source()
        instance = source[0]["Instances"][0]
        del instance["output"]
        instance["error"] = "Could not parse LLM output"
        result, client, error = self.run_script(ROOT, source, [self.completion()])
        self.assertIsNone(error)
        prompt = client.call_args.args[0][0][0]["content"]
        self.assertIn("1. Function: weather\nParameters: {'city': 'Paris'}\nRetruns: sunny\n2. Final Response: ", prompt)
        self.assertNotIn("Could not parse LLM output", prompt)
        self.assertNotIn("It is sunny.", prompt)
        self.assertTrue(result["statistics"]["complete"])
        self.assertEqual(result["statistics"]["process_accuracy"], 0.5)

    def test_gpu_memory_error_blocks_scoring_even_with_saved_tool_steps(self):
        source = self.source()
        source[0]["Instances"][0]["error"] = "torch.OutOfMemoryError: CUDA out of memory"
        result, client, error = self.run_script(ROOT, source, [self.completion()])
        self.assertIsNotNone(error)
        client.assert_not_called()
        self.assertFalse(result["statistics"]["complete"])
        self.assertIsNone(result["statistics"]["process_accuracy"])
        self.assertEqual(result["statistics"]["num"], 0)
        self.assertEqual(result["statistics"]["rollout_errors"][0]["id"], 0)


if __name__ == "__main__":
    unittest.main()
