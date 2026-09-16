"""Check model isolation and transport failure accounting without API calls."""
import asyncio
import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]


class SimulatorConfigurationTests(unittest.TestCase):
    def load(self, *extra):
        argv = ['simulator.py', '-api', str(ROOT / 'tool_unlearn/data/forget_eval_100.json'),
                '-temp', str(ROOT.parent / 'ToolAlpaca_original/prompts/Simulator.txt'), *extra]
        with patch.dict(os.environ, {'OPENAI_DEFAULT_MODEL': 'gpt-5.6-luna'}, clear=True), \
                patch.object(sys, 'argv', argv):
            return runpy.run_path(str(ROOT / 'instance_generation/simulator.py'))

    def test_simulator_does_not_inherit_judge_model(self):
        module = self.load()
        generate = module['_get_response']
        completion = AsyncMock(return_value={
            'usage': {'total_tokens': 12},
            'choices': [{'message': {'content': 'Status Code: 200\nResponse: {"ok": true}'}}]})
        with patch.dict(generate.__globals__, {'async_openai_chat_completions': completion}):
            status, response, _, _ = asyncio.run(generate([{'role': 'user', 'content': 'request'}]))
        self.assertEqual(status, 200)
        self.assertEqual(response, {'ok': True})
        self.assertEqual(completion.call_args.kwargs, {'model': 'gpt-3.5-turbo', 'temperature': 0.5})

    def test_api_failure_is_visible_to_runner(self):
        module = self.load('--model', 'gpt-3.5-turbo')
        generate = module['_get_response']
        with patch.dict(generate.__globals__, {
                'async_openai_chat_completions': AsyncMock(side_effect=RuntimeError('no credits'))}):
            with self.assertRaisesRegex(RuntimeError, 'no credits'):
                asyncio.run(generate([]))
            self.assertEqual(module['simulator_status'](), {'model': 'gpt-3.5-turbo', 'failures': 1})


if __name__ == '__main__':
    unittest.main()
