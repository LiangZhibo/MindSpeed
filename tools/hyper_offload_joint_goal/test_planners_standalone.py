"""Exercise pure planning code without importing the Torch runtime package.

This bootstrap does not mock planner behavior or hardware measurements. It
only skips the top-level runtime imports; execution correctness needs separate
Torch and NPU integration tests.
"""
from pathlib import Path
import importlib
import inspect
import sys
import types
import unittest

root = Path(__file__).resolve().parents[2]
for package in ('mindspeed', 'mindspeed.core', 'mindspeed.core.memory',
                'mindspeed.core.memory.hyper_offload'):
    module = types.ModuleType(package)
    module.__path__ = [str(root.joinpath(*package.split('.')))]
    sys.modules[package] = module
test_root = root / 'tests_extend/unit_tests/mindspeed/core/memory/hyper_offload/planning'
suite = unittest.defaultTestLoader.discover(str(test_root), pattern='test_*.py')
for path in sorted(test_root.glob('test_*.py')):
    module = importlib.import_module(path.stem)
    for name, function in inspect.getmembers(module, inspect.isfunction):
        if name.startswith('test_') and not inspect.signature(function).parameters:
            suite.addTest(unittest.FunctionTestCase(function))
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
