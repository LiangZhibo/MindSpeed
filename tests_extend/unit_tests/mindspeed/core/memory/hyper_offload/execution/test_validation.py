# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for stable HyperOffload replay signatures."""

import unittest

import torch

from mindspeed.core.memory.hyper_offload.execution.validation import signature


class TestReplaySignature(unittest.TestCase):
    def test_script_object_addresses_do_not_change_replay_signature(self):
        first = torch.ops.profiler._record_function_enter_new("hyper-offload-test")
        second = torch.ops.profiler._record_function_enter_new("hyper-offload-test")
        try:
            self.assertNotEqual(repr(first), repr(second))
            self.assertEqual(signature((first,)), signature((second,)))
        finally:
            torch.ops.profiler._record_function_exit._RecordFunction(first)
            torch.ops.profiler._record_function_exit._RecordFunction(second)

    def test_python_scalar_values_remain_part_of_replay_signature(self):
        self.assertNotEqual(signature((1,)), signature((2,)))


if __name__ == "__main__":
    unittest.main()
