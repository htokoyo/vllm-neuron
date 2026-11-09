# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vllm-neuron copy of the public nkilib MoE CTE kernel (``moe_cte_dyn``).

``bwmm_shard_on_block.py`` starts from ``nkilib/core/moe/moe_cte/bwmm_shard_on_block.py`` of the public
nkilib shipped with Neuron SDK 2.33 (neuronx-cc 2.28.7631.0; sha256
f2a730dd3e19008d68489989101e17ad684245b3d6e0811118bb567bedc32b7a), under the Apache License 2.0 above.
Our changes (vllm-neuron, Qwen4-Exp): the static + dynamic block loop (``n_static_blocks`` /
``conditions``), the FP8 row-scaled double_row path and the weight-skip mask fix; see that file's
docstring. The rest of nkilib is imported from the installed package.
"""
