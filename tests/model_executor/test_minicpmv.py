# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.models.minicpmv import MiniCPMV2_6
from vllm.model_executor.models.minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
)

pytestmark = pytest.mark.skip_global_cleanup


def make_model() -> MiniCPMV2_6:
    return object.__new__(MiniCPMV2_6)


def make_model_4_6() -> MiniCPMV4_6ForConditionalGeneration:
    model = object.__new__(MiniCPMV4_6ForConditionalGeneration)
    model._process_vision_input = lambda vision_input, use_vit_merger=None: [
        vision_input["image_embeds"]
    ]
    return model


def test_video_embeds_reach_the_vision_parser():
    embeds = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)

    modalities = make_model()._parse_and_validate_multimodal_inputs(video_embeds=embeds)

    video_input = modalities["videos"]
    assert video_input is not None
    assert video_input["type"] == "image_embeds"
    assert torch.equal(video_input["image_embeds"], embeds)


def test_video_embeds_are_embedded_by_4_6():
    embeds = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)

    embeddings = make_model_4_6().embed_multimodal(video_embeds=embeds)

    assert len(embeddings) == 1
    assert torch.equal(embeddings[0], embeds)


def test_image_and_video_embeds_stay_in_their_own_modality():
    image_embeds = torch.zeros(1, 4, 8)
    video_embeds = torch.ones(1, 4, 8)

    modalities = make_model()._parse_and_validate_multimodal_inputs(
        image_embeds=image_embeds,
        video_embeds=video_embeds,
    )

    assert torch.equal(modalities["images"]["image_embeds"], image_embeds)
    assert torch.equal(modalities["videos"]["image_embeds"], video_embeds)


# `quantization_config` from openbmb/MiniCPM-V-4.6-{,Thinking-}{GPTQ,AWQ}, which all
# store the vision tower in bf16. Neither exclusion list matches it under vLLM's
# names: GPTQ's entries are block-relative and also hit `vpm.*`, and AWQ's
# `model.vision_tower` is not renamed to `vpm` by `hf_to_vllm_mapper`.
@pytest.mark.parametrize(
    "quant_config",
    [
        AutoGPTQConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "damp_percent": 0.01,
                "desc_act": False,
                "static_groups": False,
                "sym": True,
                "true_sequential": True,
                "quant_method": "gptq",
                "checkpoint_format": "gptq",
                "modules_in_block_to_quantize": [
                    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                    ["self_attn.o_proj"],
                    ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
                    ["linear_attn.out_proj"],
                    ["mlp.gate_proj", "mlp.up_proj"],
                    ["mlp.down_proj"],
                ],
            }
        ),
        AutoAWQConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "modules_to_not_convert": [
                    "in_proj_b",
                    "in_proj_a",
                    "model.vision_tower",
                    "model.merger",
                    "lm_head",
                ],
                "quant_method": "awq",
                "version": "gemm",
                "zero_point": True,
            }
        ),
    ],
    ids=["gptq", "awq"],
)
def test_4_6_vision_tower_skips_checkpoint_quantization(quant_config):
    assert make_model_4_6()._maybe_ignore_quant_config(quant_config) is None
