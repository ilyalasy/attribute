import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from jaxtyping import Array, Float, Int
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt2 import GPT2PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import (
    eager_attention_forward as gpt2_eager_attention_forward,
)
from transformers.models.gpt_neo import GPTNeoPreTrainedModel
from transformers.models.llama import LlamaPreTrainedModel

from .utils import repeat_kv

GPT2Like = GPT2PreTrainedModel | GPTNeoPreTrainedModel


@dataclass
class MLPOutputs:
    ln_factor: Float[Array, "batch seq_len hidden_size"]
    activation: Float[Array, "batch seq_len k"]
    source_activation: Float[Array, "batch seq_len k"]
    location: Int[Array, "batch seq_len k"]
    error: Float[Array, "batch seq_len hidden_size"]
    source_error: Float[Array, "batch seq_len hidden_size"]


@dataclass
class AttentionOutputs:
    ln_factor: Float[Array, "batch seq_len hidden_size"]
    attn_values: Float[Array, "batch num_attention_heads seq_len head_dim"]
    attn_patterns: Float[Array, "batch num_attention_heads seq_len seq_len"]


@dataclass
class TranscodedOutputs:
    input_ids: Int[Array, "batch seq_len"]
    mlp_outputs: dict[int, MLPOutputs]
    attn_outputs: dict[int, AttentionOutputs]
    last_layer_activations: Float[Array, "batch seq_len hidden_size"]
    first_layer_activations: Float[Array, "batch seq_len hidden_size"]
    logits: Float[Array, "batch seq_len vocab_size"]

    @property
    def seq_len(self):
        return self.input_ids.shape[1]

    @property
    def batch_size(self):
        return self.input_ids.shape[0]


class TranscodedModel(object):
    def __init__(
        self,
        model_name: str | os.PathLike,
        transcoder_path: os.PathLike,
        hookpoint_fn=None,
        device="cuda",
    ):
        logger.info(f"Loading model {model_name} on device {device}")
        self.device = device
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model
        self.tokenizer = tokenizer

        if transcoder_path is None:
            return
        if hookpoint_fn is None:

            def hookpoint_fn(hookpoint):
                if isinstance(model, LlamaPreTrainedModel):
                    return hookpoint.replace("model.layers.", "layers.")
                elif isinstance(model, GPT2Like):
                    return hookpoint.replace("transformer.h.", "h.")
                else:
                    logger.warning(
                        f"Unknown model type: {type(model)}. Using default hookpoint."
                    )
                    return hookpoint

        self.hookpoints_mlp = [
            f"{self.layer_prefix}.{i}.mlp" for i in range(self.num_layers)
        ]
        self.temp_hookpoints_mlp = [
            hookpoint_fn(hookpoint) for hookpoint in self.hookpoints_mlp
        ]

        logger.info(f"Loading transcoders from {transcoder_path}")
        transcoder_path = Path(transcoder_path)

        clt_cfg_path = os.path.join(transcoder_path, "cfg.json")
        clt_ckpt_path = os.path.join(transcoder_path, "clt_checkpoint_latest.pt")

        with open(clt_cfg_path, "r") as f:
            config_dict = json.load(f)
        clt_config = CLTConfig(**config_dict)

        # Create and load model
        clt = CrossLayerTranscoder(clt_config, process_group=None, device=device)
        clt.load_state_dict(torch.load(clt_ckpt_path, map_location=device))

        self.hookpoints_layer = [
            f"{self.layer_prefix}.{i}" for i in range(self.num_layers)
        ]
        self.hookpoints_ln = [
            f"{self.layer_prefix}.{i}.{self.mlp_layernorm_name}"
            for i in range(self.num_layers)
        ]
        self.hookpoints_attn_ln = [
            f"{self.layer_prefix}.{i}.{self.attn_layernorm_name}"
            for i in range(self.num_layers)
        ]
        self.name_to_module = {
            name: model.get_submodule(name)
            for name in self.hookpoints_layer
            + self.hookpoints_mlp
            + self.hookpoints_ln
            + self.hookpoints_attn_ln
        }
        self.name_to_index = {
            k: i
            for arr in [self.hookpoints_layer, self.hookpoints_mlp, self.hookpoints_ln]
            for i, k in enumerate(arr)
        }
        self.module_to_name = {v: k for k, v in self.name_to_module.items()}

    def clear_hooks(self):
        for mod in self.model.modules():
            mod._forward_hooks = OrderedDict()

    @property
    def num_layers(self):
        return self.model.config.num_hidden_layers

    @property
    def repeat_kv(self):
        if not hasattr(self.model.config, "num_key_value_heads"):
            return 1
        return (
            self.model.config.num_attention_heads
            // self.model.config.num_key_value_heads
        )

    def __call__(
        self,
        prompt: str | torch.Tensor,
        mask_features: dict[int, list[int]] = {},
        errors_from: TranscodedOutputs | None = None,
        no_error: bool = False,
    ) -> TranscodedOutputs:
        if isinstance(prompt, str):
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(
                self.device
            )
            logger.info(
                f"Tokenized prompt: {[self.decode_token(i) for i in tokenized_prompt.input_ids[0]]}"
            )
        elif isinstance(prompt, torch.Tensor):
            tokenized_prompt = SimpleNamespace(input_ids=prompt.to(self.device))
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        self.clear_hooks()

        attn_values = {}
        first_ln = {}

        def ln_record_hook(module, input, output, save_dict):
            mean = input[0].mean(dim=-1, keepdim=True).detach()
            var = input[0].var(dim=-1, keepdim=True)
            multiplier = (1 / torch.sqrt(var + module.eps).detach()) * module.weight
            save_dict[self.module_to_name[module]] = multiplier
            return (input[0] - mean) * multiplier + module.bias

        def get_attention_values_hook(module, input, output):
            # this hook is in the layer
            residual = input[0]
            layer_name = self.module_to_name[module]
            layer_normed = getattr(module, self.attn_layernorm_name)(residual)

            # stuff related to attention
            input_shape = layer_normed.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            layer_idx = self.name_to_index[layer_name]
            value_states = (
                self.project_v(layer_idx, layer_normed)
                .view(hidden_shape)
                .transpose(1, 2)
            )
            values = repeat_kv(value_states, self.repeat_kv)
            attn_values[layer_name] = values

        for hookpoint in self.hookpoints_layer:
            self.name_to_module[hookpoint].register_forward_hook(
                get_attention_values_hook
            )
            ln = getattr(self.name_to_module[hookpoint], self.attn_layernorm_name)
            ln.register_forward_hook(partial(ln_record_hook, save_dict=first_ln))
            self.freeze_attention_pattern(hookpoint)

        second_ln = {}
        for hookpoint in self.hookpoints_ln:
            self.name_to_module[hookpoint].register_forward_hook(
                partial(ln_record_hook, save_dict=second_ln)
            )

        errors = {}
        activations = {}
        reconstructions = {}

        def get_mlp_hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]

            module_name = self.module_to_name[module]

            layer_idx = self.name_to_index[module_name]
            latent_acts = self.clt.encode(input, layer_idx)

            masked_features = mask_features.get(layer_idx, [])
            if masked_features:
                # TODO: CHECK THIS
                mask = torch.ones_like(latent_acts)
                mask[masked_features] = 0
                latent_acts = latent_acts * mask.float()

            activations[layer_idx] = latent_acts

            relevant_activations = {
                k: v for k, v in activations.items() if k <= layer_idx and v.numel() > 0
            }
            reconstructions[layer_idx] = self.clt.decode(
                relevant_activations, layer_idx
            )

            diff = output - reconstructions[layer_idx]
            if no_error:
                error = torch.zeros_like(output)
            elif errors_from is None:
                error = diff
            else:
                error = errors_from.mlp_outputs[layer_idx].error

            error = error.clone()
            error.detach_()
            error.requires_grad_(True)
            logger.info(f"Layer {module_name} error: {diff.norm() / output.norm()}")

            result = (reconstructions[layer_idx] + error).to(output)
            errors[module_name] = error
            return result

        for hookpoint in self.hookpoints_mlp:
            self.name_to_module[hookpoint].register_forward_hook(get_mlp_hook)

        outputs = self.model(
            input_ids=tokenized_prompt.input_ids,
            #  attention_mask=tokenized_prompt.attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
        self.clear_hooks()

        attention_patterns = outputs.attentions
        logits = outputs.logits

        logger.info("Top last token logits:")
        for index in logits[0, -1].topk(10).indices:
            logger.info(f"{self.decode_token(index)}: {logits[0, -1][index].item()}")

        first_layer_activations = outputs.hidden_states[0]
        if first_layer_activations.requires_grad:
            first_layer_activations.retain_grad()

        last_layer_activations = outputs.hidden_states[-1]
        if last_layer_activations.requires_grad:
            last_layer_activations.retain_grad()

        mlp_outputs = {}
        for i in range(self.num_layers):
            source_activation = activations[i].clone()
            indices = source_activation.nonzero()
            source_activation = source_activation[indices]
            source_activation.detach_()
            source_activation.requires_grad_(True)

            target_activation = activations[i].clone()
            indices = target_activation.nonzero()
            target_activation = target_activation[indices]

            mlp_outputs[i] = MLPOutputs(
                ln_factor=second_ln[self.hookpoints_ln[i]],
                activation=target_activation,
                source_activation=source_activation,
                location=indices,
                error=errors[self.hookpoints_mlp[i]],
                source_error=errors[self.hookpoints_mlp[i]],
            )
        attn_outputs = {}
        for i in range(self.num_layers):
            attn_outputs[i] = AttentionOutputs(
                ln_factor=first_ln[self.hookpoints_attn_ln[i]],
                attn_values=attn_values[self.hookpoints_layer[i]],
                attn_patterns=attention_patterns[i],
            )

        transcoded_outputs = TranscodedOutputs(
            input_ids=tokenized_prompt.input_ids,
            mlp_outputs=mlp_outputs,
            attn_outputs=attn_outputs,
            first_layer_activations=first_layer_activations,
            last_layer_activations=last_layer_activations,
            logits=logits,
        )

        return transcoded_outputs

    @property
    def layer_prefix(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return "model.layers"
        elif isinstance(self.model, GPT2Like):
            return "transformer.h"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def attn_layernorm_name(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return "input_layernorm"
        elif isinstance(self.model, GPT2Like):
            return "ln_1"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def mlp_layernorm_name(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return "post_attention_layernorm"
        elif isinstance(self.model, GPT2Like):
            return "ln_2"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def mlp_in_proj_name(self):
        if isinstance(self.model, GPT2Like):
            return "c_proj"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def embedding_weight(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return self.model.model.embed_tokens.weight
        elif isinstance(self.model, GPT2Like):
            return self.model.transformer.wte.weight
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def parallel_attn(self):
        return isinstance(self.model, GPTNeoPreTrainedModel)

    @property
    def final_ln(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return self.model.model.norm
        elif isinstance(self.model, GPT2Like):
            return self.model.transformer.ln_f
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def logit_weight(self):
        if isinstance(self.model, LlamaPreTrainedModel):
            return self.model.lm_head.weight * self.final_ln.weight
        elif isinstance(self.model, GPT2Like):
            return self.model.lm_head.weight * self.final_ln.weight
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    @property
    def logit_bias(self):
        bias = self.model.lm_head.bias
        if bias is None:
            bias = 0
        return bias + self.logit_weight @ self.final_ln.bias

    @property
    def vocab_size(self):
        return self.model.config.vocab_size

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def num_attention_heads(self):
        return self.model.config.num_attention_heads

    @property
    def head_dim(self):
        return self.model.config.hidden_size // self.model.config.num_attention_heads

    @torch.no_grad()
    @torch.autocast("cuda")
    def w_dec(
        self, layer_idx: int, target_layer_idx: int | None = None
    ) -> Float[Array, "features hidden_size"]:
        if target_layer_idx is None:
            logger.warning("Summing decoder weights because target_layer_idx is None")
            target_layer_idx = layer_idx
            weight_combined = torch.zeros(
                (
                    self.w_dec(layer_idx, layer_idx).shape[0],
                    self.hidden_size,
                ),
                device=self.device,
                dtype=torch.float32,
            )
            for target_layer_idx in range(layer_idx, self.num_layers):
                weight_combined += self.w_dec(layer_idx, target_layer_idx)
            return weight_combined
        assert target_layer_idx >= layer_idx
        decoder = self.clt.decoders[f"{layer_idx}->{target_layer_idx}"]
        return decoder.weight

    def w_skip(
        self, layer_idx: int, target_layer_idx: int | None = None
    ) -> Float[Array, "hidden_size hidden_size"]:
        raise NotImplementedError()

    def w_enc(self, layer_idx: int) -> Float[Array, "features hidden_size"]:
        return self.clt.encoders[layer_idx].weight

    def attn(self, layer_idx: int) -> torch.nn.Module:
        layer = self.model.get_submodule(self.layer_prefix)[layer_idx]
        if isinstance(self.model, LlamaPreTrainedModel):
            return layer.self_attn
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            return layer.attn.attention
        elif isinstance(self.model, GPT2PreTrainedModel):
            return layer.attn
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def attn_output(
        self, layer_idx: int
    ) -> Float[Array, "hidden_size num_attention_heads head_dim"]:
        if isinstance(self.model, LlamaPreTrainedModel):
            w_o = self.attn(layer_idx).o_proj.weight
        elif isinstance(self.model, GPTNeoPreTrainedModel):
            w_o = self.attn(layer_idx).out_proj.weight
        elif isinstance(self.model, GPT2PreTrainedModel):
            w_o = self.attn(layer_idx).c_proj.weight.T
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        return w_o.reshape(self.hidden_size, self.num_attention_heads, self.head_dim)

    def attn_value(
        self, layer_idx: int
    ) -> Float[Array, "num_attention_heads head_dim hidden_size"]:
        if not isinstance(self.model, GPT2PreTrainedModel):
            w_v = self.attn(layer_idx).v_proj.weight
        else:
            w_q, w_k, w_v = torch.split(
                self.attn(layer_idx).c_attn.weight, self.hidden_size, dim=1
            )
            w_v = w_v.T
        w_v = torch.repeat_interleave(w_v, self.repeat_kv, dim=0)
        return w_v.reshape(self.num_attention_heads, self.head_dim, self.hidden_size)

    def project_v(
        self, layer_idx: int, layer_normed: Float[Array, "batch seq_len hidden_size"]
    ) -> Float[Array, "batch seq_len num_attention_heads head_dim"]:
        if not isinstance(self.model, GPT2PreTrainedModel):
            return self.attn(layer_idx).v_proj(layer_normed)
        else:
            projected = self.attn(layer_idx).c_attn(layer_normed)
            q, k, v = torch.split(projected, self.hidden_size, dim=-1)
            return v

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def freeze_attention_pattern(self, hookpoint: str):
        attn = self.attn(self.name_to_index[hookpoint])
        original_implementation = attn.config._attn_implementation
        if original_implementation.startswith("no_grad_"):
            return
        impl_name = original_implementation
        if impl_name == "eager":
            impl_name = f"eager_{type(attn).__name__}"
        no_grad_name = "no_grad_" + impl_name
        if no_grad_name not in ALL_ATTENTION_FUNCTIONS:

            @torch.compile
            def new_attention_forward(
                module, query, key, value, attention_mask, *, impl_fn, **kwargs
            ):
                query = query.detach()
                key = key.detach()
                _, attn_weights = impl_fn(
                    module, query, key, value, attention_mask, **kwargs
                )
                attn_weights = attn_weights.detach()

                attn_output = torch.matmul(attn_weights, value)
                attn_output = attn_output.transpose(1, 2)

                return attn_output, attn_weights

            if isinstance(self.model, GPT2PreTrainedModel):
                eager_attention_forward = gpt2_eager_attention_forward
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")
            ALL_ATTENTION_FUNCTIONS[no_grad_name] = partial(
                new_attention_forward,
                impl_fn=ALL_ATTENTION_FUNCTIONS[original_implementation]
                if original_implementation != "eager"
                else eager_attention_forward,
            )
        attn.config._attn_implementation = no_grad_name
