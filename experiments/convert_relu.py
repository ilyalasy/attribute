#%%
from IPython import get_ipython
import re
import json
import torch
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from sparsify import SparseCoderConfig
from attribute.caching import TranscodedModel
from safetensors.torch import load_file, save_file
if (ip := get_ipython()) is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
torch.set_grad_enabled(False)
# %%
model_name = "meta-llama/Llama-3.2-1B"
repo_id = "mntss/skip-transcoder-Llama-3.2-1B-131k-nobos"
branch = "new-training"
model_hf = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": "cpu"})
transcoder_path = Path(snapshot_download(repo_id, revision=branch))
output_path = Path("results/llama-mntss-relu")
# %%
for st_file in tqdm(list(transcoder_path.glob("*.safetensors"))):
    layer_idx = int(re.match(r"layer_(\d+)\.safetensors", st_file.name).group(1))
    state_dict = load_file(st_file, device="cpu")
    state_dict["W_dec"] = state_dict["W_dec"].T.contiguous()
    state_dict["encoder.bias"] = state_dict.pop("b_enc")
    state_dict["encoder.weight"] = state_dict.pop("W_enc")
    state_dict["W_skip"] = state_dict["W_skip"]
    layer_path = output_path / f"layers.{layer_idx}.mlp"
    layer_path.mkdir(parents=True, exist_ok=True)
    cfg_path = layer_path / "cfg.json"
    weights_path = layer_path / "sae.safetensors"
    save_file(state_dict, weights_path)
    num_latents, d_in = state_dict["W_dec"].shape
    config = SparseCoderConfig(
        k=64,
        num_latents=num_latents,
        transcode=True,
        skip_connection=True,
    )
    json.dump(config.to_dict() | dict(d_in=d_in), cfg_path.open("w"))
# %%
model = TranscodedModel(
    model_name,
    output_path,
    device="cuda",
    pre_ln_hook=True,
)
#%%
model("In another moment, down went Alice after it, never once considering how in the world she was to get out again.")
# %%
