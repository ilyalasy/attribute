#%%
from huggingface_hub import hf_hub_download
import json
from argparse import Namespace
import torch
from transformers import GPT2Config, AutoTokenizer, AutoModelForCausalLM
import einops

model_name = "NeelNanda/GELU_4L512W_C4_Code"
tokenizer_name = "NeelNanda/gpt-neox-tokenizer-digits"
config_path = hf_hub_download(
    repo_id=model_name,
    filename="config.json",
)
og_config = Namespace(**json.load(open(config_path)))
checkpoint_path = hf_hub_download(
    repo_id=model_name,
    filename="model_final.pth",
)
og_state_dict = torch.load(checkpoint_path, map_location="cpu")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#%%
config = GPT2Config(
    **(vars(GPT2Config.from_pretrained("gpt2")) |
    dict(
        vocab_size=og_config.d_vocab,
        n_positions=og_config.n_ctx,
        n_embd=og_config.d_model,
        n_layer=og_config.n_layers,
        n_head=og_config.n_heads,
        n_inner=og_config.d_mlp,
        activation_function=og_config.act_fn,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=og_config.ln_eps,
        # initializer_range=0.0,
        use_cache=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        scale_attn_weights=False,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        tie_word_embeddings=False,
    ))
)
model = AutoModelForCausalLM.from_config(config).cpu()
# # %%
state_dict = model.state_dict()
new_state_dict = {}
for k in state_dict.keys():
    og_k = k
    k = k.replace("transformer.", "")
    k = k.replace("h.", "blocks.")
    k = k.replace("unembed.W_U", "lm_head.weight")
    k = k.replace("wte.weight", "embed.W_E")
    k = k.replace("wpe.weight", "pos_embed.W_pos")
    k = k.replace("ln_f.weight", "ln_final.w")
    k = k.replace("ln_f.bias", "ln_final.b")
    k = k.replace("lm_head.weight", "unembed.W_U")
    k = k.replace("ln_1.weight", "ln1.w")
    k = k.replace("ln_1.bias", "ln1.b")
    k = k.replace("ln_2.weight", "ln2.w")
    k = k.replace("ln_2.bias", "ln2.b")
    k = k.replace("attn.c_proj.weight", "attn.W_O")
    k = k.replace("attn.c_proj.bias", "attn.b_O")
    k = k.replace("mlp.c_fc.weight", "mlp.W_in")
    k = k.replace("mlp.c_fc.bias", "mlp.b_in")
    k = k.replace("mlp.c_proj.weight", "mlp.W_out")
    k = k.replace("mlp.c_proj.bias", "mlp.b_out")
    # problems: c_attn.weight, c_attn.bias, c_proj.weight
    if "c_attn.weight" in og_k:
        combined = []
        for w_letter in "QKV":
            weight = og_state_dict[k.replace("attn.c_attn.weight", f"attn.W_{w_letter}")]
            n_heads, input_dim, head_dim = weight.shape
            weight = einops.rearrange(weight, "n_heads input_dim head_dim -> input_dim (n_heads head_dim)")
            combined.append(weight)
        combined = torch.cat(combined, dim=1)
        new_state_dict[og_k] = combined
    elif "c_attn.bias" in og_k:
        combined = []
        for w_letter in "QKV":
            bias = og_state_dict[k.replace("attn.c_attn.bias", f"attn.b_{w_letter}")]
            n_heads, head_dim = bias.shape
            bias = einops.rearrange(bias, "n_heads head_dim -> (n_heads head_dim)")
            combined.append(bias)
        combined = torch.cat(combined, dim=0)
        new_state_dict[og_k] = combined
    else:
        v = og_state_dict[k]
        if "attn.c_proj.weight" in og_k:
            n_heads, head_dim, output_dim = v.shape
            v = einops.rearrange(v, "n_heads head_dim output_dim -> (n_heads head_dim) output_dim")
        if og_k == "lm_head.weight":
            v = v.T
        new_state_dict[og_k] = v
model.eval()
model.load_state_dict(new_state_dict, assign=False)
#%%
# Create input prompt
prompt = "<|BOS|>In another moment, down went Alice after it,"

# Encode the prompt text to tokens
tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")

# Generate continuation with some reasonable parameters
with torch.no_grad():
    outputs = model.generate(
        tokens,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id
    )

# Decode the generated tokens back to text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
#%%
model.push_to_hub("nev/GELU_4L512W_C4_Code")
tokenizer.push_to_hub("nev/GELU_4L512W_C4_Code")
#%%
