#%%
from IPython import get_ipython
if (ip := get_ipython()) is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")

from attribute.caching import TranscodedModel
import torch
torch.set_grad_enabled(False)

model = TranscodedModel(
    model_name="meta-llama/Llama-3.2-1B",
    transcoder_path="EleutherAI/skip-transcoder-Llama-3.2-1B-131k",
    device="cuda",
)
old_embedding_weight = model.embedding_weight.clone()
old_logit_weight = model.logit_weight.clone()
#%%
layer = 12
# layer = 10
w_dec = model.w_dec(layer, use_skip=True)
#%%
# feature = 20286

# feature = 57243
feature = 86448
token_index = 6
prompt = "The meaning of the word ? is \""
tokenized_prompt = model.tokenizer.encode(prompt)
from loguru import logger
from tqdm import tqdm
token_id = tokenized_prompt[token_index]
strength = 6.0
batch_size = 16
seq_len = 7
emb_vec = w_dec[feature]
emb_vec /= emb_vec.norm()
emb_vec *= strength
new_embedding_weight = old_embedding_weight.clone()
new_embedding_weight[token_id] = emb_vec
model.embedding_weight[:] = new_embedding_weight
# model.embedding_weight[:] = old_embedding_weight
# model.logit_weight.data = old_logit_weight
model.model.lm_head.weight = torch.nn.Parameter(old_logit_weight)
# logit_bias = model.logit_bias
logit_bias = 0.0
top_logits = model.logit_weight @ emb_vec.to(model.logit_weight) + logit_bias
print(model.tokenizer.batch_decode(top_logits.topk(10).indices.tolist()))
tokens = model.tokenizer([prompt] * batch_size, return_tensors="pt").to(model.device).input_ids
try:
    for _ in (bar := tqdm(range(seq_len))):
        logger.disable("attribute")
        og_activations = model(tokens)
        logits = model(tokens, errors_from=og_activations,
                    steer_features={layer: [(token_index, feature, strength)]}
                    ).logits
        base_logits = og_activations.logits
        # logits = logits - (base_logits -logits) * 0.4
        probs = torch.nn.functional.softmax(logits[:, -1], dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_tokens], dim=1)
        bar.set_postfix(text=model.tokenizer.decode(tokens[0].tolist()))
finally:
    logger.enable("attribute")
[seq[len(model.tokenizer.decode(tokenized_prompt)):] for seq in model.tokenizer.batch_decode(tokens.tolist())]
# %%
# %%
