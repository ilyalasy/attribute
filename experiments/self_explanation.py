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
from collections import OrderedDict
from loguru import logger
from tqdm import tqdm
# layer, feature = 10, 20286
# layer, feature = 12, 57243
layer, feature = 12, 86448
w_dec = model.w_dec(layer, use_skip=True)
# w_dec = model.w_enc(layer)
START_SPECIAL = 128011
# token_index = 6
prompt = "ONLINE DICTIONARY\nThe meaning of the word ? is \""
tokenized_prompt = model.tokenizer.encode(prompt)
token_index = tokenized_prompt.index(949)
tokenized_prompt[token_index] = START_SPECIAL
token_id = tokenized_prompt[token_index]
# strength = 4.0
min_strength = 4.0
max_strength = 6.0
batch_size = 16
seq_len = 7
emb_vec = w_dec[feature]
emb_vec /= emb_vec.norm()
strengths = torch.linspace(min_strength, max_strength, batch_size, device=model.device)
emb_vec = emb_vec[None, :] * strengths[:, None]
# new_embedding_weight = old_embedding_weight.clone()
# new_embedding_weight[token_id:token_id+batch_size] = emb_vec
# model.embedding_weight[:] = new_embedding_weight

# model.embedding_weight[:] = old_embedding_weight
# model.logit_weight.data = old_logit_weight
model.model.lm_head.weight = torch.nn.Parameter(old_logit_weight)
# logit_bias = model.logit_bias
logit_bias = 0.0
top_logits =  emb_vec.to(model.logit_weight) @ model.logit_weight.T + logit_bias
print(model.tokenizer.batch_decode(top_logits.topk(10).indices.tolist()))
tokens = model.tokenizer([prompt] * batch_size, return_tensors="pt").to(model.device).input_ids
# tokens[:, token_index] = START_SPECIAL + torch.arange(batch_size, device=model.device)

state = (tokens, tokens, None)

def step(state):
    all_tokens, tokens, cache = state
    for m in model.model.modules():
        m._forward_hooks = OrderedDict()

    if tokens.shape[1] != 1:
        def patch(module, input, output):
            output[:, token_index] = emb_vec
            return output

        model.model.model.embed_tokens.register_forward_hook(patch)

    output = model.model(tokens, past_key_values=cache)
    logits = output.logits
    probs = torch.nn.functional.softmax(logits[:, -1], dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    all_tokens = torch.cat([all_tokens, next_tokens], dim=1)
    return (all_tokens, next_tokens, output.past_key_values)

try:
    for _ in (bar := tqdm(range(seq_len))):
        logger.disable("attribute")
        state = step(state)
        tokens = state[0]
        # tokens = torch.cat([tokens, next_tokens], dim=1)
        bar.set_postfix(text=model.tokenizer.decode(tokens[0].tolist()))
finally:
    logger.enable("attribute")
[model.tokenizer.decode(seq[len(tokenized_prompt):]) for seq in tokens.tolist()]
# %%
# %%

model.tokenizer.encode("a ?")
# %%
