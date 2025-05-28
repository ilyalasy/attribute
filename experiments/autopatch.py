#%%
%env CUDA_VISIBLE_DEVICES=0
%load_ext autoreload
%autoreload 2
from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, level="INFO")
from attribute.caching import TranscodedModel
from matplotlib import pyplot as plt
from attribute.mlp_attribution import AttributionGraph, AttributionConfig
import random
import torch
from collections import defaultdict
import numpy as np

# model_name = "roneneldan/TinyStories-33M"
# transcoder_path = "../e2e/checkpoints/clt-ts/test"
model_name = "nev/GELU_4L512W_C4_Code"
transcoder_path = "../e2e/checkpoints/gelu-4l-clt/k64"
# transcoder_path = "../e2e/checkpoints/gelu-4l-nonclt-skip/baseline"
# transcoder_path = "../e2e/checkpoints/gelu-4l-clt-noskip/baseline"
# transcoder_path = "../e2e/checkpoints/gelu-4l-nonclt/baseline"

model = TranscodedModel(
    model_name=model_name,
    transcoder_path=transcoder_path,
    device="cuda",
)
#%%
# prompt = "When John and Mary went to the store, John gave a bag to"
prompt = "<|BOS|>National Data Attribution Graph ("
og_activations = model(prompt)
# %%
config = AttributionConfig(
    name="test-1-ts",
    scan="default",
)
graph = AttributionGraph(model, og_activations, config)
graph.flow()
#%%
adj_matrix, dedup_node_names = graph.adjacency_matrix(normalize=False, absolute=False,)
#%%
random.seed(10)
reals = []
preds = []
for _ in range(1024):
    mlp_node_names = [n for n in dedup_node_names if n.startswith("intermediate_")]
    for _ in range(1024):
        a, b = random.sample(mlp_node_names, 2)
        a_s, a_l = map(int, a.split("_")[1:3])
        b_s, b_l = map(int, b.split("_")[1:3])
        if a_s > b_s:
            continue
        if a_l > b_l:
            continue
        if b_l - a_l != 2:
            continue
        source, target = a, b
        source_index, target_index = dedup_node_names.index(source), dedup_node_names.index(target)
        predicted_influence = adj_matrix[target_index, source_index]
        if predicted_influence != 0:
            break
    else:
        raise ValueError("No matching pair")
    source_seq, source_layer, source_feature = map(int, source.split("_")[1:4])
    target_seq, target_layer, target_feature = map(int, target.split("_")[1:4])

    source_k = og_activations.mlp_outputs[source_layer].location[0, source_seq].tolist().index(source_feature)
    source_act = og_activations.mlp_outputs[source_layer].source_activation

    target_k = og_activations.mlp_outputs[target_layer].location[0, target_seq].tolist().index(target_feature)
    target_act = og_activations.mlp_outputs[target_layer].activation[0, target_seq, target_k]

    real = torch.autograd.grad(target_act, source_act, retain_graph=True)[0][0, source_seq, source_k].item()
    pred = float(predicted_influence)

    reals.append(real)
    preds.append(pred)
from matplotlib import pyplot as plt
plt.scatter(reals, preds, s=1)
plt.plot([-1, 1], [-1, 1], "k--")
plt.xlabel("Automatic influence")
plt.ylabel("Manual influence")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()
# %%
